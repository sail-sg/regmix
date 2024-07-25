import contextlib
import gc
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union
from safetensors import safe_open
from safetensors.torch import save_file
import os
import torch
import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import NotYetLoadedTensor, incremental_save, lazy_load
# from scripts.convert_hf_checkpoint import layer_template, load_param


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
        print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param

def copy_weights_llama(
    config: Config,
    state_dict: Dict[str, torch.Tensor],
    lit_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
):
    weight_map = {
        "transformer.wte.weight": "model.embed_tokens.weight",
        "transformer.h.{}.norm_1.weight": "model.layers.{}.input_layernorm.weight",
        "transformer.h.{}.attn.proj.weight": "model.layers.{}.self_attn.o_proj.weight",
        "transformer.h.{}.norm_2.weight": "model.layers.{}.post_attention_layernorm.weight",
        "transformer.h.{}.mlp.swiglu.w1.weight": "model.layers.{}.mlp.gate_proj.weight",
        "transformer.h.{}.mlp.swiglu.w2.weight": "model.layers.{}.mlp.up_proj.weight",
        "transformer.h.{}.mlp.swiglu.w3.weight": "model.layers.{}.mlp.down_proj.weight",
        "transformer.ln_f.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for name, param in lit_weights.items():
        if name.endswith(".attn.attn.weight"):
            from_name, number = layer_template(name, 2)
            q = "model.layers.{}.self_attn.q_proj.weight".format(number)
            k = "model.layers.{}.self_attn.k_proj.weight".format(number)
            v = "model.layers.{}.self_attn.v_proj.weight".format(number)
            qkv = load_param(param, name, None)
            qp, kp, vp = tensor_split(qkv, config)
            for to_name, param in zip((q, k, v), (qp, kp, vp)):
                if saver is not None:
                    param = saver.store_early(param)
                state_dict[to_name] = param
        elif "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            
            if to_name is None:
                continue
            to_name = to_name.format(number)
            param = load_param(param, name,None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param

        else:
            to_name = weight_map[name]
            param = load_param(param, name, None)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param

def save_huggingface_config(config: Config, out_dir: Path) -> None:
    default_config_str = """{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 16,
  "num_hidden_layers": 22,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.35.0",
  "use_cache": true,
  "vocab_size": 50432
}
"""
    config_dict = json.loads(default_config_str)
    # modify the dict according to the config
    config_dict["hidden_size"] = config.n_embd
    config_dict["intermediate_size"] = config.intermediate_size
    config_dict["vocab_size"] = config.vocab_size
    config_dict["num_attention_heads"] = config.n_head
    config_dict["num_hidden_layers"] = config.n_layer
    config_dict["max_position_embeddings"] = config.block_size
    config_dict["rope_theta"] = config.rope_base
    # save the config to the output directory
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


def tensor_split(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def kstart(start, blen, klen) -> int:
        """returns start index of keys in batch"""
        return start + (blen - (klen * 2))

    def vstart(start, blen, klen) -> int:
        """returns start index of values in batch"""
        return start + blen - klen

    def vend(start, blen) -> int:
        """returns last index of values in batch"""
        return start + blen

    # num observations
    nobs = param.shape[0]
    # batch length
    blen = nobs // config.n_query_groups
    # key length in batch
    klen = config.head_size
    # value length in batch
    vlen = config.head_size
    # the starting index of each new batch
    starts = range(0, nobs, blen)
    # the indices to splice on
    splices = [(s, kstart(s, blen, klen), vstart(s, blen, vlen), vend(s, blen)) for s in starts]

    qc = ()
    kc = ()
    vc = ()

    for splice in splices:
        qs, ks, vs, ve = splice
        qc += (param[qs:ks, :],)
        kc += (param[ks:vs, :],)
        vc += (param[vs:ve, :],)

    q = torch.cat(qc)
    k = torch.cat(kc)
    v = torch.cat(vc)

    return q, k, v


def maybe_unwrap_state_dict(lit_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return lit_weights.get("model", lit_weights)


def check_conversion_supported(lit_weights: Dict[str, torch.Tensor]) -> None:
    weight_names = {wk.split(".")[-1] for wk in lit_weights}
    # LoRA or QLoRA
    if any("lora" in wn for wn in weight_names):
        raise ValueError("Model weights must be merged using `lora.merge_lora_weights()` before conversion.")
    # adapter v2. adapter_bias will only be in adapter_v2
    elif "adapter_bias" in weight_names:
        raise NotImplementedError("Converting models finetuned with adapter_v2 not yet supported.")
    # adapter. gating_factor is in adapter and adapter_v2
    elif "gating_factor" in weight_names:
        raise NotImplementedError("Converting models finetuned with adapter not yet supported.")


@torch.inference_mode()
def convert_lit_checkpoint(*, checkpoint_name: str, inp_dir: Path, out_dir: Path, model_name: str) -> None:
    config = Config.from_name(model_name)
    copy_fn = partial(copy_weights_llama, config)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # checkpoint_name cannot be hardcoded because there exists different outputs such as
    # ("lit_model_finetuned.pth", "lit_model_lora_finetuned.pth", "lit_model_adapter_finetuned.pth"")
    pth_file = inp_dir / checkpoint_name
    bin_file = out_dir / "model.safetensors"

    with incremental_save(bin_file) as saver:
        with contextlib.ExitStack() as stack:
            lit_weights = stack.enter_context(lazy_load(pth_file))
            lit_weights = maybe_unwrap_state_dict(lit_weights)
            check_conversion_supported(lit_weights)
            # Incremental save will trigger error
            copy_fn(sd, lit_weights, saver=None)
            gc.collect()
        # if there is any, remove the original checkpoint
        os.remove(bin_file)
        print(f"Saving model to {bin_file}")
        # use safe tensor to save the model
        save_file(sd, bin_file, metadata={"format": "pt"})
        # save the config
        save_huggingface_config(config, out_dir)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)
