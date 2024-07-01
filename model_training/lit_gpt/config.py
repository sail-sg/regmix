from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self

import lit_gpt.model
from lit_gpt.utils import find_multiple


@dataclass
class Config:
    org: str = "Lightning-AI"
    name: str = "lit-GPT"
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    rope_base: int = 10000
    bias: bool = True
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(lit_gpt.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":
            from lit_gpt.rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from lit_gpt.rmsnorm import FusedRMSNorm
            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)

#############################
# Sea AI Lab - RegMix Paper
#############################
regmix_llama = [
    dict(
        org="RegMix Paper",
        name="tinyllama_1M",
        block_size=2048,
        vocab_size=50432,
        padding_multiple=64,
        n_layer=2,
        n_head=8,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=512
    ),
    dict(
        org="RegMix Paper",
        name="tinycoder_1M",
        block_size=2048,
        vocab_size=49152,
        padding_multiple=64,
        n_layer=2,
        n_head=8,
        n_embd=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=512
    ),
   dict(
        org="RegMix Paper",
        name="tinyllama_60M",
        block_size=2048,
        vocab_size=50432,
        padding_multiple=64,
        n_layer=10,
        n_head=8,
        n_embd=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1536
    ),
    dict(
        org="RegMix Paper",
        name="tinyllama_1_1b",
        block_size=2048,
        vocab_size=50432,
        padding_multiple=64,
        n_layer=22,
        n_head=16,
        n_embd=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5, #Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=5632
    )
]
configs = regmix_llama
name_to_config = {config["name"]: config for config in configs}
