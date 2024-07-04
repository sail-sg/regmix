# 🧬 RegMix: Data Mixture as Regression for Language Model Pre-training

Welcome to the official repository of [RegMix](https://huggingface.co/papers/2407.01492), a new approach to optimizing data mixtures for large language model (LLM) pre-training!

## 🌟 What is RegMix?

RegMix is an innovative method that treats data mixture selection as a **regression task**. By training small "proxy" models on diverse data mixtures and analyzing their performance, RegMix builds a regression model that can predict the optimal data mixture for training large-scale LLMs.

![RegMix Method Visualization](misc/method_figure.png)

## 🚀 How RegMix Works

RegMix follows a four-step process to optimize LLM training:

1. **Generate Configs**: Create various different data mixture configurations.
2. **Train Small Models**: Use these configs to train small "proxy" models.
3. **Fit Regression Model**: Analyze the performance of these models (e.g., the validation loss on Pile-CC) to build a predictive regression model.
4. **Train Large Model**: Use the predicted optimal mixture to train a large-scale LLM.

## 🧰 What's in This Repo?

Our repository is organized into four main components:

1. [**mixture_config**](mixture_config): Tools for synthesizing and visualizing data mixtures.
   - Generate diverse data mixture configurations.
   - Visualize the generated mixtures.

2. [**regression_fitting**](regression_fitting): The heart of RegMix.
   - Fit the regression model using small model performance data, and by default we use the validation loss on the Pile-CC dataset.
   - Simulate and predict the optimal data mixture for large-scale training.

3. [**model_training**](model_training): Leveraging [TinyLlama](https://github.com/jzhang38/TinyLlama) for model training.
   - Train small (1M parameter) proxy models.
   - Scale up to large (1B+ parameter) language models using the predicted optimal mixture.

4. [**evaluation**](evaluation): [Work in Progress] Reproduce our evaluation results.


## 🛠 Applying RegMix on Your Dataset

Want to leverage the power of RegMix for your own language model training? Follow these steps to apply RegMix to your unique dataset:

### 1. Prepare Your Data

- **Organize Your Dataset**: Split your data into distinct categories or domains, with different unique prefix.
- **Format Requirements**: Ensure your data is in a compatible format (e.g., JSON lines, where each line is a valid JSON object containing the `text`).

An example dataset organization folder structure is as follows:

```
├── train
│   ├── domain1_0.jsonl
│   ├── domain1_1.jsonl
│   ├── ...
│   ├── domain2_0.jsonl
│   ├── domain2_1.jsonl
│   ├── ...
├── valid
│   ├── domain1.jsonl
│   ├── domain2.jsonl
```

> Please avoid making each jsonl file too large, as it may cause long time during the preprocessing.

You can also find the [regmix-data-sample](https://huggingface.co/datasets/sail/regmix) for your reference.

### 2. Generate Mixture Configurations

Use the tools in the `mixture_config/` directory to:
- Create a range of data mixture configurations.
- Visualize these mixtures to understand their composition.

Before generating configurations, ensure you have changed the function `get_token_distribution` in `synthesize_mixture.py` to match your dataset's domain names and token distributions.


### 3. Train Proxy Models

Utilize the `model_training/` scripts to train small "proxy" models on your generated mixtures:

```bash
python model_training/train_proxy.py --config_path /path/to/configs --model_size 1M
```

### 4. Fit the Regression Model

With your proxy model results, use the collect scripts to prepare the data for regression fitting. The first step is to organize the mixture configs into a CSV file:

```bash
python collect_mixture_data.py --write_file_path train_mixture_1m_your_dataset.csv --config_folder /path/to/configs
```

The second step is to collect the target performance data for the proxy models. By default we use the Pile-CC validation loss as the target, which is collected from wandb using the `wandb` API.

```bash
python collect_loss_data.py --write_file_path train_loss_1m_your_dataset.csv
```

Finally, fit the regression model using the collected data and predict the optimal mixture following the instructions in `regression_fitting/regression.ipynb`.

### 5. Train Your Large-Scale LLM

You can save your final predicted optimal mixture into a yaml file `optimal_mixture.yaml` as a similar format as the config under [mixture_config/config_1b](mixture_config/config_1b). An example of the optimal mixture is as follows:

```yaml
train:
  train_the_pile_arxiv: 0.0012046169821426883
  train_the_pile_freelaw: 0.001454510048554701
  train_the_pile_nih_exporter: 0.001231640306882902
  train_the_pile_pubmed_central: 0.003108561825532002
  train_the_pile_wikipedia_en: 0.01593264140324679
  train_the_pile_dm_mathematics: 0.00031106907908634156
  train_the_pile_github: 0.00022861228152440253
  train_the_pile_philpapers: 1.329107360676338e-05
  train_the_pile_stackexchange: 0.00029547405933203174
  train_the_pile_enron_emails: 0.0016691646199353991
  train_the_pile_gutenberg_pg_19: 0.001612531300038395
  train_the_pile_pile_cc: 0.8701291419934237
  train_the_pile_ubuntu_irc: 0.06417728505869834
  train_the_pile_europarl: 2.9166170357771267e-06
  train_the_pile_hackernews: 0.011925517591888925
  train_the_pile_pubmed_abstracts: 0.02424425081714838
  train_the_pile_uspto_backgrounds: 0.0024587749419225434
valid:
  valid_the_pile_pile_cc: 1.0
model_name: tinyllama_1_1b
```

Finally, use the predicted optimal mixture to train your model via `optimal_mixture.yaml`:

```bash
cd model_training
./pretrain_tinyllama_1b.sh optimal_mixture
```

### Tips for Success

- **Data Diversity**: Ensure your initial dataset covers a wide range of domains.
- **Proxy Model Size**: While we use 1M parameter models, you might need to adjust based on your computational resources and dataset size.
- **The Number of Runs**: If you have the resources, consider training more proxy models to improve the regression model's accuracy, especially when you have a large number of domains.
- **Evaluation**: Choosing the correct target is crucial for the generic downstream performance improvement. You may want to use the loss on a high-quality and diverse validation dataset like Pile-CC. We also recommend using the awesome [paloma evaluation suite](https://huggingface.co/datasets/allenai/paloma) from AI2 for evaluation.

### Customization Options

RegMix is flexible and can be adapted to your specific needs:
- Adjust the number and size of proxy models.
- Modify the regression model architecture or features.
- Incorporate domain-specific metrics in your optimization objective.

Remember, the key to RegMix's success is in capturing the relationship between data mixture and model performance. The more informative your proxy training runs are, the better your final mixture prediction will be!

## 📦 Data and Model Release

We've made our data and trained models available on HuggingFace! Check out the following table for links:

[Table to be added here]

## 🔍 Evaluation [Work in Progress]

Stay tuned! We're currently working on providing comprehensive evaluation setup in the `evaluation` directory.

## 📚 Citation

If RegMix helps your research, please cite our paper:

```bibtex
@misc{liu2024regmix,
      title={RegMix: Data Mixture as Regression for Language Model Pre-training}, 
      author={Qian Liu and Xiaosen Zheng and Niklas Muennighoff and Guangtao Zeng and Longxu Dou and Tianyu Pang and Jing Jiang and Min Lin},
      year={2024},
      eprint={2407.01492},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01492}, 
}
```

## 🤝 Get in Touch

Excited about RegMix? Have questions? We'd love to hear from you!

Contact us at:
- liuqian@sea.com
- xszheng.2020@phdcs.smu.edu.sg

Join us in scalable and efficient data mixture with RegMix!