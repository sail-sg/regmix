import numpy as np
import random
import yaml

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Temperature for the prior distribution
TEMP = 0.1
# The minimum and maximum strength for the dirichlet distribution
MIN_STRENGH = 1.0
MAX_STRENGH = 5.0
# We first sample SAMPLE_MULTIPLIER times more samples than randomly select some of them
SAMPLE_MULTIPLIER = 100
# How many repeated epochs are allowed for each domain
MAXIMUM_USAGE = 15
# Assume that we have 512 x 1000 samples, 2e-4 will have 10 samples which have the statistical significance
MINIMUM = 2e-4

def generate_train_group(groups, weights, sample_folder=None, precision=5):
    """
    Generate a formatted string of groups and their corresponding weights.

    Args:
    groups (list): List of group names.
    weights (list): List of corresponding weights.
    sample_folder (str, optional): If provided, will be included in the group name.
    prefix (str, optional): Prefix to add before each group name. Defaults to 'train'.
    precision (int, optional): Number of decimal places for rounding weights. Defaults to 4.

    Returns:
    str: Formatted string of groups and weights.
    """
    assert len(groups) == len(weights), "Length of groups and weights must be equal"
    
    def format_weight(weight):
        return f"{weight:.{precision}f}".rstrip('0').rstrip('.')
    
    if sample_folder:
        output_group = [f"  train_{sample_folder}_{group}: {format_weight(num)}" 
                        for group, num in zip(groups, weights)]
    else:
        output_group = [f"  train_{group}: {format_weight(num)}" 
                        for group, num in zip(groups, weights)]
    
    return "\n".join(output_group)

def generate_valid_group(groups, sample_folder=None):
    weights = [1.0] * len(groups)
    if sample_folder is None:
        output_group = [f"  {group}: {num}" for group, num in zip(groups, weights)]
    else:
        output_group = [f"  valid_{sample_folder}_{group}: {num}" for group, num in zip(groups, weights)]
    return "\n".join(output_group)


def generate_weights_exponential(prior_dist,
                                 minimum_number,
                                 num_samples=128, 
                                 enable_bound=True,
                                 temperature=1.0):

    final_samples = []
    
    if enable_bound:
        # generate the bound for reject sampling
        number_bound = []
        for i in range(len(prior_dist)):
            # the token cannot be used more than 4 times
            number_bound.append([0.0,
                                 prior_dist[i] * MAXIMUM_USAGE])
    else:
        number_bound = None
    print("The usage bound is: ", number_bound)

    # apply temperature
    if temperature < 1.0:
        prior_dist = prior_dist ** TEMP
        prior_dist = prior_dist / np.sum(prior_dist)
        print("After temperature: ", prior_dist)

    # combine reject sampling with dirichlet distribution
    for i in range(num_samples * SAMPLE_MULTIPLIER):
        if MIN_STRENGH == MAX_STRENGH:
            samples = np.random.dirichlet(prior_dist * MIN_STRENGH, 1)
        else:
            samples = []
            min_strength_log = np.log10(MIN_STRENGH)
            max_strength_log = np.log10(MAX_STRENGH)
            for strength in np.logspace(min_strength_log, max_strength_log, 10):
                # add a noise to the strength
                samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)
                samples.append(samples_per_strength)
            # random sample one
            samples = random.choice(samples)
        # if there is a bound, the bound is a list of tuples indicating the lower and upper bound of each group
        ensure_flag = True
        if number_bound is not None:
            for j in range(len(samples[0])):
                if samples[0][j] < number_bound[j][0] or samples[0][j] > number_bound[j][1]:
                    ensure_flag = False
                    break
        if ensure_flag is False:
            continue
        # post normalization, set zero for the number less than minimum_number
        samples = np.where(samples < minimum_number, 0.0, samples)
        # round samples into the same scale of minimum_number
        samples = samples / np.sum(samples, axis=1).reshape(-1, 1)
        samples = np.round(samples / minimum_number) * minimum_number
        # add the samples to the final_samples
        final_samples.append(samples[0])

    # remove the samples with the nearly same values
    print("The number of final samples: ", len(final_samples))        
    selected_samples = random.sample(final_samples, num_samples)
    print("The number of selected samples: ", len(selected_samples))
    selected_samples = np.stack(selected_samples, axis=0)
    return selected_samples


def generate_config_from_prior(output_paths, prior_yaml):
    number_of_samples = len(output_paths)
    # read the yaml file and get the prior distribution
    if prior_yaml is not None:
        # using yaml to load the prior distribution
        prior_config = yaml.load(open(prior_yaml, "r", encoding="utf8"), Loader=yaml.FullLoader)
        train_config = prior_config["train"]
        train_groups, prior_dist = [], []
        for k, v in train_config.items():
            train_groups.append(k)
            prior_dist.append(v)

        print("Before normalization: ", prior_dist)
        # renormalize the prior distribution
        prior_dist = prior_dist / np.sum(prior_dist)
        print("After normalization: ", prior_dist)

        valid_config = prior_config["valid"]
        valid_groups = list(valid_config.keys())
        
        train_weights = generate_weights_exponential(prior_dist, MINIMUM,
                                                     number_of_samples,
                                                     temperature=TEMP)

    else:
        print("please input the prior distirbution")
        return

    for output_path, weights in zip(output_paths, train_weights):
        train_group = generate_train_group(train_groups, weights, sample_folder=None)
        valid_group = generate_valid_group(valid_groups, sample_folder=None)
        with open(output_path, "w", encoding="utf8") as f:
            f.write("train:\n")
            f.write(train_group)
            f.write("\n")
            f.write("valid:\n")
            f.write(valid_group)
            f.write("\n")
            f.write(f"seed: {SEED}\n")
            f.write(f"temperature: {TEMP}\n")
            f.write(f"min_strength: {MIN_STRENGH}\n")
            f.write(f"max_strength: {MAX_STRENGH}\n")
            f.write(f"minimum: {MINIMUM}\n")
            f.write(f"sample_multiplier: {SAMPLE_MULTIPLIER}\n")
            f.write(f"maximum_usage: {MAXIMUM_USAGE}\n")
            
            # these are configurations for the model
            content = ""
            content += "\n" + "model_name: tinyllama_1M"
            # content += "\n" + "model_name: tinycoder_1M"
            content += "\n" + "total_devices: 1"
            content += "\n" + "num_of_devices: 1"
            content += "\n" + "global_batch_size: 512"
            content += "\n" + "micro_batch_size: 16"
            # 1001 instead of 1000 because wandb has the bug of not showing the last step
            content += "\n" + "max_step: 1001"
            # never save the model, just using the wandb log
            content += "\n" + "save_step_interval: 2000"
            content += "\n" + "eval_step_interval: 100"
            # constant learning rate for the small model
            content += "\n" + "learning_rate: 0.0004"
            content += "\n" + "min_lr: 0.0004"
            # the warmup step is 50
            content += "\n" + "warmup_steps: 100"
            f.write(content)


if __name__ == "__main__":
    output_paths = []
    for i in range(1, 513):
        output_paths.append(f"config_1m/n{i}.yaml")
        
    generate_config_from_prior(output_paths,
                               prior_yaml="token_prior.yaml")
    