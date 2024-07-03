# Regression Fitting

This directory contains the code for fitting the regression model and for predicting the optimal data mixture.

## Prepare Data

Before fitting the regression model, you need to prepare the data obtained after training the proxy models. If you have not trained the proxy models, please refer to the [mixture_config](../mixture_config/README.md) directory and the [model_training](../model_training/README.md) directory for more details.

In our paper, we use the validation loss on the Pile-CC subset as the **Target**, and the domain weights as the **Features** for regression model fitting. The already prepared data is stored in the [data](../data) directory. You can also prepare your own data by following the instructions in the [mixture_config](../mixture_config/README.md) directory.


## Model Fitting

You can follow the [notebook](regression.ipynb) to do both:
- Regression fitting with proxy model training logs
- Simulate and choose the optimal data mixture

With the notebook, you can easily fit the regression model and predict the optimal data mixture for training the large language models.