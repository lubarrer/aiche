import os
import argparse
from ray import tune


def evalfn(current_params):

    seed_everything(123)
    num_small = current_params.pop("num_small", None)
    num_medium = current_params.pop("num_medium", None)
    num_large = current_params.pop("num_large", None)
    npv = current_params.pop("npv", None)

    embed_params = current_params
    predict_params = {
        "num_small": num_small,
        "num_medium": num_medium,
        "num_large": num_large,
    }

    tune.report(val_score=npv)


if __name__ == "main":
    hyperopt = AxOptimizer(config)
    hyperopt.run(evalfn)
