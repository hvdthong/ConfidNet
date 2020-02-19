import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from confidnet.loaders import get_loader
from confidnet.learners import get_learner
from confidnet.models import get_model
from confidnet.utils import trust_scores
from confidnet.utils.logger import get_logger
from confidnet.utils.metrics import Metrics
from confidnet.utils.misc import load_yaml
import os

LOGGER = get_logger(__name__, level="DEBUG")

MODE_TYPE = ["confidnet"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml")
    parser.add_argument("--epoch", "-e", type=int, default=None, help="Epoch to analyse")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="normal",
        choices=MODE_TYPE,
        help="Type of confidence testing",
    )
    parser.add_argument(
        "--samples", "-s", type=int, default=50, help="Samples in case of MCDropout"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()

    config_args = load_yaml(args.config_path)

    # Overwrite for release
    config_args["training"]["output_folder"] = Path(args.config_path).parent

    config_args["training"]["metrics"] = [
        "accuracy",
        "auc",
        "ap_success",
        "ap_errors",
        "fpr_at_95tpr",
    ]

    print(config_args)