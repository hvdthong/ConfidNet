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

# USING ONLY FOR ADVERSARIAL EXAMPLES

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

    # load the dataset and attack type from adversarial examples
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--attack", "-attack", help="Define Attack Type", type=str, default="fgsm")

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.attack in ["fgsm", "bim-a", "bim-b", "bim", "jsma", "c+w"], "Attack we should used"
    print(args)

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

    if config_args["training"]["task"] == "segmentation":
        config_args["training"]["metrics"].append("mean_iou")

    # Special case of MC Dropout
    if args.mode == "mc_dropout":
        config_args["training"]["mc_dropout"] = True

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")
    learner = get_learner(
        config_args, dloader.train_loader, dloader.val_loader, dloader.test_loader, -1, device
    )

    # Initialize and load model
    ckpt_path = config_args["training"]["output_folder"] / f"model_epoch_{args.epoch:03d}.ckpt"
    checkpoint = torch.load(ckpt_path)
    learner.model.load_state_dict(checkpoint["model_state_dict"])

     # Get scores
    LOGGER.info(f"Inference mode: {args.mode}")

    if args.d == 'mnist' or args.d == 'cifar':
        adv = np.load('./data_adv/{}_{}.npy'.format(args.d, args.attack))    

    confidnet_adv = learner.evaluate_adv(adv, config_args=config_args)
    from test import write_file    
    write_file('./results/%s_adv_confidnet_epoch_%i_%s.txt' % (config_args['data']['dataset'], args.epoch, args.attack), confidnet_adv)
