import argparse
import os
from shutil import copyfile, rmtree

import click
import torch

from confidnet.loaders import get_loader
from confidnet.learners import get_learner
from confidnet.utils.logger import get_logger
from confidnet.utils.misc import load_yaml
from confidnet.utils.tensorboard_logger import TensorboardLogger


LOGGER = get_logger(__name__, level="DEBUG")

def train(args, config_args, config_path):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    LOGGER.info("Starting from scratch")
    if not os.path.exists(config_args["training"]["output_folder"]):
        os.makedirs(config_args["training"]["output_folder"])    
    start_epoch = 1

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")    
    dloader = get_loader(config_args)

    # Make loaders
    dloader.make_loaders()

    # Set learner
    LOGGER.warning(f"Learning type: {config_args['training']['learner']}")
    learner = get_learner(
        config_args,
        dloader.train_loader,
        dloader.val_loader,
        dloader.test_loader,
        start_epoch,
        device,
    )

    # Log files
    LOGGER.info(f"Using model {config_args['model']['name']}")
    learner.model.print_summary(
        input_size=tuple([shape_i for shape_i in learner.train_loader.dataset[0][0].shape])
    )
    learner.tb_logger = TensorboardLogger(config_args["training"]["output_folder"])
    copyfile(
        config_path, config_args["training"]["output_folder"] / f"config_{start_epoch}.yaml"
    )
    LOGGER.info(
        "Sending batches as {}".format(
            tuple(
                [config_args["training"]["batch_size"]]
                + [shape_i for shape_i in learner.train_loader.dataset[0][0].shape]
            )
        )
    )
    LOGGER.info(f"Saving logs in: {config_args['training']['output_folder']}")

    # Parallelize model
    nb_gpus = torch.cuda.device_count()
    if nb_gpus > 1:
        LOGGER.info(f"Parallelizing data to {nb_gpus} GPUs")
        learner.model = torch.nn.DataParallel(learner.model, device_ids=range(nb_gpus))

    # Set scheduler
    learner.set_scheduler()

    # Start training
    for epoch in range(start_epoch, config_args["training"]["nb_epochs"] + 1):
        learner.train(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument("--d", "-d", type=str, default=None, help="Dataset (i.e., mnist or cifar)")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--s", "-s", help="Start times of random sampling", type=int, default=0)
    parser.add_argument("--e", "-e", help="End times of random sampling", type=int, default=100)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    for t in range(args.s, args.e):
        config_args = load_yaml('./random_confs/%s_selfconfid_%i.yaml' % (args.d, t))
        config_path = './random_confs/%s_selfconfid_%i.yaml' % (args.d, t)
        train(args, config_args, config_path)

        


