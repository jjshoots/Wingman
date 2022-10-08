#!/usr/bin/env python3
import argparse
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
import yaml


class Wingman:
    """Wingman.

    Class to handle checkpointing of loss and handling of weights files.
    Minimal example:

    ```py
    helper = Wingman(./config.yaml)
    cfg = helper.cfg

    # load the model and optimizer
    model = Model().to(cfg.device)
    optim = optimizer.AdamW(model.parameters(), lr=cfg.learning_rate, amsgrad=True)

    # get the weight files if they exist
    have_file, weight_file, optim_file = self.get_weight_files()
    if have_file:
        model.load(model_file)
        optim.load(optim_file)

    # run some training:
    while(training):
        ...
        # training code here
        ...

        update_weights, model_file, optim_file = self.checkpoint(loss, batch_number, epoch_number)
        if update_weights:
            model.save(model_file)
            optim.save(optim_file)
    ```
    """

    def __init__(
        self,
        config_yaml: str,
        experiment_description: str = "",
    ):
        """__init__.

        Args:
            config_yaml (str): location of where the config yaml is described
            experiment_description (str): optional description of the experiment
        """
        # save our experiment description
        self.config_yaml = config_yaml
        self.experiment_description = experiment_description
        self.cfg = self.__yaml_to_args()

        # make sure that either only epoch or batch interval is set
        assert (
            self.cfg.epoch_interval * self.cfg.batch_interval < 0
        ), "epoch_interval or batch_interval must be positive number"

        # runtime variables
        self.iter_passed = 0
        self.cumulative_loss = 0
        self.lowest_loss = math.inf
        self.previous_save_step = 0
        self.skips = 0

        # the logger
        self.log = dict()

        # minimum required before new weight file is made
        self.greater_than = self.cfg.greater_than

        # the interval before we save things
        self.interval = (
            self.cfg.epoch_interval
            if self.cfg.epoch_interval > 0
            else self.cfg.batch_interval
        )

        # maximum skips allowed before save to intermediary
        self.max_skips = self.cfg.max_skips

        # weight file variables
        self.directory = os.path.dirname(__file__)
        self.version_number = self.cfg.version_number
        self.mark_number = self.cfg.mark_number

        # directory itself
        self.version_directory = os.path.join(
            self.cfg.weights_directory, f"Version{self.version_number}"
        )
        self.version_dir_print = self.version_directory.split("/")[-2]

        # file paths
        self.model_file = os.path.join(
            self.version_directory,
            f"weights{self.mark_number}.pth",
        )
        self.optim_file = os.path.join(
            self.version_directory,
            "optimizer.pth",
        )
        self.status_file = os.path.join(
            self.version_directory,
            "lowest_loss.npy",
        )
        self.intermediary_file = os.path.join(
            self.version_directory,
            "weights_intermediary.pth",
        )

        print(
            "\n----------------------------------------------------------------------\n"
        )
        print(f"Using Device {self.device}")
        print(f"Saving weights to {self.version_directory}...")

        # record that we're in a new training session
        if not os.path.isfile(
            # new training session when there's no training log
            os.path.join(self.version_directory, "training_log.txt")
        ):
            print("Weights directory not found, generating new one in 3 seconds...")
            time.sleep(3)
            os.makedirs(self.version_directory)

        with open(
            os.path.join(self.version_directory, "training_log.txt"),
            "a",
        ) as f:
            f.write(f"New Session, Net Version {self.version_number} \n")
            f.write("Epoch, Batch, Running Loss, Lowest Running Loss, Mark Number \n")

    def __get_device(self):
        """__get_device."""
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        self.device = device
        return device

    def __yaml_to_args(self):
        """__yaml_to_args.

        Reads the yaml file provided at init and converts it to commandline arguments.

        """
        # parse the arguments
        parser = argparse.ArgumentParser(description=self.experiment_description)

        with open(self.config_yaml) as f:

            # read in the file
            config = yaml.load(f, Loader=yaml.FullLoader)

            # checks that we have the default params in the file
            assertation_list = [
                "debug",
                "weights_directory",
                "version_number",
                "mark_number",
                "increment",
                "epoch_interval",
                "batch_interval",
                "max_skips",
                "greater_than",
                "wandb",
                "wandb_name",
                "wandb_notes",
                "wandb_id",
                "wandb_entity",
                "wandb_project",
            ]
            for item in assertation_list:
                assert item in config, f"Missing parameter {item} in config file."

            for item in config:
                # exclusively for version number only
                if item == "version_number":
                    if config[item] is None:
                        config[item] = np.random.randint(999999)

                parser.add_argument(
                    f"--{item}",
                    type=type(config[item]),
                    nargs="?",
                    const=True,
                    default=config[item],
                    help="None",
                )

        # dict of all arguments, by default will be overridden by commandline args
        config = {**vars(parser.parse_args())}

        # add the gpu
        config["device"] = self.__get_device()

        # change the version if debugging
        config["version_number"] = (
            "Debug" if config["debug"] else str(config["version_number"])
        )

        # cfg depending on whether wandb is enabled
        cfg = None
        if config["wandb"]:
            wandb.init(
                project=config["wandb_project"],
                entity=config["wandb_entity"],
                config=config,
                name=config["wandb_name"] + ", v=" + config["version_number"]
                if config["wandb_name"] != ""
                else config["version_number"],
                notes=config["wandb_notes"],
                id=config["wandb_id"] if config["wandb_id"] != "" else None,
            )

            # also save the code if wandb
            wandb.run.log_code(".", exclude_fn=lambda path: "venv" in path)  # type: ignore

            # set to be consistent with wandb config
            cfg = wandb.config
        else:
            # otherwise just merge settings with args
            cfg = argparse.Namespace(**config)

        return cfg

    def checkpoint(self, loss: float, batch: int, epoch: int) -> Tuple[bool, str, str]:
        """checkpoint.

        Depending on whether epoch_interval or batch_interval is used,
        records training every epoch/batch steps.

        Returns three things:
        - indicator on whether we should save weights
        - directory of where the weight files should be saved
        - directory of where the optim files should be saved

        Args:
            loss (float): learning loss of the model as a detached float
            batch (int): batch number
            epoch (int): epoch number

        Returns:
            Tuple[bool, Optional[str], Optional[str]]:
        """
        # indicator on whether we need to save the weights
        update = False

        step = epoch if self.cfg.epoch_interval > 0 else batch

        # this is triggered when n intervals have passed
        if step % self.interval == 0:
            self.cumulative_loss = loss
            self.iter_passed = 1.0

            # perform the wandb log here
            if self.cfg.wandb:
                wandb.log(self.log)
        else:
            self.cumulative_loss += loss
            self.iter_passed += 1.0

        # check if n intervals have passed and it is not the first interval, we save here
        if step % self.interval == 0 and step != 0 and step != self.previous_save_step:
            self.previous_save_step = step

            # we use the loss per step as an indicator
            avg_loss = self.cumulative_loss / self.iter_passed

            # always print on n intervals
            print(
                f"Epoch {epoch}; Batch Number {batch}; Running Loss {avg_loss:.5f}; Lowest Running Loss {self.lowest_loss:.5f}"
            )

            # record training log
            with open(
                os.path.join(
                    self.version_directory,
                    "training_log.txt",
                ),
                "a",
            ) as f:
                f.write(
                    f"{epoch}, {batch}, {self.cumulative_loss}, {self.lowest_loss}, {self.mark_number} \n"
                )

            # record the lowest running loss in the status file
            np.save(self.status_file, self.lowest_loss)

            # save the network if the running loss is lower than the one we have
            if avg_loss < self.lowest_loss:
                # redefine the new running loss
                self.lowest_loss = avg_loss

                # reset the number of skips
                self.skips = 0

                # increment the mark number
                if self.cfg.increment:
                    self.mark_number += 1

                    # regenerate the weights_file path
                    self.model_file = os.path.join(
                        self.version_directory,
                        f"weights{self.mark_number}.pth",
                    )

                print(
                    f"New lowest point, saving weights to: {self.version_dir_print}/weights{self.mark_number}.pth"
                )

                update = True
            else:
                # if we don't get a new running loss, record that we skipped one interval
                self.skips += 1

            # save the network to intermediary if we crossed the max number of skips
            if self.skips >= self.max_skips:
                self.skips = 0
                print(
                    f"Passed {self.max_skips} intervals without saving so far, saving weights to: /weights_intermediary.pth"
                )

                return True, self.intermediary_file, self.optim_file

        return update, self.model_file, self.optim_file

    def write_auxiliary(
        self, data: np.ndarray, variable_name: str, precision: str = "%1.3f"
    ) -> None:
        """write_auxiliary.

        Args:
            data (np.ndarray): data
            variable_name (str): variable_name
            precision (str): precision

        Returns:
            None:
        """
        assert len(data.shape) == 1, "Data must be only 1 dimensional ndarray"
        filename = os.path.join(self.version_directory, f"{variable_name}.csv")
        with open(filename, "ab") as f:
            np.savetxt(f, [data], delimiter=",", fmt=precision)

    def get_weight_files(
        self, latest: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """get_weight_files.

        Returns three things:
        - indicator on whether we have weight files
        - directory of where the weight files are
        - directory of where the optim files are

        Args:
            latest (bool): whether we want the latest file or the one determined by `mark_number`

        Returns:
            Tuple[bool, Optional[str], Optional[str]]:
        """
        have_file = False

        # if we don't need the latest file, get the one specified
        if not latest:
            if os.path.isfile(self.model_file):
                self.model_file = os.path.join(
                    self.version_directory,
                    f"weights{self.mark_number}.pth",
                )
                have_file = True
                return have_file, self.model_file, self.optim_file

        # while the file exists, try to look for a file one version later
        while os.path.isfile(self.model_file):
            self.mark_number += 1
            self.model_file = os.path.join(
                self.version_directory,
                f"weights{self.mark_number}.pth",
            )

        # once the file version doesn't exist, decrement by one and use that file
        self.mark_number -= 1 if self.mark_number > 0 else 0
        self.model_file = os.path.join(
            self.version_directory,
            f"weights{self.mark_number}.pth",
        )

        # if there's no files, ignore, otherwise, print the file
        if os.path.isfile(self.model_file):
            # hitch a ride to update the lowest running loss
            self.lowest_loss = np.load(self.status_file).item()

            print(
                f"Using weights file: /{self.version_dir_print}/weights{self.mark_number}.pth"
            )

            print(f"Lowest Running Loss for Net: {self.lowest_loss}")

            have_file = True
        else:
            print("No weights file found, generating new one during training.")
            have_file = False

        # check if the optim file exists
        if not os.path.isfile(self.optim_file):
            print("Optim file not found, please be careful!")

        # return depending on whether we've found the file
        if have_file:
            return have_file, self.model_file, self.optim_file
        else:
            return have_file, None, None
