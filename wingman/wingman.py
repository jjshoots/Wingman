#!/usr/bin/env python3
import argparse
import math
import os
import time
from typing import Tuple

import numpy as np
import torch
import wandb
import yaml

from .print_utils import cstr


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

        update_weights, model_file, optim_file = self.checkpoint(loss, step_number)
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

        # make sure that logging_interval is positive
        assert (
            self.cfg.logging_interval > 0
        ), f"logging_interval must be a positive number, got {self.cfg.logging_interval}."

        # the logger
        self.log = dict()

        # runtime variables
        self.num_losses = 0
        self.cumulative_loss = 0
        self.lowest_loss = math.inf
        self.next_log_step = self.cfg.logging_interval
        self.previous_checkpoint_step = 0
        self.skips = 0

        # weight file variables
        self.directory = os.path.dirname(__file__)
        self.version_number = self.cfg.version_number
        self.mark_number = self.cfg.mark_number
        self.previous_mark_number = -1

        # directory itself
        self.version_directory = os.path.join(
            self.cfg.weights_directory, f"Version{self.version_number}"
        )

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
            "weights-1.pth",
        )

        print("--------------𓆩𓆪--------------")
        print(f"Using device {cstr(self.device, 'HEADER')}")
        print(f"Saving weights to {cstr(self.version_directory, 'HEADER')}...")

        # check to record that we're in a new training session
        self.fresh_directory = False
        if not os.path.isdir(self.version_directory):
            self.fresh_directory = True
            print(
                cstr(
                    "New training instance detected, generating weights directory in 3 seconds...",
                    "WARNING",
                )
            )
            time.sleep(3)
            os.makedirs(self.version_directory)

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
                "logging_interval",
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
                assert item in config, cstr(
                    f"Missing parameter {item} in config file.", "FAIL"
                )

            # override version number if needed
            if config["version_number"] is None:
                config["version_number"] = np.random.randint(999999)

            # add all to argparse
            for item in config:
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
            # wandb.run.log_code(".", exclude_fn=lambda path: "venv" in path)  # type: ignore

            # set to be consistent with wandb config
            cfg = wandb.config
        else:
            # otherwise just merge settings with args
            cfg = argparse.Namespace(**config)

        return cfg

    def checkpoint(self, loss: float, step: int | None = None) -> Tuple[bool, str, str]:
        """checkpoint.

        Records training every logging_interval steps.

        Returns three things:
        - indicator on whether we should save weights
        - directory of where the weight files should be saved
        - directory of where the optim files should be saved

        Args:
            loss (float): learning loss of the model as a detached float
            step (int | None): step number, automatically incremented if None

        Returns:
            Tuple[bool, str, str]: to_update, weights_file, optim_file
        """
        # if step is None, we automatically increment
        if step is None:
            step = self.previous_checkpoint_step + 1

        # check that our step didn't go in reverse
        assert step >= self.previous_checkpoint_step, cstr(
            f"We can't step backwards! Got step {step} but the previous logging step was {self.previous_checkpoint_step}.",
            "FAIL",
        )
        self.previous_checkpoint_step = step

        """ACCUMULATE LOSS"""
        self.cumulative_loss += loss
        self.num_losses += 1.0

        # if we haven't passed the required number of steps
        if step < self.next_log_step:
            return False, self.model_file, self.optim_file

        # log to wandb if needed, but only on the logging steps
        if self.cfg.wandb:
            self.wandb_log()

        """GET NEW AVG LOSS"""
        # record the losses and reset the cumulative
        avg_loss = self.cumulative_loss / self.num_losses
        self.cumulative_loss = 0.0
        self.num_losses = 0.0

        # compute the next time we need to log
        self.next_log_step = (
            int(step / self.cfg.logging_interval) + 1
        ) * self.cfg.logging_interval

        # always print on n intervals
        print(
            f"Step {cstr(step, 'OKCYAN')}; Average Loss {cstr(f'{avg_loss:.5f}', 'OKCYAN')}; Lowest Average Loss {cstr(f'{self.lowest_loss:.5f}', 'OKCYAN')}"
        )

        """CHECK IF NEW LOSS IS BETTER"""
        # if we don't meet the criteria for saving
        if avg_loss >= self.lowest_loss - self.cfg.greater_than:
            # accumulate skips
            if self.skips < self.cfg.max_skips:
                self.skips += 1
                return False, self.model_file, self.optim_file
            else:
                # save the network to intermediary if we crossed the max number of skips
                print(
                    f"Passed {self.cfg.max_skips} intervals without saving so far, saving weights to: {cstr(self.intermediary_file, 'OKCYAN')}"
                )
                self.skips = 0
                return True, self.intermediary_file, self.optim_file

        """NEW LOSS IS BETTER"""
        # redefine the new lowest loss and reset the skips
        self.lowest_loss = avg_loss
        self.skips = 0

        # no increment means just return the files
        if not self.cfg.increment:
            return True, self.model_file, self.optim_file

        # check if we are safe to increment mark numbers and regenerate weights file
        if self.previous_mark_number == -1 or os.path.isfile(self.model_file):
            self.previous_mark_number = self.mark_number
            self.model_file = os.path.join(
                self.version_directory,
                f"weights{self.mark_number}.pth",
            )
            self.mark_number += 1

        else:
            print(
                cstr(
                    "Didn't save weights file for the previous mark number (self.mark_number), not incrementing mark number.",
                    "WARNING",
                )
            )
            self.mark_number = self.previous_mark_number

        # record the lowest running loss in the status file
        np.save(self.status_file, self.lowest_loss)

        print(
            f"New lowest point, saving weights to: {cstr(self.model_file, 'OKGREEN')}"
        )

        return True, self.model_file, self.optim_file

    def wandb_log(self) -> None:
        """wandb_log.

        Logs the internal log to WandB.
        Start logging by adding things to it using:
        ```
        wm.log["foo"] = "bar"
        ```

        Returns:
            None:
        """
        assert isinstance(self.log, dict), cstr(
            f"log must be dictionary, currently it is {self.log}.", "FAIL"
        )
        if self.cfg.wandb:
            wandb.log(self.log)

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

    def get_weight_files(self, latest: bool = True) -> Tuple[bool, str, str]:
        """get_weight_files.

        Returns three things:
        - indicator on whether we have weight files
        - directory of where the weight files are
        - directory of where the optim files are

        Args:
            latest (bool): whether we want the latest file or the one determined by `mark_number`

        Returns:
            Tuple[bool, str, str]: have_file, weights_file, optim_file
        """
        # if we don't need the latest file, get the one specified
        if not latest:
            if os.path.isfile(self.model_file):
                self.model_file = os.path.join(
                    self.version_directory,
                    f"weights{self.mark_number}.pth",
                )
                return True, self.model_file, self.optim_file
            else:
                raise ValueError(
                    cstr(
                        f"Mark number {self.mark_number} was requested, but it doesn't exist.",
                        "FAIL",
                    )
                )

        # while the file exists, try to look for a file one version later
        self.mark_number = 0
        while os.path.isfile(self.model_file):
            self.mark_number += 1
            self.model_file = os.path.join(
                self.version_directory,
                f"weights{self.mark_number}.pth",
            )

        # once the file version doesn't exist, decrement by one and use that file
        self.mark_number = max(self.mark_number - 1, 0)
        self.model_file = os.path.join(
            self.version_directory,
            f"weights{self.mark_number}.pth",
        )

        # if the file doesn't exist, notify and ignore
        if not os.path.isfile(self.model_file):
            if not self.fresh_directory:
                print(
                    cstr(
                        "No weights file found, generating new one during training.",
                        "WARNING",
                    )
                )
            self.fresh_directory = False

            return False, self.model_file, self.optim_file
        else:
            # hitch a ride to update the lowest running loss
            self.lowest_loss = np.load(self.status_file).item()

            print(
                f"Using weights file: {cstr(f'{self.version_directory}/weights{self.mark_number}.pth', 'OKGREEN')}"
            )
            print(f"Lowest Running Loss for Net: {cstr(self.lowest_loss, 'OKCYAN')}")

            # check if the optim file exists
            if not os.path.isfile(self.optim_file):
                print(cstr("Optim file not found, please be careful!", "WARNING"))

            return True, self.model_file, self.optim_file
