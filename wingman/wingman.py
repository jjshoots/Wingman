"""The core of Wingman."""
from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import wandb
import yaml

from .print_utils import cstr, wm_print


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
        config_yaml: str | Path,
        experiment_description: str = "",
    ):
        """__init__.

        Args:
            config_yaml (str): location of where the config yaml is described
            experiment_description (str): optional description of the experiment
        """
        # save our experiment description
        self.config_yaml: Path = Path(config_yaml)
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
        self.version_number = self.cfg.version_number
        self.mark_number = self.cfg.mark_number
        self.previous_mark_number = -1

        # directory itself
        self.version_directory = (
            Path(self.cfg.weights_directory) / f"Version{self.version_number}"
        )

        # file paths
        self.model_file = self.version_directory / f"weights{self.mark_number}.path"
        self.optim_file = self.version_directory / "optimizer_path"
        self.status_file = self.version_directory / "lowest_loss.npy"
        self.intermediary_file = self.version_directory / "weights-1.npy"
        if self.log_file:
            self.log_file = self.version_directory / "log.txt"

        wm_print("--------------𓆩𓆪--------------")
        wm_print(f"Using device {cstr(self.device, 'HEADER')}")
        wm_print(f"Saving weights to {cstr(self.version_directory, 'HEADER')}...")

        # check to record that we're in a new training session and save the config file
        self.fresh_directory = False
        if not self.version_directory.is_dir():
            self.fresh_directory = True
            wm_print(
                cstr(
                    "New training instance detected, generating weights directory in 3 seconds...",
                    "WARNING",
                ),
            )
            time.sleep(3)
            self.version_directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                self.config_yaml,
                self.version_directory / "config_copy.yaml",
            )

    def __get_device(self):
        """__get_device."""
        from warnings import warn

        try:
            import torch

            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        except ImportError:
            warn(
                "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
            )
            self.device = "Not Found!"

        return self.device

    def __yaml_to_args(self):
        """Reads the yaml file provided at init and converts it to commandline arguments."""
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
                "log_status",
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

            # make a logging file if required
            self.log_file = config["log_status"]

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

    def checkpoint(
        self, loss: float, step: int | None = None
    ) -> Tuple[bool, Path, Path]:
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
            Tuple[bool, Path, Path]: to_update, weights_file, optim_file
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
        wm_print(
            f"Step {cstr(step, 'OKCYAN')}; Average Loss {cstr(f'{avg_loss:.5f}', 'OKCYAN')}; Lowest Average Loss {cstr(f'{self.lowest_loss:.5f}', 'OKCYAN')}",
            self.log_file,
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
                wm_print(
                    f"Passed {self.cfg.max_skips} intervals without saving so far, saving weights to: {cstr(self.intermediary_file, 'OKCYAN')}",
                    self.log_file,
                )
                self.skips = 0
                return True, self.intermediary_file, self.optim_file

        """NEW LOSS IS BETTER"""
        # redefine the new lowest loss and reset the skips
        self.lowest_loss = avg_loss
        self.skips = 0

        # increment means return the files with incremented mark number
        if self.cfg.increment:
            # check if we are safe to increment mark numbers and regenerate weights file
            if self.previous_mark_number == -1 or self.model_file.exists():
                self.previous_mark_number = self.mark_number
                self.model_file = (
                    self.version_directory / f"weights{self.mark_number}.pth"
                )
                self.mark_number += 1

            else:
                wm_print(
                    cstr(
                        "Didn't save weights file for the previous mark number (self.mark_number), not incrementing mark number.",
                        "WARNING",
                    ),
                    self.log_file,
                )
                self.mark_number = self.previous_mark_number

        # record the lowest running loss in the status file
        np.save(self.status_file, self.lowest_loss)

        wm_print(
            f"New lowest point, saving weights to: {cstr(self.model_file, 'OKGREEN')}",
            self.log_file,
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
        filename = self.version_directory / f"{variable_name}.csv"
        with open(filename, "ab") as f:
            np.savetxt(f, [data], delimiter=",", fmt=precision)

    def get_weight_files(self, latest: bool = True) -> Tuple[bool, Path, Path]:
        """get_weight_files.

        Returns three things:
        - indicator on whether we have weight files
        - directory of where the weight files are
        - directory of where the optim files are

        Args:
            latest (bool): whether we want the latest file or the one determined by `mark_number`

        Returns:
            Tuple[bool, Path, Path]: have_file, weights_file, optim_file
        """
        # if we don't need the latest file, get the one specified
        if not latest:
            if os.path.isfile(self.model_file):
                self.model_file = (
                    self.version_directory / f"weights{self.mark_number}.pth"
                )
                wm_print(
                    f"Using weights file: {cstr(f'{self.version_directory}/weights{self.mark_number}.pth', 'OKGREEN')}",
                    self.log_file,
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
        while self.model_file.is_file():
            self.mark_number += 1
            self.model_file = self.version_directory / f"weights{self.mark_number}.pth"

        # once the file version doesn't exist, decrement by one and use that file
        self.mark_number = max(self.mark_number - 1, 0)
        self.model_file = self.version_directory / f"weights{self.mark_number}.pth"

        # if the file doesn't exist, notify and ignore
        if not self.model_file.is_file():
            if not self.fresh_directory:
                wm_print(
                    cstr(
                        "No weights file found, generating new one during training.",
                        "WARNING",
                    ),
                    self.log_file,
                )
            self.fresh_directory = False

            return False, self.model_file, self.optim_file
        else:
            # hitch a ride to update the lowest running loss
            self.lowest_loss = np.load(self.status_file).item()

            wm_print(
                f"Using weights file: {cstr(f'{self.version_directory}/weights{self.mark_number}.pth', 'OKGREEN')}",
                self.log_file,
            )
            wm_print(
                f"Lowest Running Loss for Net: {cstr(self.lowest_loss, 'OKCYAN')}",
                self.log_file,
            )

            # check if the optim file exists
            if not os.path.isfile(self.optim_file):
                wm_print(
                    cstr("Optim file not found, please be careful!", "WARNING"),
                    self.log_file,
                )

            return True, self.model_file, self.optim_file
