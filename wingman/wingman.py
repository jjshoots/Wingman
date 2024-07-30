"""The core of Wingman."""

from __future__ import annotations

import math
import os
import shutil
import time
from functools import cached_property
from pathlib import Path

import numpy as np
import wandb

from wingman.config_utils import LockedNamespace, generate_wingman_config
from wingman.exceptions import WingmanException
from wingman.print_utils import cstr, wm_print


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
    ):
        """__init__.

        Args:
        ----
            config_yaml (str): location of where the config yaml is described

        """
        # save our experiment description
        self.cfg: LockedNamespace = generate_wingman_config(config_yaml)

        # make sure that logging_interval is positive
        if self.cfg.logging.interval <= 0:
            raise WingmanException(
                cstr(
                    f"logging_interval must be a positive number, got {self.cfg.logging.interval}.",
                    "FAIL",
                )
            )

        # the logger
        self.log = dict()

        # runtime variables
        self._num_losses: int = 0
        self._cumulative_loss: float = 0.0
        self._lowest_cumulative_lost: float = math.inf
        self._next_log_step: int = self.cfg.logging.interval
        self._previous_ckpt_step: int = 0
        self._skips: int = 0

        # weight file variables
        self._current_ckpt: int = self.cfg.model.ckpt
        self._previous_ckpt: int = -1

        # directory itself
        self._model_directory: Path = Path(self.cfg.model.save_directory) / str(
            self.cfg.model.id
        )

        # file paths
        self._model_file: Path = (
            self._model_directory / f"weights{self._current_ckpt}.path"
        )
        self._optim_file: Path = self._model_directory / "optimizer_path"
        self._lowest_loss_file: Path = self._model_directory / "lowest_loss.npy"
        self._intermediary_file: Path = self._model_directory / "weights-1.npy"
        self._log_file: Path = self._model_directory / "log.txt"

        wm_print("--------------𓆩𓆪--------------")
        wm_print(f"Using device {cstr(self.device, 'HEADER')}")
        wm_print(f"Saving weights to {cstr(self._model_directory, 'HEADER')}...")

        # check to record that we're in a new training session and save the config file
        if self._model_directory.is_dir():
            self._fresh_directory = False
        else:
            self._fresh_directory = True
            wm_print(
                cstr(
                    "New training instance detected, generating weights directory in 3 seconds...",
                    "WARNING",
                ),
            )
            time.sleep(3)
            self._model_directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                config_yaml,
                self._model_directory / "config_copy.yaml",
            )

    @cached_property
    def device(self):
        """__get_device."""
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except ImportError:
            raise WingmanException(
                "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
            )

        return device

    def checkpoint(
        self, loss: float, step: int | None = None
    ) -> tuple[bool, Path, Path]:
        """checkpoint.

        Records training every logging_interval steps.

        Returns three things:
        - indicator on whether we should save weights
        - path of where the weight files should be saved
        - path of where the optim files should be saved

        Args:
        ----
            loss (float): learning loss of the model as a detached float
            step (int | None): step number, automatically incremented if None

        Returns:
        -------
            tuple[bool, Path, Path]: to_update, weights_file, optim_file

        """
        # if step is None, we automatically increment
        if step is None:
            step = self._previous_ckpt_step + 1

        # check that our step didn't go in reverse
        if step < self._previous_ckpt_step:
            raise WingmanException(
                cstr(
                    f"We can't step backwards! Got step {step} but the previous logging step was {self._previous_ckpt_step}.",
                    "FAIL",
                )
            )
        self._previous_ckpt_step = step

        """ACCUMULATE LOSS"""
        self._cumulative_loss += loss
        self._num_losses += 1

        # if we haven't passed the required number of steps
        if step < self._next_log_step:
            return False, self._model_file, self._optim_file

        # log to wandb if needed, but only on the logging steps
        if self.cfg.wandb.enable:
            self.wandb_log()

        """GET NEW AVG LOSS"""
        # record the losses and reset the cumulative
        avg_loss = self._cumulative_loss / self._num_losses
        self._cumulative_loss = 0.0
        self._num_losses = 0

        # compute the next time we need to log
        self._next_log_step = (
            int(step / self.cfg.logging.interval) + 1
        ) * self.cfg.logging.interval

        # always print on n intervals
        wm_print(
            f"Step {cstr(step, 'OKCYAN')}; Average Loss {cstr(f'{avg_loss:.5f}', 'OKCYAN')}; Lowest Average Loss {cstr(f'{self._lowest_cumulative_lost:.5f}', 'OKCYAN')}",
            self.cfg.logging.filename,
        )

        """CHECK IF NEW LOSS IS BETTER"""
        # if we don't meet the criteria for saving
        if avg_loss >= self._lowest_cumulative_lost - self.cfg.logging.greater_than:
            # accumulate skips
            if self._skips < self.cfg.logging.max_skips:
                self._skips += 1
                return False, self._model_file, self._optim_file
            else:
                # save the network to intermediary if we crossed the max number of skips
                wm_print(
                    f"Passed {self.cfg.logging.max_skips} intervals without saving so far, saving weights to: {cstr(self._intermediary_file, 'OKCYAN')}",
                    self.cfg.logging.filename,
                )
                self._skips = 0
                return True, self._intermediary_file, self._optim_file

        """NEW LOSS IS BETTER"""
        # redefine the new lowest loss and reset the skips
        self._lowest_cumulative_lost = avg_loss
        self._skips = 0

        # increment means return the files with incremented checkpoint number
        if self.cfg.model.increment_ckpt:
            # check if we are safe to increment checkpoint numbers and regenerate weights file
            if self._previous_ckpt == -1 or self._model_file.exists():
                self._previous_ckpt = self._current_ckpt
                self._model_file = (
                    self._model_directory / f"weights{self._current_ckpt}.pth"
                )
                self._current_ckpt += 1

            else:
                wm_print(
                    cstr(
                        "Didn't save weights file for the previous checkpoint number (self.ckpt_number), not incrementing checkpoint number.",
                        "WARNING",
                    ),
                    self.cfg.logging.filename,
                )
                self._current_ckpt = self._previous_ckpt

        # record the lowest running loss in the status file
        np.save(self._lowest_loss_file, self._lowest_cumulative_lost)

        wm_print(
            f"New lowest point, saving weights to: {cstr(self._model_file, 'OKGREEN')}",
            self.cfg.logging.filename,
        )

        return True, self._model_file, self._optim_file

    def wandb_log(self) -> None:
        """wandb_log.

        Logs the internal log to WandB.
        Start logging by adding things to it using:
        ```
        wm.log["foo"] = "bar"
        ```

        Returns
        -------
            None:

        """
        if not isinstance(self.log, dict):
            raise WingmanException(
                cstr(f"log must be dictionary, currently it is {self.log}.", "FAIL")
            )
        if self.cfg.wandb.enable:
            wandb.log(self.log)

    def write_auxiliary(
        self, data: np.ndarray, variable_name: str, precision: str = "%1.3f"
    ) -> None:
        """write_auxiliary.

        Args:
        ----
            data (np.ndarray): data
            variable_name (str): variable_name
            precision (str): precision

        Returns:
        -------
            None:

        """
        if not len(data.shape) == 1:
            raise WingmanException(
                cstr("Data must be only 1 dimensional ndarray", "FAIL")
            )
        filename = self._model_directory / f"{variable_name}.csv"
        with open(filename, "ab") as f:
            np.savetxt(f, [data], delimiter=",", fmt=precision)

    def get_weight_files(self, latest: bool = True) -> tuple[bool, Path, Path]:
        """get_weight_files.

        Returns three things:
        - indicator on whether we have weight files
        - directory of where the weight files are
        - directory of where the optim files are

        Args:
        ----
            latest (bool): whether we want the latest file or the one determined by `ckpt_number`

        Returns:
        -------
            tuple[bool, Path, Path]: have_file, weights_file, optim_file

        """
        # if we don't need the latest file, get the one specified
        if not latest:
            if os.path.isfile(self._model_file):
                self._model_file = (
                    self._model_directory / f"weights{self._current_ckpt}.pth"
                )
                wm_print(
                    f"Using weights file: {cstr(f'{self._model_directory}/weights{self._current_ckpt}.pth', 'OKGREEN')}",
                    self.cfg.logging.filename,
                )
                return True, self._model_file, self._optim_file
            else:
                raise ValueError(
                    cstr(
                        f"Checkpoint number {self._current_ckpt} was requested, but it doesn't exist.",
                        "FAIL",
                    )
                )

        # while the file exists, try to look for a file one checkpoint later
        self._current_ckpt = 0
        while self._model_file.is_file():
            self._current_ckpt += 1
            self._model_file = (
                self._model_directory / f"weights{self._current_ckpt}.pth"
            )

        # once the checkpoint doesn't exist, decrement by one and use that file
        self._current_ckpt = max(self._current_ckpt - 1, 0)
        self._model_file = self._model_directory / f"weights{self._current_ckpt}.pth"

        # if the file doesn't exist, notify and ignore
        if not self._model_file.is_file():
            if not self._fresh_directory:
                wm_print(
                    cstr(
                        "No weights file found, generating new one during training.",
                        "WARNING",
                    ),
                    self.cfg.logging.filename,
                )
            self._fresh_directory = False

            return False, self._model_file, self._optim_file
        else:
            # hitch a ride to update the lowest running loss
            self._lowest_cumulative_lost = np.load(self._lowest_loss_file).item()

            wm_print(
                f"Using weights file: {cstr(f'{self._model_directory}/weights{self._current_ckpt}.pth', 'OKGREEN')}",
                self.cfg.logging.filename,
            )
            wm_print(
                f"Lowest Running Loss for Net: {cstr(self._lowest_cumulative_lost, 'OKCYAN')}",
                self.cfg.logging.filename,
            )

            # check if the optim file exists
            if not os.path.isfile(self._optim_file):
                wm_print(
                    cstr("Optim file not found, please be careful!", "WARNING"),
                    self.cfg.logging.filename,
                )

            return True, self._model_file, self._optim_file
