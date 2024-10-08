"""The core of Wingman."""

from __future__ import annotations

import math
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
    have_file, model_dir, ckpt_dir = self.get_weight_files()
    if have_file:
        model.load(f"{ckpt_dir}/weights.pth")
        optim.load(f"{model_dir}/optim.pth")

    # run some training:
    while(training):
        ...
        # training code here
        ...

        update_weights, model_dir, ckpt_dir = self.checkpoint(loss, step_number)
        if update_weights:
            model.save(f"{ckpt_dir}/weights.pth")
            optim.save(f"{model_dir}/optim.pth")
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

        # file paths
        self._model_dir: Path = Path(self.cfg.model.save_directory) / str(
            self.cfg.model.id
        )
        self._log_file: Path = self._model_dir / "log.txt"
        self._lowest_loss_file: Path = self._model_dir / "lowest_loss.npy"

        wm_print("---------------------------------------------")
        wm_print(f"Using device {cstr(self.device, 'HEADER')}")
        wm_print(f"Saving weights to {cstr(self._model_dir, 'HEADER')}...")

        # check to record that we're in a new training session and save the config file
        if self._model_dir.is_dir():
            self._fresh_directory = False
        else:
            self._fresh_directory = True
            wm_print(
                cstr(
                    "New training instance detected, generating model directory in 3 seconds...",
                    "WARNING",
                ),
            )
            time.sleep(3)
            self._model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                config_yaml,
                self._model_dir / "config_copy.yaml",
            )

    @property
    def _ckpt_dir(self) -> Path:
        ckpt_dir = self._model_dir / f"{self._current_ckpt}/"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    @cached_property
    def _temp_dir(self) -> Path:
        temp_dir = self._model_dir / "-1/"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

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
        - path of the model directory (ie: "/home/you/project/weights/12345/")
        - path of the checkpoint directory (ie: "/home/you/project/weights/12345/0/")

        Args:
        ----
            loss (float): learning loss of the model as a detached float
            step (int | None): step number, automatically incremented if None

        Returns:
        -------
            tuple[bool, Path, Path]: to_update, model_dir, ckpt_dir

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
            return False, self._model_dir, self._ckpt_dir

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
            f"Step {cstr(step, 'OKCYAN')}; "
            f"Average Loss {cstr(f'{avg_loss:.5f}', 'OKCYAN')}; "
            f"Lowest Average Loss {cstr(f'{self._lowest_cumulative_lost:.5f}', 'OKCYAN')}",
            self.cfg.logging.filename,
        )

        """CHECK IF NEW LOSS IS BETTER"""
        # if we don't meet the criteria for saving
        if avg_loss >= self._lowest_cumulative_lost - self.cfg.logging.greater_than:
            # accumulate skips
            if self._skips < self.cfg.logging.max_skips:
                self._skips += 1
                return False, self._model_dir, self._ckpt_dir
            else:
                # save the network to intermediary if we crossed the max number of skips
                wm_print(
                    f"Passed {self.cfg.logging.max_skips} intervals without saving so far. "
                    f"Issuing new checkpoint directory: {cstr(self._temp_dir, 'OKCYAN')}",
                    self.cfg.logging.filename,
                )
                self._skips = 0
                return True, self._model_dir, self._temp_dir

        """NEW LOSS IS BETTER"""
        # redefine the new lowest loss and reset the skips
        self._lowest_cumulative_lost = avg_loss
        self._skips = 0

        # increment means return the files with incremented checkpoint number
        if self.cfg.model.increment_ckpt:
            # check if we are safe to increment checkpoint numbers and regenerate weights file
            if self._previous_ckpt == -1 or self._ckpt_dir.exists():
                self._previous_ckpt = self._current_ckpt
                self._current_ckpt += 1

            else:
                wm_print(
                    cstr(
                        f"Didn't populate previous checkpoint directory ({self._ckpt_dir}). "
                        "Therefore, not incrementing checkpoint number.",
                        "WARNING",
                    ),
                    self.cfg.logging.filename,
                )
                self._current_ckpt = self._previous_ckpt

        # record the lowest running loss in the status file
        np.save(self._lowest_loss_file, self._lowest_cumulative_lost)

        wm_print(
            f"New lowest point, issuing new checkpoint directory: {cstr(self._ckpt_dir, 'OKGREEN')}",
            self.cfg.logging.filename,
        )

        return True, self._model_dir, self._ckpt_dir

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
        filename = self._model_dir / f"{variable_name}.csv"
        with open(filename, "ab") as f:
            np.savetxt(f, [data], delimiter=",", fmt=precision)

    def get_weight_files(self, latest: bool = True) -> tuple[bool, Path, Path]:
        """get_weight_files.

        Returns three things:
        - indicator on whether we should save weights
        - path of the model directory (ie: "/home/you/project/weights/12345/")
        - path of the checkpoint directory (ie: "/home/you/project/weights/12345/0/")

        Args:
        ----
            latest (bool): whether to retrieve the latest checkpoint or just get the one specified in the config

        Returns:
        -------
            tuple[bool, Path, Path]: to_update, model_dir, ckpt_dir

        """
        # if we don't need the latest file, get the one specified
        if not latest:
            ckpt_dir = self._model_dir / f"{self._current_ckpt}/"
            if ckpt_dir.exists():
                wm_print(
                    f"Using checkpoint directory: {cstr(f'{self._ckpt_dir}', 'OKGREEN')}",
                    self.cfg.logging.filename,
                )
                return True, self._model_dir, self._ckpt_dir
            else:
                raise ValueError(
                    cstr(
                        f"Checkpoint number {self._current_ckpt} was requested, "
                        f"but checkpoint directory {ckpt_dir} does not exist.",
                        "FAIL",
                    )
                )

        # while the file exists, try to look for a file one checkpoint later
        # once the checkpoint doesn't exist, decrement by one and use that file
        self._current_ckpt = 0
        while (self._model_dir / f"{self._current_ckpt}/").exists():
            self._current_ckpt += 1
        self._current_ckpt = max(self._current_ckpt - 1, 0)

        # if the file doesn't exist, notify and ignore
        if not (self._model_dir / f"{self._current_ckpt}/").exists():
            if not self._fresh_directory:
                wm_print(
                    cstr(
                        "No checkpoint directory found, generating new one during training.",
                        "WARNING",
                    ),
                    self.cfg.logging.filename,
                )
            self._fresh_directory = False

            return False, self._model_dir, self._ckpt_dir
        else:
            # hitch a ride to update the lowest running loss
            self._lowest_cumulative_lost = np.load(self._lowest_loss_file).item()

            wm_print(
                f"Using checkpoint directory: {cstr(f'{self._ckpt_dir}', 'OKGREEN')}",
                self.cfg.logging.filename,
            )
            wm_print(
                f"Lowest Running Loss: {cstr(self._lowest_cumulative_lost, 'OKCYAN')}",
                self.cfg.logging.filename,
            )

            return True, self._model_dir, self._ckpt_dir
