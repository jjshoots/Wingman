"""The configurator for Wingman."""

from __future__ import annotations

import argparse
import types
from pathlib import Path
from typing import Any

import numpy as np
import wandb
import yaml

from wingman.exceptions import WingmanException


class LockedNamespace(types.SimpleNamespace):
    """A locked namespace, where new attributes can be set but old ones cannot be overwritten."""

    def __setattr__(self, name, value):
        """__setattr__."""
        if hasattr(self, name):
            raise WingmanException(
                "Can't override values for locked_namespace. "
                f"Tried to set `{name}` which already has value {value}."
            )
        super().__setattr__(name, value)


def dict_to_locked_ns(nested_dict: dict[str, Any]) -> LockedNamespace:
    """dict_to_ns.

    Args:
    ----
        nested_dict (dict[str, Any]): nested_dict

    Returns:
    -------
        types.SimpleNamespace:

    """
    return LockedNamespace(
        **{
            k: (dict_to_locked_ns(v) if isinstance(v, dict) else v)
            for k, v in nested_dict.items()
        }
    )


def check_dict_superset(base: dict[str, Any], target: dict[str, Any]) -> bool:
    """Checks that the `base` is a superset of the `target`.

    Args:
    ----
        base (dict[str, Any]): base
        target (dict[str, Any]): target

    Returns:
    -------
        bool:

    """
    for k, v in target.items():
        if k not in base:
            return False
        if isinstance(v, dict):
            check_dict_superset(base[k], v)
    return True


def dict_cli_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """When provided a `config_dict`, allows the overriding of values via the cli.

    For example, if we have the following dict:
    ```
    data = {
        "a": {
            "b": 3
        }
    }
    ```

    This function allows us to override the value at "b" via
    ```
    python3 <program.py> --a.b=5
    ```

    Args:
    ----
        config_dict (dict[str, Any]): a nested dictionary of values.

    Returns:
    -------
        dict[str, Any]: an altered nested dictionary.

    """

    def nested_argparse(
        parser: argparse.ArgumentParser,
        nested_dict: dict[str, Any],
        basename: str = "",
    ) -> argparse.ArgumentParser:
        """Builds an argparser by recursively nesting arguments.

        Args:
        ----
            parser (argparse.ArgumentParser): parser
            nested_dict (dict[str, Any]): nested_dict
            basename (str): basename

        Returns:
        -------
            argparse.ArgumentParser: the resulting argparser with nested arguments.

        """
        for k, v in nested_dict.items():
            # extend the basename
            name = f"{basename}.{k}" if basename else k

            # if dict, we need to recurse
            if isinstance(v, dict):
                parser = nested_argparse(
                    parser=parser,
                    nested_dict=v,
                    basename=name,
                )

            # otherwise, just add to parser like usual
            else:
                parser.add_argument(
                    f"--{name}",
                    type=type(v),
                    nargs="?",
                    const=True,
                    default=v,
                    help="None",
                )

        return parser

    # recursively convert arguments to cli args
    raw_overrides = vars(
        nested_argparse(
            parser=argparse.ArgumentParser(allow_abbrev=False),
            nested_dict=config_dict,
        ).parse_args()
    )

    # pull out each item in the raw overrides and
    # replace the variable within the main config
    for joint_k, v in raw_overrides.items():
        k_list = joint_k.split(sep=".")
        current = config_dict
        for key in k_list[:-1]:
            current = current[key]
        current[k_list[-1]] = v

    return config_dict


def generate_wingman_config(config_yaml: Path | str) -> LockedNamespace:
    """Reads the yaml file provided at init and converts it to commandline arguments."""
    # read in the file
    with open(config_yaml) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # check the config is a superset of the expected yaml
    with open(Path(__file__).parent / "config.yaml") as f:
        if not check_dict_superset(
            base=config_dict,
            target=yaml.load(f, Loader=yaml.FullLoader),
        ):
            raise WingmanException(
                f"Incomplete yaml file `{config_yaml}` used. "
                "The yaml structure must be a superset of the yaml generated via `wingman-generate-config`."
            )

    # allow cli to override dict values
    config_dict = dict_cli_overrides(config_dict)

    # override model_id if needed
    if not config_dict["model"]["id"]:
        config_dict["model"]["id"] = str(np.random.randint(999999))

    # conditionally wandb
    if config_dict["wandb"]["enable"]:
        # generate the wandb run display name
        if config_dict["wandb"]["run"]["name"] != "":
            run_name = (
                f'{config_dict["wandb"]["run"]["name"]}, v={config_dict["model"]["id"]}'
            )
        else:
            run_name = config_dict["wandb"]["model"]["id"]

        # initialize wandb
        wandb.init(
            project=config_dict["wandb"]["project"]["name"],
            entity=config_dict["wandb"]["project"]["entity"],
            config=config_dict,
            name=run_name,
            notes=config_dict["wandb"]["run"]["notes"],
        )

        # optionally save code
        if config_dict["wandb"]["save_code"]:
            raise NotImplementedError("Save code is not yet implemented.")
            wandb.run.log_code(".", exclude_fn=lambda path: "venv" in path)

    return dict_to_locked_ns(config_dict)
