"""Helper for converting a list of nested dictionaries to a nested dictionary of lists."""

from __future__ import annotations

import functools
from typing import Any, Generator

import numpy as np


def _iter_nested_keys(base_dict: dict[str, Any]) -> Generator[list[str], None, None]:
    """Given a nested dictionary, yields a list of keys to each element.

    Args:
    ----
        base_dict (dict[str, Any]): base_dict

    Returns:
    -------
        Generator[list[str], None, None]:

    """
    for key, value in base_dict.items():
        if isinstance(value, dict):
            for sub_keys in _iter_nested_keys(value):
                yield [key, *sub_keys]
        else:
            yield [key]


def listed_dict_to_dicted_list(
    list_dict: list[dict[str, Any]], stack: bool
) -> dict[str, Any]:
    """Given a list of nested dicts, returns a nested dict of lists.

    Args:
    ----
        list_dict (list[dict[str, Any]]): list_dict
        stack (bool): stack

    Returns:
    -------
        dict[str, Any]:

    """
    result = {}
    for key_list in _iter_nested_keys(list_dict[0]):
        # for each element in the expected dictionary
        # scaffold to that point
        ptr = result
        for key in key_list[:-1]:
            ptr = ptr.setdefault(key, {})

        # this goes through the main list of dicts
        # and collects elements at the position determined by the key_list
        dicted_list: list = [
            functools.reduce(lambda d, k: d[k], key_list, dict_item)
            for dict_item in list_dict
        ]

        # we can't call `concatenate` on a list of non-np.ndarray items
        if isinstance(dicted_list[0], np.ndarray) and len(dicted_list[0].shape) > 0:
            if stack:
                ptr[key_list[-1]] = np.stack(dicted_list, axis=0)
            else:
                ptr[key_list[-1]] = np.concatenate(dicted_list, axis=0)
        else:
            if stack:
                ptr[key_list[-1]] = np.expand_dims(
                    np.asarray(dicted_list),
                    axis=0,
                )
            else:
                ptr[key_list[-1]] = np.asarray(dicted_list)
    return result
