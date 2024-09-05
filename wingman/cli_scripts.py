"""CLI functionality of Wingman."""

from __future__ import annotations

import glob
import os
import re
import shutil
import sys

from wingman.print_utils import cstr, wm_print


def _get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += _get_dir_size(entry.path)
    return total


def compress_weights():
    """Compresses the weights directory by deleting empty directories and compressing all mark numbers down to 0.

    Example CLI usage:
    `wingman-compress-weights [optional weights directory]`
    """
    wm_print("---------------------------------------------")
    wm_print("Beginning compression...")
    wm_print("---------------------------------------------")

    target_dir = "./weights"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]

    # get the original filesize
    original_size = _get_dir_size(target_dir)

    # all ids
    id_dirs = glob.glob(f"./{target_dir}/*")

    # which folders to delete
    empty_ids = []

    # start going through each folder
    for id_dir in id_dirs:
        mark_dirs = [
            f
            for f in glob.glob(os.path.join(id_dir, "*"))
            if (re.compile(r"^-?\d+$").match(f.split("/")[-1]) and os.path.isdir(f))
        ]

        # if no weights files
        if len(mark_dirs) == 0:
            empty_ids.append(id_dir)
            continue

        # start finding the latest file
        _mark_number = 0
        latest_mark_dir = os.path.join(id_dir, f"{_mark_number}")
        while latest_mark_dir in mark_dirs:
            # as long as it exists, increment it
            _mark_number += 1
            latest_mark_dir = os.path.join(id_dir, f"{_mark_number}")

        # remove the latest mark number from the list, we want to keep this
        _mark_number = min(0, _mark_number - 1)
        latest_mark_dir = os.path.join(id_dir, f"{_mark_number}")
        if latest_mark_dir in mark_dirs:
            mark_dirs.remove(latest_mark_dir)

        # remove the intermediary from our list if it's there
        intermediary = os.path.join(id_dir, "-1")
        if intermediary in mark_dirs:
            mark_dirs.remove(intermediary)

        # exit if nothing to delete
        if len(mark_dirs) == 0:
            wm_print(f"Nothing to compress in {cstr(id_dir, 'OKGREEN')}.")
            continue

        # for the remainder of the list, delete all the weights
        wm_print(f"Compressing {id_dir}...")
        for mark_dir in mark_dirs:
            shutil.rmtree(mark_dir)

        # rename the latest weights to 0
        os.rename(latest_mark_dir, os.path.join(id_dir, "0"))

        wm_print(f"Compressed {cstr(id_dir, 'WARNING')}.")

    # delete flagged directories
    for id_dir in empty_ids:
        shutil.rmtree(id_dir)
        wm_print(f"Deleted {cstr(id_dir, 'WARNING')}.")

    # get the final filesize
    final_size = _get_dir_size(target_dir)

    # printout
    wm_print("---------------------------------------------")
    wm_print(f"Original disk usage: {cstr(original_size / 1e9, 'OKGREEN')} gigabytes.")
    wm_print(
        f"Removed {cstr((original_size - final_size) / 1e9, 'WARNING')} gigabytes."
    )
    wm_print(f"Final disk usage: {cstr(final_size / 1e9, 'WARNING')} gigabytes.")
    wm_print("---------------------------------------------")


def generate_yaml():
    """Generates the basic yaml file for the Wingman module.

    Example CLI usage:
    `wingman-generate-yaml [optional filename]`
    """
    source = os.path.join(os.path.dirname(__file__), "./config.yaml")

    target = "./config.yaml"
    if len(sys.argv) > 1:
        target = sys.argv[1]

    shutil.copyfile(source, target)

    wm_print(f"Generated config file at: {cstr(target, 'OKGREEN')}.")
