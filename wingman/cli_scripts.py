"""CLI functionality of Wingman."""
import glob
import os
import shutil
import sys

from .print_utils import cstr, wm_print


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

    # all versions
    versions = glob.glob(f"./{target_dir}/Version*")

    # debug version, remove from all versions
    debug_version = f"./{target_dir}/VersionDebug"
    if debug_version in versions:
        versions.remove(debug_version)

    # which folders to delete
    empty_versions = []

    # start going through each folder
    for version in versions:
        marks = glob.glob(os.path.join(version, "weights*.pth"))

        # if no weights files
        if len(marks) == 0:
            empty_versions.append(version)
            continue

        # start finding the latest file
        mark_number = 0
        latest_mark = os.path.join(version, f"weights{mark_number}.pth")
        while latest_mark in marks:
            # as long as it exists, increment it
            mark_number += 1
            latest_mark = os.path.join(version, f"weights{mark_number}.pth")

        # remove the latest mark number from the list, we want to keep this
        mark_number = min(0, mark_number - 1)
        latest_mark = os.path.join(version, f"weights{mark_number}.pth")
        if latest_mark in marks:
            marks.remove(latest_mark)

        # remove the intermediary from our list if it's there
        intermediary = os.path.join(version, "weights-1.pth")
        if intermediary in marks:
            marks.remove(intermediary)

        # exit if nothing to delete
        if len(marks) == 0:
            wm_print(f"Nothing to compress in {cstr(version, 'OKGREEN')}.")
            continue

        # for the remainder of the list, delete all the weights
        wm_print(f"Compressing {version}...")
        for weight in marks:
            os.remove(weight)

        # rename the latest weights to 0
        os.rename(latest_mark, os.path.join(version, "weights0.pth"))

        wm_print(f"Compressed {cstr(version, 'WARNING')}.")

    # delete flagged directories
    for version in empty_versions:
        shutil.rmtree(version, ignore_errors=False, onerror=None)
        wm_print(f"Deleted {cstr(version, 'WARNING')}.")

    # delete the debug directory
    if debug_version in versions:
        shutil.rmtree(debug_version, ignore_errors=False, onerror=None)
        wm_print(f"Deleted {cstr(debug_version, 'WARNING')}.")

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
