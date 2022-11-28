import glob
import os
import shutil
import sys


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
    print("---------------------------------------------")
    print("Beginning compression...")
    print("---------------------------------------------")

    target_dir = "./weights"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]

    # get the original filesize
    original_size = _get_dir_size(target_dir)

    # all versions
    dirs = glob.glob(f"./{target_dir}/Version*")

    # which folders to delete
    to_delete = []

    # start going through each folder
    for dir in dirs:
        weights = glob.glob(os.path.join(dir, "weights*.pth"))

        # if no weights files
        if len(weights) == 0:
            to_delete.append(dir)
            continue

        # start finding the latest file
        num = 0
        has_num = True
        latest = ""
        while has_num:
            latest = os.path.join(dir, f"weights{num}.pth")

            # find the latest version and remove it from the list
            if latest in weights:
                num += 1
                has_num = True
            else:
                num -= 1
                has_num = False
                latest = os.path.join(dir, f"weights{num}.pth")
                weights.remove(latest)

        # remove the intermediary from our list if it's there
        intermediary = os.path.join(dir, "weights-1.pth")
        if intermediary in weights:
            weights.remove(intermediary)

        if len(weights) > 0:
            print(f"Compressing {dir}...")

            # for the remainder of the list, delete all the weights
            for weight in weights:
                os.remove(weight)

            # rename the latest weights to 0
            os.rename(latest, os.path.join(dir, "weights0.pth"))

            print(f"Compressed {dir}.")
        else:
            print(f"Nothing to compress in {dir}.")

    # delete flagged directories
    for dir in to_delete:
        shutil.rmtree(dir, ignore_errors=False, onerror=None)
        print(f"Deleted {dir}.")

    # get the final filesize
    final_size = _get_dir_size(target_dir)

    # printout
    print("---------------------------------------------")
    print(f"Original disk usage: {original_size / 1e9} gigabytes")
    print(f"Removed {(original_size - final_size) / 1e9} gigabytes")
    print(f"Final disk usage: {final_size / 1e9} gigabytes")
    print("---------------------------------------------")


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
