"""For fancy printing in Wingman."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

# colour list
c_colors = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKCYAN": "\033[96m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}
end_c = "\033[0m"


def cstr(
    x: Any,
    ctype: Literal[
        "HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARNING", "FAIL", "BOLD", "UNDERLINE"
    ],
) -> str:
    """Makes a string colourful.

    Args:
    ----
        x (Any): the string
        ctype (str): the colour

    Returns:
    -------
        str: the coloured string

    """
    return f"{c_colors[ctype]}{x}{end_c}"


log_flag = cstr(cstr("wingman", "BOLD"), "OKCYAN")
ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")


def wm_print(x: Any, log_file: Path | str = ""):
    """Prints out strings decorated with the wingman status.

    Args:
    ----
        x (Any): the input string
        log_file (str): a target path to save the log file if any

    """
    print(f"{log_flag}: {x}")

    if not log_file:
        return

    with open(log_file, "a") as f:
        f.write(ansi_escape.sub("", x))
        f.write("\n")
        f.write("\n")
