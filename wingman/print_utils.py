"""For fancy printing in Wingman."""
# colour list
from typing import Any

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


def cstr(x: Any, ctype: str) -> str:
    """Makes a string colourful.

    Args:
        x (Any): the string
        ctype (str): the colour

    Returns:
        str: the coloured string
    """
    return f"{c_colors[ctype]}{x}{end_c}"


log_flag = cstr(cstr("wingman", "BOLD"), "OKCYAN")


def wm_print(x: Any):
    """Prints out strings decorated with the wingman status.

    Args:
        x (Any): the input string.
    """
    print(f"{log_flag}: {x}")
