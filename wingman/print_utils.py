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
    return f"{c_colors[ctype]}{x}{end_c}"


log_flag = cstr(cstr("wingman", "BOLD"), "OKBLUE")


def wm_print(x: Any):
    print(f"{log_flag}: {x}")
