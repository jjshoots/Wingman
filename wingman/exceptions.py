"""Exceptions."""

from wingman.print_utils import cstr


class WingmanException(Exception):
    """WingmanException."""

    def __init__(self, message: str = ""):
        """__init__.

        Args:
        ----
            message (str): the message

        """
        message = cstr(message, "FAIL")
        super().__init__(message)
        self.message = message


class NeuralBlocksException(Exception):
    """NeuralBlocksException."""

    def __init__(self, message: str = ""):
        """__init__.

        Args:
        ----
            message (str): the message

        """
        message = cstr(message, "FAIL")
        super().__init__(message)
        self.message = message


class ReplayBufferException(Exception):
    """ReplayBufferException."""

    def __init__(self, message: str = ""):
        """__init__.

        Args:
        ----
            message (str): the message

        """
        message = cstr(message, "FAIL")
        super().__init__(message)
        self.message = message
