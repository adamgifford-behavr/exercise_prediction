"""
Module that runs development environment testing. Call to `main()` checks if the major
version of the Python interpreter is equal to the major version of the Python interpreter
specified in the REQUIRED_PYTHON variable. If not, then raise a TypeError.
"""
import sys

REQUIRED_PYTHON = "python3"


def main():
    """
    Method that tests for the appropriate development environment. If the major version
    of the Python interpreter is not equal to the major version of the Python interpreter
    specified in the REQUIRED_PYTHON variable, then raise a TypeError.
    """
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError(f"Unrecognized python interpreter: {REQUIRED_PYTHON}")

    if system_major != required_major:
        raise TypeError(
            f"This project requires Python {required_major}. Found: Python {sys.version}"
        )

    print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    main()
