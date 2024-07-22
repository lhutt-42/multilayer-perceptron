"""
This module is used to run all the tests in the tests directory.
"""

import sys
import unittest


def main() -> None:
    """
    The main function for running the tests.
    """

    tests = unittest.TestLoader().discover('tests', pattern='test_*.py')
    result = unittest.TextTestRunner().run(tests)

    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == '__main__':
    main()
