"""
This module contains the tests for the Parser class.
"""

import unittest
from unittest.mock import patch

from mlp.parser.arguments import create_parser


class TestParser(unittest.TestCase):
    """
    This class contains the tests for the Parser class.
    """

    def setUp(self):
        self.parser = create_parser()


    @patch('sys.argv', ['mlp'])
    def test_parse_arguments_none(self):
        """
        Test the case when no arguments are passed.
        """

        with self.assertRaises(SystemExit) as cm:
            _ = self.parser.parse_args()

        self.assertEqual(cm.exception.code, 2)


    @patch('sys.argv', ['mlp', 'test'])
    def test_parse_arguments_incorrect(self):
        """
        Test the case when incorrect arguments are passed.
        """

        with self.assertRaises(SystemExit) as cm:
            _ = self.parser.parse_args()

        self.assertEqual(cm.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
