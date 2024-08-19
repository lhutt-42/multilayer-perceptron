"""
This module contains the tests for the Parser class.
"""

import unittest
from unittest.mock import patch, mock_open, call, Mock

import pandas as pd
import pandas.testing as pd_testing

from mlp.parser.file import load_dataset, save_dataset


class TestParserFiles(unittest.TestCase):
    """
    This class contains the tests for the Parser class.
    """

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_read_dataset_not_found(self, _):
        """
        Test the case when the file is not found.
        """

        with self.assertRaises(SystemExit) as cm:
            load_dataset('test.csv')

        self.assertEqual(cm.exception.code, 1)


    @patch('builtins.open', side_effect=PermissionError)
    def test_read_dataset_permission_error(self, _):
        """
        Test the case when there is a permission error.
        """

        with self.assertRaises(SystemExit) as cm:
            load_dataset('test.csv')

        self.assertEqual(cm.exception.code, 1)


    @patch('builtins.open', new_callable=mock_open, read_data='1.0,2.0\n3.0,4.0\n')
    def test_read_dataset_success(self, _):
        """
        Test the case when the model is read correctly.
        """

        df = load_dataset('test.csv')

        pd_testing.assert_frame_equal(
            df,
            pd.DataFrame([
                [1.0, 2.0],
                [3.0, 4.0]
            ])
        )


    @patch('builtins.open', side_effect=PermissionError)
    def test_save_dataset_permission_error(self, _):
        """
        Test the case when there is a permission error.
        """

        with self.assertRaises(SystemExit) as cm:
            save_dataset(pd.DataFrame(), 'test.csv')

        self.assertEqual(cm.exception.code, 1)


    # pylint: disable=unused-argument
    @patch('os.makedirs', new_callable=Mock)
    @patch('builtins.open', new_callable=mock_open)
    def test_save_dataset_success(self, mock_file, *args):
        """
        Test the case when the model is saved correctly.
        """

        save_dataset(pd.DataFrame([
            [1.0, 2.0],
            [3.0, 4.0]
        ]), 'test.csv')

        calls = [call('1.0,2.0\n'), call('3.0,4.0\n')]
        mock_file().write.assert_has_calls(calls, any_order=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
