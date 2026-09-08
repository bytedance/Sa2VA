import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = (
    Path(__file__).parents[1]
    / 'projects'
    / 'sa2va'
    / 'evaluation'
    / 'run_all_evals.py'
)


def load_script():
    spec = importlib.util.spec_from_file_location('run_all_evals_under_test',
                                                   SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunAllEvalsFailureStatusTest(unittest.TestCase):

    def test_subprocess_failure_is_propagated(self):
        module = load_script()
        failure = subprocess.CalledProcessError(7, ['evaluation'])

        with patch.object(sys, 'argv', ['run_all_evals.py', 'model']), \
                patch.object(module, 'run_command', side_effect=failure):
            with self.assertRaises(subprocess.CalledProcessError) as context:
                module.main()

        self.assertEqual(context.exception.returncode, 7)


if __name__ == '__main__':
    unittest.main()
