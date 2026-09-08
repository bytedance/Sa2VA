import argparse
import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


EVALUATOR = (
    Path(__file__).parents[1]
    / 'projects'
    / 'sasasa2va'
    / 'evaluation'
    / 'ref_vos_eval.py'
)


def load_argument_parser():
    """Load only the evaluator's dependency-free argument parsing code."""
    source = EVALUATOR.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(EVALUATOR))
    selected = [
        node for node in tree.body
        if (isinstance(node, (ast.Assign, ast.FunctionDef))
            and (not isinstance(node, ast.Assign)
                 or any(isinstance(target, ast.Name)
                        and target.id in {'DATASETS_INFO', 'INFERENCE_MODES'}
                        for target in node.targets))
            and (not isinstance(node, ast.FunctionDef)
                 or node.name == 'parse_args'))
    ]
    namespace = {'argparse': argparse, 'os': os}
    exec(compile(ast.Module(selected, type_ignores=[]), str(EVALUATOR), 'exec'),
         namespace)
    return namespace['parse_args']


class SaSaSa2VAEvalCliTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parse_args = staticmethod(load_argument_parser())

    def parse(self, *arguments):
        with patch.object(sys, 'argv', ['ref_vos_eval.py', *arguments]):
            return self.parse_args()

    def test_documented_work_dir_and_default_mode(self):
        args = self.parse('model', '--work-dir', 'outputs')

        self.assertEqual(args.work_dir, 'outputs')
        self.assertEqual(args.mode, 'uniform')

    def test_underscored_work_dir_remains_supported(self):
        args = self.parse('model', '--work_dir', 'outputs')

        self.assertEqual(args.work_dir, 'outputs')

    def test_invalid_mode_is_rejected_by_argument_parser(self):
        with self.assertRaises(SystemExit):
            self.parse('model', '--mode', 'default')


if __name__ == '__main__':
    unittest.main()
