import ast
import os
import types
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / 'sa2va_eval'
    / 'projects'
    / 'ST'
    / 'eve'
    / 'eval'
    / 'model_vqa_qbench.py'
)


def load_eval_model(namespace):
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'), filename=str(SCRIPT))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'eval_model')
    exec(compile(ast.Module([function], type_ignores=[]), str(SCRIPT), 'exec'),
         namespace)
    return namespace['eval_model']


class QBenchEntrypointTest(unittest.TestCase):

    def test_model_path_is_expanded_before_loading(self):
        received_paths = []

        class StopAfterLoad(unittest.TestCase.failureException):
            pass

        def load_pretrained_model(path, model_type):
            received_paths.append(path)
            raise StopAfterLoad

        eval_model = load_eval_model({
            'os': os,
            'disable_torch_init': lambda: None,
            'load_pretrained_model': load_pretrained_model,
        })
        args = types.SimpleNamespace(model_path='~/model', model_type='qbench')

        with self.assertRaises(StopAfterLoad):
            eval_model(args)

        self.assertEqual(received_paths, [os.path.expanduser('~/model')])


if __name__ == '__main__':
    unittest.main()
