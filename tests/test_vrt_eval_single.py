import ast
import types
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / 'projects'
    / 'vrt_sa2va'
    / 'evaluation'
    / 'vrt_eval_single.py'
)


def load_main():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'), filename=str(SCRIPT))
    main_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'main')
    namespace = {}
    exec(compile(ast.Module([main_node], type_ignores=[]), str(SCRIPT), 'exec'),
         namespace)
    return namespace['main'], namespace


class VRTSingleGpuEvaluationTest(unittest.TestCase):

    def test_single_gpu_runs_evaluator_and_honors_sample_limit(self):
        main, namespace = load_main()
        calls = []

        class Dataset:

            def __init__(self, path):
                calls.append(('dataset', path))

            def get_evaluation_samples(self):
                return ['sample-1', 'sample-2', 'sample-3']

        class Evaluator:

            def __init__(self, model_path, dataset, use_thinking):
                calls.append(('evaluator', model_path, use_thinking))
                self.dataset = dataset

            def evaluate_all_samples(self, output_dir):
                calls.append(
                    ('evaluate', output_dir,
                     list(self.dataset.get_evaluation_samples())))
                return {'total_samples': 2, 'metrics': {}}

        args = types.SimpleNamespace(
            model_path='model',
            tfrecord_path='data/*.tfrecord',
            output_dir='outputs',
            max_samples=2,
            no_thinking=False,
            gpus=1,
        )
        namespace.update({
            'parse_args': lambda: args,
            'derive_output_dir_from_model_path': lambda path: 'unused',
            'PackedVRTEvalDataset': Dataset,
            'VERLLMEvaluator': Evaluator,
            'print_evaluation_summary': lambda result: calls.append(
                ('summary', result['total_samples'])),
        })

        main()

        self.assertIn(('evaluate', 'outputs', ['sample-1', 'sample-2']), calls)
        self.assertIn(('summary', 2), calls)


if __name__ == '__main__':
    unittest.main()
