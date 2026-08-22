import ast
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / 'projects'
    / 'samtok'
    / 'evaluation'
    / 'perceptionlm'
    / 'plm_gcg_eval.py'
)


class PerceptionLMFallbackTest(unittest.TestCase):

    def test_device_transfer_fallback_appends_empty_text(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'),
                         filename=str(SCRIPT))
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'main')
        transfer_handler = next(
            handler for handler in ast.walk(main)
            if isinstance(handler, ast.ExceptHandler)
            and any(isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'process_and_save_output'
                    for node in ast.walk(handler)))
        appended_values = [
            call.args[0].value for call in ast.walk(transfer_handler)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'append'
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ]

        self.assertEqual(appended_values, [''])


if __name__ == '__main__':
    unittest.main()
