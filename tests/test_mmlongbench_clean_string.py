import ast
import re
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / 'sa2va_eval'
    / 'vlmeval'
    / 'dataset'
    / 'mmlongbench.py'
)


def load_get_clean_string():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'), filename=str(SCRIPT))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'get_clean_string')
    namespace = {'re': re}
    exec(compile(ast.Module([function], type_ignores=[]), str(SCRIPT), 'exec'),
         namespace)
    return namespace['get_clean_string']


class MMLongBenchCleanStringTest(unittest.TestCase):

    def test_removes_exact_numeric_suffixes(self):
        clean = load_get_clean_string()

        self.assertEqual(clean('12 miles'), '12')
        self.assertEqual(clean('5 mile'), '5')
        self.assertEqual(clean('10 million'), '10')

    def test_does_not_strip_suffix_characters_from_other_words(self):
        clean = load_get_clean_string()

        self.assertEqual(clean('smiles'), 'smiles')
        self.assertEqual(clean('profile'), 'profile')


if __name__ == '__main__':
    unittest.main()
