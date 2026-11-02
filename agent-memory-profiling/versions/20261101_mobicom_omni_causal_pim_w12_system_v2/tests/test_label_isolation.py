import ast
import builtins
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from label_read_guard import install_label_read_guard, is_label_path


FORBIDDEN_STRINGS = (
    'labels.jsonl',
    'golden_answers',
    '"question_type"',
    "'question_type'",
    '"difficulty"',
    "'difficulty'",
)


def _serving_files() -> list[Path]:
    files = [
        SCRIPTS / 'run_proof_mainline.py',
        SCRIPTS / 'run_main_experiment.py',
        *sorted(SCRIPTS.glob('omni_proof_*.py')),
        *sorted((ROOT / 'pim_backend').glob('*.py')),
    ]
    return [path for path in files if path.is_file()]


class LabelIsolationTests(unittest.TestCase):
    def test_serving_sources_do_not_reference_evaluation_labels(self):
        files = _serving_files()
        self.assertGreater(len(files), 10)
        violations = []
        for path in files:
            source = path.read_text()
            for marker in FORBIDDEN_STRINGS:
                if marker in source:
                    violations.append(f'{path.relative_to(ROOT)}: {marker}')
        self.assertEqual(violations, [])

    def test_mainline_installs_guard_before_argument_parsing(self):
        source = (SCRIPTS / 'run_proof_mainline.py').read_text()
        tree = ast.parse(source)
        main_function = next(node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name == 'main')
        first = main_function.body[0]
        self.assertIsInstance(first, ast.Expr)
        self.assertIsInstance(first.value, ast.Call)
        self.assertIsInstance(first.value.func, ast.Name)
        self.assertEqual(first.value.func.id, 'install_label_read_guard')

    def test_runtime_guard_blocks_labels_and_allows_other_files(self):
        original_open = builtins.open
        try:
            install_label_read_guard()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.assertTrue(is_label_path(root / 'labels.jsonl'))
                self.assertTrue(is_label_path(str(root / 'labels.jsonl').encode()))
                self.assertFalse(is_label_path(root / 'labels.jsonl.bak'))
                with self.assertRaises(PermissionError):
                    open(root / 'labels.jsonl', 'w')
                allowed = root / 'allowed.json'
                with open(allowed, 'w') as handle:
                    handle.write('{}')
                self.assertEqual(allowed.read_text(), '{}')
        finally:
            builtins.open = original_open


if __name__ == '__main__':
    unittest.main()
