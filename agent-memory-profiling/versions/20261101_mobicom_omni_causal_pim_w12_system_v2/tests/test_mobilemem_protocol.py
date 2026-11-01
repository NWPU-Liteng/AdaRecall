import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from score_mobilemem import load_pinned_protocol


class MobileMemProtocolTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'third_party/mobilemem_eval/MANIFEST.json').exists(), 'upstream protocol not downloaded')
    def test_pinned_f1_is_not_a_chinese_semantic_metric(self):
        _, token_f1, _, _ = load_pinned_protocol()
        self.assertEqual(token_f1('板羽球', '板羽球'), 1.0)
        self.assertEqual(token_f1('答案是板羽球', '板羽球'), 0.0)
        self.assertEqual(token_f1('', ''), 1.0)

    @unittest.skipUnless((ROOT / 'third_party/mobilemem_eval/membase/inference_utils/operators.py').exists(), 'upstream operator not downloaded')
    def test_upstream_single_and_multi_reference_serialization(self):
        _, _, template, _ = load_pinned_protocol()
        source = ROOT / 'third_party/mobilemem_eval/membase/inference_utils/operators.py'
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LLMExactMatch']
        namespace = {'NonCachedLLMOperator': object}
        exec(compile(tree, str(source), 'exec'), namespace)
        operator = namespace['LLMExactMatch']()
        operator._prompt = template
        for references in (['alpha'], ['alpha', 'beta']):
            actual = operator._preprocess(['question'], [references], ['prediction'])[0][0]['content']
            formatted = references[0] if len(references) == 1 else f"[{', '.join(references)}]"
            expected = template.substitute(question='question', prediction='prediction', golden_answers=formatted)
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
