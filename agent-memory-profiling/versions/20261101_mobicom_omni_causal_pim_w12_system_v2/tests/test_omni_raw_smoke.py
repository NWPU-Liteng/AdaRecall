import ast
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_omni_raw_image_smoke import pack_session_turns, token_vector


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class OmniSmokeTests(unittest.TestCase):
    def test_encoder_function_identity_can_be_rebuilt_from_frozen_source(self):
        source = Path(inspect.getsourcefile(token_vector)).read_text()
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'token_vector')
        self.assertEqual(inspect.getsource(token_vector), ast.get_source_segment(source, function) + '\n')

    def test_budget_rejects_whole_turns_and_records_ids(self):
        turns = [{"atom_id": "long", "role": "user", "content_type": "text", "content": "x" * 100},
                 {"atom_id": "short", "role": "user", "content_type": "text", "content": "ok"},
                 {"atom_id": "image", "role": "user", "content_type": "image", "content": "ignored"}]
        rendered, accepted, rejected = pack_session_turns(turns, CharacterTokenizer(), 30)
        self.assertEqual(accepted, ["short"])
        self.assertEqual(rejected, ["long"])
        self.assertIn("ok", rendered)
        self.assertNotIn("ignored", rendered)
        self.assertLessEqual(len(rendered), 30)

    def test_nonpositive_budget_rejected(self):
        with self.assertRaises(ValueError):
            pack_session_turns([], CharacterTokenizer(), 0)



if __name__ == "__main__":
    unittest.main()
