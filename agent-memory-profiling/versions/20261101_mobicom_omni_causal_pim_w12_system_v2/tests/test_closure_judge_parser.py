from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from judge_omni_closure import parse_label


class JudgeParserTests(unittest.TestCase):
    def test_strict_json_not_substring_matching(self):
        self.assertEqual(parse_label('{"reason":"fact matches", "label":"CORRECT"}'), 'CORRECT')
        self.assertEqual(parse_label('{"label":"WRONG"}'), 'WRONG')
        for response in ('CORRECT', 'yes', '{"label":"NOT_CORRECT"}', '["CORRECT"]',
                         'The answer is not CORRECT', '{"label":null}'):
            self.assertIsNone(parse_label(response))


if __name__ == '__main__':
    unittest.main()
