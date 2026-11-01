import copy
import hashlib
from pathlib import Path
from string import Template
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_omni_smoke_judge_requests import build_requests


class OmniJudgeTests(unittest.TestCase):
    def fixture(self):
        row = {'question_id': 'q', 'answer': 'prediction', 'stop_reason': 'eos',
               'answer_sha256': hashlib.sha256(b'prediction').hexdigest(),
               'question_sha256': hashlib.sha256(b'question').hexdigest()}
        return ({'complete': True, 'rows': [row]}, {'q': {'golden_answers': ['gold']}},
                {'q': {'question': 'question', 'split': 'development'}},
                Template('$question | $prediction | $golden_answers'))

    def test_only_label_side_formats_frozen_judge_prompt(self):
        requests = build_requests(*self.fixture())
        self.assertEqual(requests[0]['prompt'], 'question | prediction | gold')
        self.assertEqual(requests[0]['reader_stop_reason'], 'eos')

    def test_holdout_requires_explicit_scope(self):
        smoke, labels, manifests, template = self.fixture()
        manifests['q']['split'] = 'holdout'
        with self.assertRaises(ValueError):
            build_requests(smoke, labels, manifests, template)
        requests = build_requests(smoke, labels, manifests, template,
                                  allowed_splits=('holdout',))
        self.assertEqual(requests[0]['question_id'], 'q')

    def test_sealed_duplicate_mutated_and_incomplete_fail_closed(self):
        original = self.fixture()
        for kind in ('sealed', 'duplicate', 'prediction', 'question', 'incomplete'):
            smoke, labels, manifests, template = copy.deepcopy(original)
            if kind == 'sealed':
                manifests['q']['split'] = 'sealed_test'
            elif kind == 'duplicate':
                smoke['rows'] *= 2
            elif kind == 'prediction':
                smoke['rows'][0]['answer'] = 'changed'
            elif kind == 'question':
                manifests['q']['question'] = 'changed'
            else:
                smoke['complete'] = False
            with self.assertRaises(ValueError):
                build_requests(smoke, labels, manifests, template)


if __name__ == '__main__':
    unittest.main()
