import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_atom_token_cost import AtomUnionTokenCosts, QWEN_LEGACY_SPLIT, QWEN_SPLIT, supports_fragment_costs
from omni_evidence_planner import render_atoms


class BackendWrapper:
    def __init__(self, backend):
        self.backend_tokenizer = backend

    def encode(self, text, add_special_tokens=False):
        return self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


class AtomTokenCostTests(unittest.TestCase):
    def tokenizer(self, pattern=QWEN_SPLIT):
        from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers, pre_tokenizers, trainers
        backend = Tokenizer(models.BPE(unk_token='[UNK]'))
        backend.normalizer = normalizers.NFC()
        backend.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(pattern), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)])
        backend.train_from_iterator(['[session][atom][user] hello é 世界!\n[] foo 123\n\n'],
            trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                special_tokens=['[UNK]']))
        backend.add_special_tokens([AddedToken('<|im_end|>', lstrip=False, rstrip=False, normalized=False)])
        return BackendWrapper(backend)

    def atoms(self):
        contents = ['hello!  ', 'é 世界\n\n', '<|im_end|>\n', 'foo\r\n', '   ', '!?!', '\u0301']
        return {str(index): {'atom_id': str(index), 'session_id': 'session', 'ordinal': index,
                            'role': 'user', 'content_type': 'text', 'content': text}
                for index, text in enumerate(contents)}

    def test_all_nonempty_subsets_preserve_full_token_cost(self):
        for pattern in (QWEN_SPLIT, QWEN_LEGACY_SPLIT):
            tokenizer = self.tokenizer(pattern)
            self.assertTrue(supports_fragment_costs(tokenizer))
            atoms = self.atoms()
            costs = AtomUnionTokenCosts(tokenizer, atoms)
            for count in range(1, len(atoms) + 1):
                for subset in itertools.combinations(atoms, count):
                    self.assertEqual(costs.count(subset), len(tokenizer.encode(render_atoms(
                        [atoms[identity] for identity in subset]), add_special_tokens=False)))
            self.assertGreater(costs.summary()['fragment_state_hits'], 0)
            self.assertLessEqual(costs.fragment_producer_calls, 2 * len(atoms))
            self.assertEqual(costs.fallback_calls, 0)

    def test_unknown_tokenizer_uses_full_rendering(self):
        class CharacterTokenizer:
            def encode(self, text, **kwargs):
                return list(text)
        atoms = self.atoms()
        costs = AtomUnionTokenCosts(CharacterTokenizer(), atoms)
        self.assertFalse(costs.supported)
        self.assertEqual(costs.count(atoms), len(render_atoms(list(atoms.values()))))
        self.assertEqual(costs.fallback_calls, 1)

    def test_changed_normalizer_or_whitespace_stripping_tokens_rejected(self):
        from tokenizers import AddedToken, normalizers
        tokenizer = self.tokenizer()
        tokenizer.backend_tokenizer.normalizer = normalizers.NFD()
        self.assertFalse(supports_fragment_costs(tokenizer))
        tokenizer = self.tokenizer()
        tokenizer.backend_tokenizer.add_tokens([AddedToken('<strip>', lstrip=True)])
        self.assertFalse(supports_fragment_costs(tokenizer))

    def test_atom_snapshot_does_not_follow_content_mutations(self):
        tokenizer = self.tokenizer()
        atoms = self.atoms()
        costs = AtomUnionTokenCosts(tokenizer, atoms)
        expected = costs.count(atoms)
        atoms['0']['content'] = 'different text' * 100
        self.assertEqual(costs.count(atoms), expected)


if __name__ == '__main__':
    unittest.main()
