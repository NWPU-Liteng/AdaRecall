from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from operation_executor import execute, render_execution_results


GRAPH = {'sessions': []}


def text_object(identity, text):
    return {'object_id': identity, 'kind': 'text', 'session_id': 's0',
            'raw_text': text,
            'time': ['2025-01-01 08:00:00', '2025-12-31 22:00:00']}


def witness(demand_id, object_id, text, value, value_type='date', bindings=None):
    return {
        'demand_id': demand_id,
        'bindings': bindings or {},
        'values': [{'name': f'{demand_id}.date', 'type': value_type,
                    'value': value, 'record_id': 'r0', 'text_index': 0,
                    'object_id': object_id}],
        '_text': text,
    }


def fixture():
    objects = {
        'text:a': text_object('text:a', '2025-01-01'),
        'text:b': text_object('text:b', '2025-01-03'),
        'text:c': text_object('text:c', '2025-01-10'),
        'text:d': text_object('text:d', '2025-01-14'),
    }
    witnesses = {
        'a': [witness('a', 'text:a', '2025-01-01', '2025-01-01')],
        'b': [witness('b', 'text:b', '2025-01-03', '2025-01-03')],
        'c': [witness('c', 'text:c', '2025-01-10', '2025-01-10')],
        'd': [witness('d', 'text:d', '2025-01-14', '2025-01-14')],
    }
    return objects, witnesses


class OperationExecutorTests(unittest.TestCase):
    def test_difference_returns_an_interval(self):
        objects, witnesses = fixture()
        result = execute([{'operation_id': 'op0', 'kind': 'difference',
                           'inputs': ['a', 'b'], 'query_quote': 'how long'}],
                         witnesses, objects, GRAPH)
        self.assertEqual(result['op0']['status'], 'ok')
        self.assertEqual(result['op0']['result']['interval'],
                         {'lo': -2.0, 'hi': -2.0, 'unit': 'day'})

    def test_comparison_is_disjoint_or_undetermined(self):
        objects, witnesses = fixture()
        operations = [
            {'operation_id': 'left', 'kind': 'difference', 'inputs': ['b', 'a'],
             'query_quote': 'first'},
            {'operation_id': 'right', 'kind': 'difference', 'inputs': ['d', 'c'],
             'query_quote': 'second'},
            {'operation_id': 'cmp', 'kind': 'comparison', 'inputs': ['left', 'right'],
             'query_quote': 'which'},
        ]
        result = execute(operations, witnesses, objects, GRAPH)
        self.assertEqual(result['cmp']['result']['order'], 'left_before_right')
        objects['text:e'] = text_object('text:e', '2025-01-01')
        witnesses['e'] = [witness('e', 'text:e', '2025-01-01', '2025-01-01')]
        result = execute([{'operation_id': 'cmp', 'kind': 'comparison',
                           'inputs': ['a', 'e'], 'query_quote': 'which'}],
                         witnesses, objects, GRAPH)
        self.assertEqual(result['cmp']['result']['order'], 'undetermined')

    def test_missing_input_is_not_executable(self):
        objects, witnesses = fixture()
        result = execute([{'operation_id': 'op0', 'kind': 'difference',
                           'inputs': ['a', 'missing'], 'query_quote': 'how long'}],
                         witnesses, objects, GRAPH)
        self.assertEqual(result['op0']['status'], 'not_executable')
        self.assertIn('缺少可解析值', result['op0']['reason'])

    def test_later_operation_can_consume_an_earlier_result(self):
        objects, witnesses = fixture()
        operations = [
            {'operation_id': 'op0', 'kind': 'difference', 'inputs': ['b', 'a'],
             'query_quote': 'first'},
            {'operation_id': 'op1', 'kind': 'difference', 'inputs': ['d', 'c'],
             'query_quote': 'second'},
            {'operation_id': 'op2', 'kind': 'comparison', 'inputs': ['op0', 'op1'],
             'query_quote': 'which longer'},
        ]
        result = execute(operations, witnesses, objects, GRAPH)
        self.assertEqual(result['op2']['status'], 'ok')
        self.assertEqual(result['op2']['result']['order'], 'left_before_right')

    def test_selection_and_aggregation_are_deterministic(self):
        objects, witnesses = fixture()
        witnesses['a'].append(witness('a', 'text:a', '2025-01-01', '2025-01-01',
                                      bindings={'project': 'Atlas'}))
        witnesses['b'].append(witness('b', 'text:b', '2025-01-03', '2025-01-03',
                                      bindings={'project': 'Borealis'}))
        operations = [
            {'operation_id': 'op0', 'kind': 'selection', 'inputs': ['a', 'b'],
             'query_quote': '哪个最早'},
            {'operation_id': 'op1', 'kind': 'aggregation', 'inputs': ['a'],
             'query_quote': '多少'},
        ]
        result = execute(operations, witnesses, objects, GRAPH)
        self.assertEqual(result['op0']['result']['selected'], 'a')
        self.assertEqual(result['op1']['result']['count_lower_bound'], 1)

    def test_execution_render_keeps_success_and_failure_visible(self):
        objects, witnesses = fixture()
        results = execute([
            {'operation_id': 'op0', 'kind': 'difference', 'inputs': ['a', 'b'],
             'query_quote': 'how long'},
            {'operation_id': 'op1', 'kind': 'comparison', 'inputs': ['a', 'missing'],
             'query_quote': 'which'},
        ], witnesses, objects, GRAPH)
        rendered = render_execution_results(results)
        self.assertIn('程序计算结果', rendered)
        self.assertIn('op0: difference = -2 天', rendered)
        self.assertIn('op1: not_executable（原因：', rendered)
        self.assertIn('缺少可解析值', rendered)
        self.assertEqual(render_execution_results({}), '')


if __name__ == '__main__':
    unittest.main()
