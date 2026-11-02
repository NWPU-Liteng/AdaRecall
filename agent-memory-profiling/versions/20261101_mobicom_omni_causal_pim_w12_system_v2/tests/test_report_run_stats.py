import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from report_run_stats import (SCHEMA, build_stats, distribution,
                              load_run_rows, render_markdown, write_report)


def make_plan(*, proof_status='verified', selection_mode='proof_join',
              answer_mode='structured', parse_status='ok', optimal=None,
              support_gap=None, failed_strict_search=None, grounding=None,
              shadow=None, final_render=None, operation_execution=None):
    plan = {
        'proof_status': proof_status,
        'selection_mode': selection_mode,
        'query_parse': {'status': parse_status},
        'query_contract': {'answer_mode': answer_mode},
    }
    if optimal is not None:
        plan['optimal'] = optimal
    if support_gap is not None:
        plan['support_gap'] = support_gap
    if failed_strict_search is not None:
        plan['failed_strict_search'] = failed_strict_search
    if grounding is not None:
        plan['grounding'] = grounding
    if shadow is not None:
        plan['multi_witness_shadow'] = shadow
    if final_render is not None:
        plan['final_render'] = final_render
    if operation_execution is not None:
        plan['operation_execution'] = operation_execution
    return plan


def make_row(question_id, plan, *, person_id='0', omitted=None,
             stages=None):
    row = {
        'person_id': person_id,
        'question_id': question_id,
        'proof_plan': plan,
    }
    if omitted is not None:
        grounding = plan.setdefault('grounding', {})
        grounding['omitted_whole_record_ids_due_to_context'] = omitted
    if stages is not None:
        row['stage_profile'] = {
            'stages': {name: {'wall_ms': value}
                       for name, value in stages.items()},
        }
    return row


def write_shard(directory, name, rows, *, complete=True):
    path = Path(directory) / name
    path.write_text(json.dumps({'complete': complete, 'rows': rows}))
    return path


class ReportRunStatsTests(unittest.TestCase):
    def test_distribution_uses_nearest_rank_percentiles(self):
        summary = distribution([40, 10, 30, 20])
        self.assertEqual(summary, {
            'count': 4,
            'mean': 25.0,
            'p50': 20,
            'p90': 40,
            'min': 10,
            'max': 40,
        })
        self.assertIsNone(distribution([]))

    def test_load_rows_ignores_partial_and_requires_unique_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', [
                make_row('0_q_1', make_plan()),
            ])
            write_shard(directory, 'user_0_shard_0.partial.json', [
                make_row('0_q_2', make_plan()),
            ], complete=False)

            paths, rows = load_run_rows(directory)

            self.assertEqual([path.name for path in paths],
                             ['user_0_shard_0.json'])
            self.assertEqual([row['question_id'] for row in rows], ['0_q_1'])

            write_shard(directory, 'user_1_shard_0.json', [
                make_row('0_q_1', make_plan(), person_id='0'),
            ])
            with self.assertRaisesRegex(ValueError, 'duplicate question'):
                load_run_rows(directory)

    def test_load_rows_rejects_incomplete_and_empty_runs(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaisesRegex(FileNotFoundError, 'no complete'):
                load_run_rows(empty_dir)

        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', [], complete=False)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_run_rows(directory)

            write_shard(directory, 'user_0_shard_0.json', [])
            with self.assertRaisesRegex(ValueError, 'no question rows'):
                load_run_rows(directory)

    def test_build_stats_covers_all_sections(self):
        rows = [
            make_row(
                '0_q_1',
                make_plan(
                    proof_status='verified',
                    selection_mode='source_located_proof_join',
                    optimal=True,
                    support_gap=0.0,
                    grounding={
                        'rejections': [{'reason': 'unlocated_claim'}],
                        'omitted_whole_record_ids_due_to_context': ['a', 'b'],
                    },
                    final_render={
                        'text_tokens': 10,
                        'image_count': 2,
                        'input_tokens': 100,
                    },
                ),
                stages={'contract_parse': 10, 'join': 5},
            ),
            make_row(
                '0_q_2',
                make_plan(
                    proof_status='no_complete_source_located_proof',
                    selection_mode='nonproof_retrieval_package',
                    answer_mode='contextual',
                    parse_status='contextual_query_must_remain_whole',
                ),
                omitted=[],
            ),
            make_row(
                '1_q_3',
                make_plan(
                    proof_status='no_complete_source_located_proof',
                    selection_mode='nonproof_retrieval_package',
                    parse_status='invalid_json',
                    failed_strict_search={
                        'optimal': False,
                        'support_gap': 0.25,
                    },
                    grounding={'rejections': []},
                    shadow={
                        'rejections': [
                            {'reason': 'invalid_witness'},
                            {'reason': 'unlocated_claim'},
                        ],
                        'search_rejections': [
                            {'reason': 'invalid_witness'},
                        ],
                    },
                    final_render={
                        'text_tokens': 20,
                        'image_count': 0,
                        'input_tokens': 200,
                    },
                ),
                person_id='1',
                omitted=['c'],
                stages={'contract_parse': 20, 'join': 7},
            ),
            make_row(
                '1_q_4',
                make_plan(
                    proof_status='source_located_not_semantically_verified',
                    selection_mode='nonproof_retrieval_package',
                    optimal=False,
                    support_gap=0.5,
                    grounding={'rejections': []},
                    final_render={
                        'text_tokens': 30,
                        'image_count': 1,
                    },
                ),
                person_id='1',
                omitted=[],
                stages={'contract_parse': 30},
            ),
            make_row(
                '0_q_5',
                {
                    'proof_status': None,
                    'selection_mode': None,
                    'query_parse': {'status': None},
                    'final_render': {
                        'text_tokens': 40,
                        'image_count': 3,
                        'input_tokens': 400,
                    },
                },
                omitted=[],
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', rows[:2])
            write_shard(directory, 'user_1_shard_0.json', rows[2:])
            stats = build_stats(directory)

        self.assertEqual(stats['schema'], SCHEMA)
        self.assertEqual(stats['questions'], 5)
        self.assertEqual(stats['users'], ['0', '1'])
        self.assertEqual(stats['source_files'],
                         ['user_0_shard_0.json', 'user_1_shard_0.json'])
        self.assertEqual(stats['proof_status'], {
            'missing': 1,
            'no_complete_source_located_proof': 2,
            'source_located_not_semantically_verified': 1,
            'verified': 1,
        })
        self.assertEqual(stats['selection_mode'], {
            'missing': 1,
            'nonproof_retrieval_package': 3,
            'source_located_proof_join': 1,
        })
        self.assertEqual(stats['answer_mode'], {
            'contextual': 1,
            'missing': 1,
            'structured': 3,
        })
        self.assertAlmostEqual(stats['answer_mode_contextual_ratio'], 0.2)
        self.assertEqual(stats['parse_status'], {
            'contextual_query_must_remain_whole': 1,
            'invalid_json': 1,
            'missing': 1,
            'ok': 2,
        })
        self.assertAlmostEqual(stats['parse_status_non_ok_ratio'], 0.6)

        join = stats['join']
        self.assertEqual(join['applicable_questions'], 4)
        self.assertEqual(join['not_applicable_contextual'], 1)
        self.assertEqual(join['optimal'], {
            'true': 1, 'false': 2, 'unknown': 1})
        self.assertAlmostEqual(join['optimal_ratio'], 1 / 3)
        self.assertEqual(join['missing_support_gap'], 0)
        self.assertEqual(join['support_gap'], {
            'count': 2,
            'mean': 0.375,
            'p50': 0.25,
            'p90': 0.5,
            'min': 0.25,
            'max': 0.5,
        })

        rejections = stats['witness_rejections']
        self.assertEqual(rejections['total'], 4)
        self.assertEqual(rejections['rows_with_rejections'], 2)
        self.assertEqual(rejections['by_reason'], {
            'invalid_witness': 2,
            'unlocated_claim': 2,
        })
        self.assertEqual(rejections['by_source'], {
            'grounding': 1,
            'independent_witness_search': 1,
            'multi_witness_shadow': 2,
        })

        omitted = stats['omitted_whole_record_ids_due_to_context']
        self.assertEqual(omitted['rows_with_value'], 5)
        self.assertEqual(omitted['rows_missing'], 0)
        self.assertEqual(omitted['total'], 3)
        self.assertEqual(omitted['average'], 0.6)
        self.assertEqual(omitted['distribution']['p90'], 2)

        final_evidence = stats['final_evidence']
        self.assertEqual(final_evidence['rows_with_final_render'], 4)
        self.assertEqual(final_evidence['missing_fields'],
                         {'final_render': 1, 'input_tokens': 1})
        self.assertEqual(final_evidence['text_tokens']['mean'], 25.0)
        self.assertEqual(final_evidence['text_tokens']['p90'], 40)
        self.assertEqual(final_evidence['image_count']['p50'], 1.0)
        self.assertEqual(final_evidence['processor_tokens']['count'], 3)
        self.assertAlmostEqual(final_evidence['processor_tokens']['mean'],
                               700 / 3)

        stage_profile = stats['stage_profile']
        self.assertEqual(stage_profile['rows_with_stage_profile'], 3)
        self.assertEqual(stage_profile['rows_missing'], 2)
        self.assertEqual(stage_profile['stages']['contract_parse'], {
            'count': 3,
            'mean_ms': 20.0,
            'p50_ms': 20.0,
            'p90_ms': 30.0,
        })
        self.assertEqual(stage_profile['stages']['join']['count'], 2)

    def test_join_unknown_and_missing_support_gap_are_reported(self):
        rows = [
            make_row('0_q_1', make_plan(optimal=False)),
            make_row('0_q_2', make_plan()),
            make_row('0_q_3', make_plan(answer_mode='contextual',
                                        optimal=False, support_gap=9.0)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', rows)
            join = build_stats(directory)['join']

        self.assertEqual(join['applicable_questions'], 2)
        self.assertEqual(join['not_applicable_contextual'], 1)
        self.assertEqual(join['optimal'], {
            'true': 0, 'false': 1, 'unknown': 1})
        self.assertEqual(join['optimal_ratio'], 0.0)
        self.assertEqual(join['missing_support_gap'], 1)
        self.assertIsNone(join['support_gap'])

    def test_visibility_and_no_visible_history_are_counted(self):
        first = make_row('0_q_1', make_plan())
        first.update({'status': 'ok', 'visibility_protocol': 'as_of_question',
                      'n_visible_sessions': 3})
        second = make_row('0_q_2', make_plan(
            proof_status='no_visible_history', selection_mode='no_visible_history',
            parse_status='not_run_no_visible_history'))
        second.update({'status': 'no_visible_history',
                       'visibility_protocol': 'as_of_question',
                       'n_visible_sessions': 0})

        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', [first, second])
            visibility = build_stats(directory)['visibility']

        self.assertEqual(visibility['protocols'], {'as_of_question': 2})
        self.assertEqual(visibility['status'], {
            'no_visible_history': 1, 'ok': 1})
        self.assertEqual(visibility['no_visible_history'], 1)
        self.assertEqual(visibility['rows_missing_n_visible_sessions'], 0)
        self.assertEqual(visibility['n_visible_sessions']['count'], 2)
        self.assertEqual(visibility['n_visible_sessions']['min'], 0)
        self.assertEqual(visibility['n_visible_sessions']['max'], 3)

    def test_operation_status_and_kind_are_counted(self):
        rows = [
            make_row('0_q_1', make_plan(operation_execution={
                'op0': {'status': 'ok', 'kind': 'difference', 'result': {}},
                'op1': {'status': 'not_executable', 'kind': 'comparison',
                        'result': None, 'reason': 'missing'},
            })),
            make_row('0_q_2', make_plan(operation_execution={})),
            make_row('0_q_3', make_plan()),
        ]
        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', rows)
            stats = build_stats(directory)

        operations = stats['operation_execution']
        self.assertEqual(operations['rows_with_execution'], 2)
        self.assertEqual(operations['rows_missing'], 1)
        self.assertEqual(operations['total'], 2)
        self.assertEqual(operations['by_status'], {
            'not_executable': 1, 'ok': 1})
        self.assertEqual(operations['by_kind'], {
            'comparison': 1, 'difference': 1})
        self.assertEqual(operations['by_kind_status'], {
            'comparison': {'not_executable': 1},
            'difference': {'ok': 1},
        })
        markdown = render_markdown(stats)
        self.assertIn('| operation_execution | status.ok | 1 |', markdown)
        self.assertIn('| operation_execution | kind.difference | 1 |', markdown)

    def test_invalid_shapes_raise_clear_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', [{
                'person_id': '0', 'question_id': '0_q_1'}])
            with self.assertRaisesRegex(ValueError, 'proof_plan missing'):
                build_stats(directory)

        with tempfile.TemporaryDirectory() as directory:
            row = make_row('0_q_1', make_plan(
                grounding={'rejections': 'not-a-list'}))
            write_shard(directory, 'user_0_shard_0.json', [row])
            with self.assertRaisesRegex(ValueError, 'rejection source'):
                build_stats(directory)

        with tempfile.TemporaryDirectory() as directory:
            row = make_row('0_q_1', make_plan(operation_execution=[]))
            write_shard(directory, 'user_0_shard_0.json', [row])
            with self.assertRaisesRegex(ValueError, 'operation_execution must be an object'):
                build_stats(directory)

        with tempfile.TemporaryDirectory() as directory:
            row = make_row('0_q_1', make_plan(
                final_render={'text_tokens': float('nan'),
                              'image_count': 0, 'input_tokens': 0}))
            write_shard(directory, 'user_0_shard_0.json', [row])
            with self.assertRaisesRegex(ValueError, 'must be finite'):
                build_stats(directory)

        with tempfile.TemporaryDirectory() as directory:
            row = make_row('0_q_1', make_plan())
            row['stage_profile'] = {'stages': {'join': {'wall_ms': True}}}
            write_shard(directory, 'user_0_shard_0.json', [row])
            with self.assertRaisesRegex(ValueError, 'must be numeric'):
                build_stats(directory)

    def test_markdown_and_json_reports_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            write_shard(directory, 'user_0_shard_0.json', [
                make_row('0_q_1', make_plan(
                    optimal=True,
                    grounding={
                        'rejections': [],
                        'omitted_whole_record_ids_due_to_context': [],
                    },
                    final_render={
                        'text_tokens': 12,
                        'image_count': 1,
                        'input_tokens': 24,
                    },
                ), stages={'join': 1.5}),
            ])

            json_path, markdown_path, stats = write_report(directory)
            written_stats = json.loads(json_path.read_text())
            markdown = markdown_path.read_text()

        self.assertEqual(written_stats, stats)
        self.assertEqual(stats['questions'], 1)
        self.assertTrue(markdown.startswith('# Run statistics\n'))
        self.assertEqual(markdown.count('|---|---|---|'), 1)
        self.assertIn('| join | optimal_true | 1 |', markdown)
        self.assertIn('| final_evidence | text_tokens.mean | 12.000000 |',
                      markdown)
        self.assertIn('| stage_profile | join.p90_ms | 1.500000 |', markdown)
        self.assertEqual(render_markdown(stats), markdown)
        self.assertFalse(json_path.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
