from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compile_omni_pim_routes import load_requests
from audit_omni_execution_cohort import analyze_graph
from audit_omni_smoke_closure import digest
from test_execution_route import graph_fixture


class PimRouteInputTests(unittest.TestCase):
    def write_fixture(self, root):
        def write(path, content):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
            return digest(path)
        graph_path, input_path = root / 'q.graph.json', root / 'q.input.json'
        graph = graph_fixture()
        graph_hash = write(graph_path, graph)
        input_hash = write(input_path, {'question_id': 'q', 'input_identity': 'immutable'})
        embedder = root / 'embedder'
        config_hash = write(embedder / 'config.json', {'hidden_size': 1024})
        smoke_path = root / 'smoke.json'
        smoke_hash = write(smoke_path, {'embedder': str(embedder), 'embedder_config_sha256': config_hash,
            'embedding_windows': 5, 'embedding_state_reuse': {'capacity_bytes': 1024},
            'evaluation_label_files_read': []})
        plan_hash = write(root / 'execution_plan.json', {'source_sha256': {},
            'evaluation_label_files_read': [], 'jobs': [{'person_id': '0', 'question_ids': ['q'],
                'smoke': str(smoke_path), 'smoke_sha256': smoke_hash}]})
        status_hash = write(root / 'execution_status.json', {'all_workers_completed': True,
            'all_workers_succeeded': True, 'execution_plan_sha256': plan_hash})
        produced = {'question_id': 'q', 'graph': str(graph_path), 'graph_sha256': graph_hash,
                    'input': str(input_path), 'input_sha256': input_hash}
        result_path = root / 'user_0/result.json'
        result_hash = write(result_path, {'complete': True, 'evaluation_label_files_read': [],
            'execution_plan_sha256': plan_hash, 'rows': [produced]})
        audit = {'complete': True, 'evaluation_label_files_read': [], 'requests': 1,
            'traced_requests': 1, 'complete_greedy_token_matches': 1, 'stop_reason_matches': 1,
            'execution_plan_sha256': plan_hash, 'execution_status_sha256': status_hash,
            'user_result_sha256': {str(result_path): result_hash},
            'users': [{'person_id': '0', 'rows': [{'question_id': 'q', 'token_ids_equal': True,
                'stop_reason_equal': True, 'graph_sha256': graph_hash, 'input_sha256': input_hash,
                'graph_analysis': analyze_graph(graph)}]}]}
        audit_path = root / 'audit.json'
        write(audit_path, audit)
        return audit_path, graph_path, audit

    def test_every_graph_and_input_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path, graph_path, unused = self.write_fixture(root)
            audit, requests, manifest = load_requests(audit_path, root)
            self.assertEqual(len(requests), audit['requests'])
            self.assertEqual(audit['route_reservations']['reserved_bytes'], 1024 + 5 * 1024 * 4)
            self.assertIn(str(graph_path), manifest)
            graph_path.write_text(graph_path.read_text() + ' ')
            with self.assertRaises(ValueError):
                load_requests(audit_path, root)

    def test_changed_denominator_or_replay_equality_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path, unused, audit = self.write_fixture(root)
            audit['requests'] = 2
            audit_path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                load_requests(audit_path, root)
            audit['requests'] = 1
            audit['users'][0]['rows'][0]['token_ids_equal'] = False
            audit_path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                load_requests(audit_path, root)

    def test_reduced_or_changed_graph_analysis_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path, unused, audit = self.write_fixture(root)
            audit['users'][0]['rows'][0]['graph_analysis']['dependency_edges'] = 0
            audit_path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                load_requests(audit_path, root)

    def test_incomplete_or_nonisolated_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / 'audit.json'
            audit.write_text('{"complete": false}')
            with self.assertRaises(ValueError):
                load_requests(audit, root)
            audit.write_text('{"complete": true, "users": [1], "evaluation_label_files_read": ["labels"]}')
            with self.assertRaises(ValueError):
                load_requests(audit, root)


if __name__ == '__main__':
    unittest.main()
