"""Evaluation-only empirical coverage for T3.3 conformal candidate policies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_evidence_recall import build_image_ref_map, digest, load_labels
from fit_calibration import (_question_coverage_score, _question_rows,
                             load_candidate_dumps, load_source_objects,
                             validate_calibrator)
from pim_backend.candidate_policy import threshold_for_alpha


SCHEMA = 'omni-conformal-coverage-v1'
COVERAGE_NOTE = '覆盖保证以问题可交换为前提，且只针对 48 个候选池。'


def _covered(g_score, tau):
    if math.isinf(g_score) and g_score < 0:
        return False
    return g_score >= tau


def summarize_coverage(g_scores, calibrator):
    """Summarize empirical coverage for every alpha stored in a calibrator."""
    n = len(g_scores)
    coverage = {}
    for alpha in calibrator.get('alphas', []):
        tau = threshold_for_alpha(calibrator, alpha)
        covered = sum(_covered(score, tau) for score in g_scores)
        empirical = covered / n if n else None
        coverage[str(alpha)] = {
            'alpha': float(alpha),
            'target_coverage': 1.0 - float(alpha),
            'tau': (None if math.isinf(tau) and tau < 0 else tau),
            'tau_encoding': 'JSON null denotes negative infinity',
            'n': n,
            'covered': covered,
            'empirical_coverage': empirical,
            'difference_from_target': (
                empirical - (1.0 - float(alpha))
                if empirical is not None else None),
        }
    pool_miss_count = sum(
        math.isinf(score) and score < 0 for score in g_scores)
    return {
        'n': n,
        'pool_miss_count': pool_miss_count,
        'pool_miss_ratio': pool_miss_count / n if n else None,
        'coverage': coverage,
    }


def build_coverage_report(dump_root, labels_path, calibrator_path,
                          index_root=None):
    calibrator = validate_calibrator(
        json.loads(Path(calibrator_path).read_text()))
    if calibrator.get('split') != 'development':
        raise ValueError('coverage calibrator must be fitted on development split')
    if calibrator.get('candidate_width') != 48:
        raise ValueError('coverage calibrator candidate width must be 48')
    questions, dump_paths, users = load_candidate_dumps(
        dump_root, required_split='validation')
    labels = load_labels(labels_path)
    source_root, sources = load_source_objects(
        labels_path, users, index_root=index_root)
    image_maps = {
        user_id: build_image_ref_map(objects)
        for user_id, objects in sources.items()
    }

    g_scores = []
    questions_without_evidence = 0
    unmapped_image_ref_count = 0
    for identity, payload in sorted(questions.items()):
        person_id, _ = identity
        label = labels.get(identity)
        if label is None:
            raise ValueError(f'validation candidate question has no label: {identity}')
        rows, sessions, images, unmapped_images = _question_rows(
            payload, label, sources[person_id], image_maps[person_id])
        score = _question_coverage_score(
            rows, sessions, images, unmapped_images)
        if score is None:
            questions_without_evidence += 1
        else:
            g_scores.append(score)
        unmapped_image_ref_count += len(unmapped_images)

    summary = summarize_coverage(g_scores, calibrator)
    return {
        'schema': SCHEMA,
        'split': 'validation',
        'users': users,
        'inputs': {
            'dump_root': str(Path(dump_root).resolve()),
            'candidate_dump_count': len(dump_paths),
            'labels': {
                'path': str(Path(labels_path).resolve()),
                'sha256': digest(labels_path),
            },
            'calibrator': {
                'path': str(Path(calibrator_path).resolve()),
                'sha256': calibrator['sha256'],
            },
            'source_root': str(Path(source_root).resolve()),
        },
        'questions_total': len(questions),
        'questions_without_evidence_units': questions_without_evidence,
        'unmapped_evidence_image_refs': unmapped_image_ref_count,
        'pool_miss_count': summary['pool_miss_count'],
        'pool_miss_ratio': summary['pool_miss_ratio'],
        'coverage': summary['coverage'],
        'coverage_rule': (
            'per question g_q >= tau_alpha; g_q=-inf is never counted as covered'),
        'coverage_note': COVERAGE_NOTE,
    }


def write_coverage_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False) + '\n')
    temporary.replace(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--calibrator', type=Path, required=True)
    parser.add_argument('--index-root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_coverage_report(
        args.dump, args.labels, args.calibrator, index_root=args.index_root)
    output = write_coverage_report(report, args.out)
    print(output)
    print(json.dumps({
        'users': report['users'],
        'pool_miss_ratio': report['pool_miss_ratio'],
        'coverage': {
            key: {
                'n': value['n'],
                'covered': value['covered'],
                'empirical_coverage': value['empirical_coverage'],
                'target_coverage': value['target_coverage'],
            }
            for key, value in report['coverage'].items()
        },
    }, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
