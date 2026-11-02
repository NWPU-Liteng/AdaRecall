"""Label-free aggregate statistics for a MobileMem-Omni serving run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


SCHEMA = 'omni-run-stats-v1'


def _number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{path} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{path} must be finite')
    return result


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def distribution(values):
    values = list(values)
    if not values:
        return None
    return {
        'count': len(values),
        'mean': statistics.fmean(values),
        'p50': _percentile(values, 0.50),
        'p90': _percentile(values, 0.90),
        'min': min(values),
        'max': max(values),
    }


def load_run_rows(run_dir):
    run_dir = Path(run_dir)
    paths = [path for path in sorted(run_dir.glob('user_*_shard_*.json'))
             if not path.name.endswith('.partial.json')]
    if not paths:
        raise FileNotFoundError(f'no complete user shard outputs in {run_dir}')
    rows = []
    identities = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get('complete') is not True:
            raise ValueError(f'incomplete user shard output: {path}')
        values = payload.get('rows')
        if not isinstance(values, list):
            raise ValueError(f'rows must be a list: {path}')
        for row in values:
            if not isinstance(row, dict):
                raise ValueError(f'row must be an object: {path}')
            person_id = str(row.get('person_id'))
            question_id = str(row.get('question_id'))
            identity = (person_id, question_id)
            if identity in identities:
                raise ValueError(f'duplicate question output: {identity}')
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError(f'run contains no question rows: {run_dir}')
    return paths, rows


def _counts(values):
    counts = Counter(str(value) if value is not None else 'missing' for value in values)
    return {name: counts[name] for name in sorted(counts)}


def _join_result(plan):
    if isinstance(plan.get('optimal'), bool):
        return plan
    failure = plan.get('failed_strict_search')
    if isinstance(failure, dict):
        return failure
    return None


def _rejection_sources(plan):
    sources = {}
    grounding = plan.get('grounding')
    if isinstance(grounding, dict):
        sources['grounding'] = grounding.get('rejections', [])
    shadow = plan.get('multi_witness_shadow')
    if isinstance(shadow, dict):
        sources['multi_witness_shadow'] = shadow.get('rejections', [])
        sources['independent_witness_search'] = shadow.get('search_rejections', [])
    for name, values in sources.items():
        if not isinstance(values, list):
            raise ValueError(f'proof_plan rejection source must be a list: {name}')
        if any(not isinstance(item, dict) for item in values):
            raise ValueError(f'proof_plan rejection entry must be an object: {name}')
    return sources


def build_stats(run_dir):
    run_dir = Path(run_dir)
    paths, rows = load_run_rows(run_dir)
    plans = []
    for row in rows:
        plan = row.get('proof_plan')
        if not isinstance(plan, dict):
            raise ValueError(f"proof_plan missing for {row.get('question_id')}")
        plans.append(plan)

    proof_status = _counts(plan.get('proof_status') for plan in plans)
    selection_mode = _counts(plan.get('selection_mode') for plan in plans)
    visibility_status = _counts(row.get('status') for row in rows)
    visibility_protocols = _counts(
        row.get('visibility_protocol') for row in rows)
    n_visible_sessions = []
    missing_n_visible_sessions = 0
    for row in rows:
        value = row.get('n_visible_sessions')
        if value is None:
            missing_n_visible_sessions += 1
            continue
        number = _number(value, 'n_visible_sessions')
        if number < 0 or not number.is_integer():
            raise ValueError('n_visible_sessions must be a non-negative integer')
        n_visible_sessions.append(int(number))
    visibility = {
        'protocols': visibility_protocols,
        'status': visibility_status,
        'no_visible_history': visibility_status.get('no_visible_history', 0),
        'n_visible_sessions': distribution(n_visible_sessions),
        'rows_missing_n_visible_sessions': missing_n_visible_sessions,
    }
    parse_status = _counts(
        (plan.get('query_parse') or {}).get('status')
        if isinstance(plan.get('query_parse'), dict) else None
        for plan in plans)
    answer_modes = []
    for plan in plans:
        contract = plan.get('query_contract')
        answer_modes.append(contract.get('answer_mode')
                            if isinstance(contract, dict) else None)
    answer_mode = _counts(answer_modes)
    contextual = sum(value == 'contextual' for value in answer_modes)
    non_ok_parse = len(rows) - sum(
        isinstance(plan.get('query_parse'), dict)
        and plan['query_parse'].get('status') == 'ok' for plan in plans)

    optimal_counts = {'true': 0, 'false': 0, 'unknown': 0}
    support_gaps = []
    missing_support_gaps = 0
    applicable = 0
    contextual_rows = 0
    for answer_mode_value, plan in zip(answer_modes, plans):
        if answer_mode_value == 'contextual':
            contextual_rows += 1
            continue
        applicable += 1
        result = _join_result(plan)
        if result is None:
            optimal_counts['unknown'] += 1
            continue
        optimal = result.get('optimal')
        if not isinstance(optimal, bool):
            optimal_counts['unknown'] += 1
            continue
        optimal_counts['true' if optimal else 'false'] += 1
        if not optimal:
            gap = result.get('support_gap')
            if gap is None:
                missing_support_gaps += 1
            else:
                support_gaps.append(_number(gap, 'join.support_gap'))
    resolved = optimal_counts['true'] + optimal_counts['false']
    join = {
        'applicable_questions': applicable,
        'not_applicable_contextual': contextual_rows,
        'optimal': optimal_counts,
        'optimal_ratio': (optimal_counts['true'] / resolved if resolved else None),
        'support_gap': distribution(support_gaps),
        'missing_support_gap': missing_support_gaps,
    }

    rejection_by_reason = Counter()
    rejection_by_source = Counter()
    rejection_rows = 0
    for plan in plans:
        row_rejections = 0
        for source, values in _rejection_sources(plan).items():
            rejection_by_source[source] += len(values)
            row_rejections += len(values)
            for item in values:
                reason = item.get('reason')
                rejection_by_reason[str(reason) if reason is not None else 'missing'] += 1
        rejection_rows += int(row_rejections > 0)
    witness_rejections = {
        'total': sum(rejection_by_source.values()),
        'rows_with_rejections': rejection_rows,
        'by_reason': {name: rejection_by_reason[name]
                      for name in sorted(rejection_by_reason)},
        'by_source': {name: rejection_by_source[name]
                      for name in sorted(rejection_by_source)},
    }

    omitted_values = []
    omitted_missing = 0
    for plan in plans:
        grounding = plan.get('grounding')
        values = (grounding.get('omitted_whole_record_ids_due_to_context')
                  if isinstance(grounding, dict) else None)
        if not isinstance(values, list):
            omitted_missing += 1
            continue
        omitted_values.append(len(values))
    omitted = {
        'rows_with_value': len(omitted_values),
        'rows_missing': omitted_missing,
        'total': sum(omitted_values),
        'average': statistics.fmean(omitted_values) if omitted_values else None,
        'distribution': distribution(omitted_values),
    }

    text_tokens, image_counts, processor_tokens = [], [], []
    final_render_rows = 0
    missing_render_fields = Counter()
    for plan in plans:
        render = plan.get('final_render')
        if not isinstance(render, dict):
            missing_render_fields['final_render'] += 1
            continue
        final_render_rows += 1
        values = {}
        for name, target in (
                ('text_tokens', text_tokens), ('image_count', image_counts),
                ('input_tokens', processor_tokens)):
            if name not in render:
                missing_render_fields[name] += 1
                continue
            number = _number(render[name], f'final_render.{name}')
            if number < 0:
                raise ValueError(f'final_render.{name} must be non-negative')
            target.append(number)
            values[name] = number
    final_evidence = {
        'rows_with_final_render': final_render_rows,
        'missing_fields': {name: missing_render_fields[name]
                           for name in sorted(missing_render_fields)},
        'text_tokens': distribution(text_tokens),
        'image_count': distribution(image_counts),
        'processor_tokens': distribution(processor_tokens),
    }

    stage_values = defaultdict(list)
    stage_profile_rows = 0
    for row in rows:
        profile = row.get('stage_profile')
        if not isinstance(profile, dict):
            continue
        stages = profile.get('stages')
        if not isinstance(stages, dict):
            raise ValueError(f"stage_profile.stages missing for {row.get('question_id')}")
        stage_profile_rows += 1
        for name, entry in stages.items():
            if not isinstance(entry, dict) or 'wall_ms' not in entry:
                raise ValueError(f'invalid stage profile entry: {name}')
            stage_values[name].append(_number(entry['wall_ms'], f'stage.{name}.wall_ms'))
    stages = {}
    for name in sorted(stage_values):
        summary = distribution(stage_values[name])
        stages[name] = {
            'count': summary['count'],
            'mean_ms': summary['mean'],
            'p50_ms': summary['p50'],
            'p90_ms': summary['p90'],
        }
    stage_profile = {
        'rows_with_stage_profile': stage_profile_rows,
        'rows_missing': len(rows) - stage_profile_rows,
        'stages': stages,
    }

    return {
        'schema': SCHEMA,
        'run_dir': str(run_dir.resolve()),
        'source_files': [path.name for path in paths],
        'questions': len(rows),
        'users': sorted({str(row.get('person_id')) for row in rows}),
        'proof_status': proof_status,
        'selection_mode': selection_mode,
        'visibility': visibility,
        'answer_mode': answer_mode,
        'answer_mode_contextual_ratio': contextual / len(rows),
        'parse_status': parse_status,
        'parse_status_non_ok_ratio': non_ok_parse / len(rows),
        'join': join,
        'witness_rejections': witness_rejections,
        'omitted_whole_record_ids_due_to_context': omitted,
        'final_evidence': final_evidence,
        'stage_profile': stage_profile,
    }


def _markdown_value(value):
    if value is None:
        return 'null'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value).replace('|', '\\|')


def render_markdown(stats):
    lines = [
        '# Run statistics',
        '',
        '| Section | Metric | Value |',
        '|---|---|---|',
    ]

    def add(section, metric, value):
        lines.append(f'| {section} | {metric} | {_markdown_value(value)} |')

    add('overview', 'questions', stats['questions'])
    add('overview', 'users', ','.join(stats['users']))
    for name, value in stats['proof_status'].items():
        add('proof_status', name, value)
    for name, value in stats['selection_mode'].items():
        add('selection_mode', name, value)
    for name, value in stats['visibility']['protocols'].items():
        add('visibility', f'protocol.{name}', value)
    for name, value in stats['visibility']['status'].items():
        add('visibility', f'status.{name}', value)
    add('visibility', 'no_visible_history',
        stats['visibility']['no_visible_history'])
    add('visibility', 'rows_missing_n_visible_sessions',
        stats['visibility']['rows_missing_n_visible_sessions'])
    visible = stats['visibility']['n_visible_sessions']
    if visible is None:
        add('visibility', 'n_visible_sessions.count', 0)
    else:
        for metric in ('mean', 'p50', 'p90', 'min', 'max'):
            add('visibility', f'n_visible_sessions.{metric}', visible[metric])
    for name, value in stats['answer_mode'].items():
        add('answer_mode', name, value)
    add('answer_mode', 'contextual_ratio', stats['answer_mode_contextual_ratio'])
    for name, value in stats['parse_status'].items():
        add('parse_status', name, value)
    add('parse_status', 'non_ok_ratio', stats['parse_status_non_ok_ratio'])

    join = stats['join']
    add('join', 'applicable_questions', join['applicable_questions'])
    add('join', 'not_applicable_contextual', join['not_applicable_contextual'])
    for name, value in join['optimal'].items():
        add('join', f'optimal_{name}', value)
    add('join', 'optimal_ratio', join['optimal_ratio'])
    for name, value in (join['support_gap'] or {}).items():
        add('join', f'support_gap_{name}', value)
    add('join', 'missing_support_gap', join['missing_support_gap'])

    rejections = stats['witness_rejections']
    add('witness_rejections', 'total', rejections['total'])
    add('witness_rejections', 'rows_with_rejections',
        rejections['rows_with_rejections'])
    for name, value in rejections['by_reason'].items():
        add('witness_rejections', f'reason.{name}', value)
    for name, value in rejections['by_source'].items():
        add('witness_rejections', f'source.{name}', value)

    omitted = stats['omitted_whole_record_ids_due_to_context']
    add('omitted_whole_record_ids', 'rows_with_value', omitted['rows_with_value'])
    add('omitted_whole_record_ids', 'rows_missing', omitted['rows_missing'])
    add('omitted_whole_record_ids', 'average', omitted['average'])

    final_evidence = stats['final_evidence']
    add('final_evidence', 'rows_with_final_render',
        final_evidence['rows_with_final_render'])
    for name, missing in final_evidence['missing_fields'].items():
        add('final_evidence', f'missing.{name}', missing)
    for name in ('text_tokens', 'image_count', 'processor_tokens'):
        values = final_evidence[name]
        if values is None:
            add('final_evidence', f'{name}.count', 0)
            continue
        for metric in ('mean', 'p50', 'p90', 'min', 'max'):
            add('final_evidence', f'{name}.{metric}', values[metric])

    stages = stats['stage_profile']
    add('stage_profile', 'rows_with_stage_profile',
        stages['rows_with_stage_profile'])
    add('stage_profile', 'rows_missing', stages['rows_missing'])
    for name, values in stages['stages'].items():
        for metric in ('mean_ms', 'p50_ms', 'p90_ms'):
            add('stage_profile', f'{name}.{metric}', values[metric])
    return '\n'.join(lines) + '\n'


def write_report(run_dir):
    run_dir = Path(run_dir)
    stats = build_stats(run_dir)
    json_path = run_dir / 'run_stats.json'
    markdown_path = run_dir / 'run_stats.md'
    temporary = json_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(stats, ensure_ascii=False, indent=2,
                                    sort_keys=True) + '\n')
    temporary.replace(json_path)
    markdown_path.write_text(render_markdown(stats))
    return json_path, markdown_path, stats


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    json_path, markdown_path, _ = write_report(args.run_dir)
    print(json_path)
    print(markdown_path)


if __name__ == '__main__':
    main()
