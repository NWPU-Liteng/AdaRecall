"""Execute query operations over source-located witness values."""
from __future__ import annotations

import math

from omni_prepare import parse_time
from typed_values import Interval, parse_value


SELECTION_MIN_CUES = (
    '最早', '最少', '最小', '最低', '最短', '最慢', '最便宜', '最轻',
)
UNIT_LABELS = {'day': '天', 'second': '秒', 'number': '数值', 'count': '个'}


def _interval(value: Interval) -> dict:
    return {'lo': value.lo, 'hi': value.hi, 'unit': value.unit}


def _as_interval(value):
    if not isinstance(value, dict):
        return None
    lo, hi, unit = value.get('lo'), value.get('hi'), value.get('unit')
    if (isinstance(lo, bool) or isinstance(hi, bool)
            or not isinstance(lo, (int, float)) or not isinstance(hi, (int, float))
            or not math.isfinite(float(lo)) or not math.isfinite(float(hi))
            or float(lo) > float(hi) or not isinstance(unit, str) or not unit):
        return None
    return Interval(float(lo), float(hi), unit)


def _session_window(obj, graph):
    times = obj.get('time') if isinstance(obj, dict) else None
    if isinstance(times, list) and len(times) == 2:
        try:
            start, end = parse_time(times[0]), parse_time(times[1])
            if end >= start:
                return start, end
        except (TypeError, ValueError):
            pass
    session_id = obj.get('session_id') if isinstance(obj, dict) else None
    if isinstance(session_id, str) and isinstance(graph, dict):
        session = next((item for item in graph.get('sessions', [])
                        if str(item.get('session_id')) == session_id), None)
        if session is not None:
            start_value = session.get('visibility_start', session.get('start_time'))
            end_value = session.get('visibility_end', session.get('end_time'))
            if isinstance(start_value, str) and isinstance(end_value, str):
                start, end = parse_time(start_value), parse_time(end_value)
                if end >= start:
                    return start, end
    return None


def _demand_intervals(demand_id, witnesses_by_demand, objects, graph):
    values = []
    failures = []
    for witness in witnesses_by_demand.get(demand_id, []):
        for value in witness.get('values', []):
            if not isinstance(value, dict) or value.get('type') == 'text':
                continue
            obj = objects.get(value.get('object_id'))
            if obj is None:
                failures.append('value object missing')
                continue
            window = _session_window(obj, graph)
            if window is None:
                failures.append('session interval missing')
                continue
            parsed = parse_value(value, window)
            if parsed is None:
                failures.append(f"无法解析 {value.get('name', demand_id)}")
                continue
            label = str(value.get('name') or demand_id)
            values.append((parsed, f'{label}={value.get("value")}'))
    unique = {}
    for interval, summary in values:
        unique.setdefault((interval.lo, interval.hi, interval.unit), (interval, summary))
    if not unique:
        reason = failures[0] if failures else f'{demand_id} 缺少可解析值'
        return None, reason
    if len(unique) > 1:
        return None, f'{demand_id} 存在多个不同可解析值'
    return next(iter(unique.values()))


def _resolve_interval(identity, witnesses_by_demand, objects, graph, results):
    if identity in results:
        entry = results[identity]
        if entry.get('status') != 'ok':
            return None, f'{identity} 不可执行：{entry.get("reason", "unknown")}'
        result = entry.get('result')
        interval = _as_interval(result.get('interval') if isinstance(result, dict) else None)
        if interval is None:
            return None, f'{identity} 没有可计算区间'
        return interval, f'{identity}=[{_number(interval.lo)}, {_number(interval.hi)}] {interval.unit}'
    return _demand_intervals(identity, witnesses_by_demand, objects, graph)


def _number(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else f'{value:.6f}'.rstrip('0').rstrip('.')


def _display_interval(value):
    interval = _as_interval(value)
    if interval is None:
        return 'unknown'
    if interval.lo == interval.hi:
        rendered = _number(interval.lo)
    else:
        rendered = f'[{_number(interval.lo)}, {_number(interval.hi)}]'
    return f'{rendered} {UNIT_LABELS.get(interval.unit, interval.unit)}'


def render_execution_results(results):
    """Render deterministic operation results for the Reader prompt."""
    if not results:
        return ''
    lines = ['程序计算结果（由原文定位值确定性计算，区间表示时间粒度带来的不确定性）：']
    for operation_id, entry in results.items():
        kind = entry.get('kind') or 'operation'
        if entry.get('status') != 'ok':
            lines.append(f"- {operation_id}: not_executable（原因：{entry.get('reason', 'unknown')}）")
            continue
        result = entry.get('result') or {}
        inputs = '；'.join(str(value) for value in result.get('inputs', []))
        if kind == 'difference':
            detail = f"difference = {_display_interval(result.get('interval'))}"
        elif kind == 'comparison':
            detail = (f"comparison = {result.get('order')}"
                      f"（left={_display_interval(result.get('left'))}，"
                      f"right={_display_interval(result.get('right'))}）")
        elif kind == 'selection':
            detail = (f"selection = {result.get('selected')} "
                      f"{_display_interval(result.get('interval'))}")
        elif kind == 'aggregation':
            detail = f"aggregation = count_lower_bound={result.get('count_lower_bound')}"
        else:
            detail = kind
        lines.append(f"- {operation_id}: {detail}" + (f"；输入：{inputs}" if inputs else ''))
    return '\n'.join(lines)


def _difference(inputs, witnesses_by_demand, objects, graph, results):
    if len(inputs) != 2:
        return None, 'difference 需要两个输入'
    resolved = [_resolve_interval(identity, witnesses_by_demand, objects, graph, results)
                for identity in inputs]
    if any(value is None for value, _ in resolved):
        return None, '；'.join(reason for value, reason in resolved if value is None)
    left, left_text = resolved[0]
    right, right_text = resolved[1]
    if left.unit != right.unit:
        return None, 'difference 输入单位不一致'
    value = Interval(left.lo - right.hi, left.hi - right.lo, left.unit)
    return {'kind': 'difference', 'interval': _interval(value),
            'inputs': [left_text, right_text]}, None


def _comparison(inputs, witnesses_by_demand, objects, graph, results):
    if len(inputs) != 2:
        return None, 'comparison 需要两个输入'
    resolved = [_resolve_interval(identity, witnesses_by_demand, objects, graph, results)
                for identity in inputs]
    if any(value is None for value, _ in resolved):
        return None, '；'.join(reason for value, reason in resolved if value is None)
    left, left_text = resolved[0]
    right, right_text = resolved[1]
    if left.unit != right.unit:
        return None, 'comparison 输入单位不一致'
    if left.hi < right.lo:
        order = 'left_before_right'
    elif right.hi < left.lo:
        order = 'right_before_left'
    else:
        order = 'undetermined'
    return {'kind': 'comparison', 'order': order,
            'left': _interval(left), 'right': _interval(right),
            'inputs': [left_text, right_text]}, None


def _selection(operation, witnesses_by_demand, objects, graph, results):
    inputs = operation.get('inputs', [])
    if len(inputs) < 2:
        return None, 'selection 至少需要两个输入'
    resolved = [_resolve_interval(identity, witnesses_by_demand, objects, graph, results)
                for identity in inputs]
    if any(value is None for value, _ in resolved):
        return None, '；'.join(reason for value, reason in resolved if value is None)
    intervals = [value for value, _ in resolved]
    if len({value.unit for value in intervals}) != 1:
        return None, 'selection 输入单位不一致'
    choose_min = any(cue in operation.get('query_quote', '') for cue in SELECTION_MIN_CUES)
    winner = None
    for index, candidate in enumerate(intervals):
        others = [value for other_index, value in enumerate(intervals)
                  if other_index != index]
        if choose_min and all(candidate.hi < value.lo for value in others):
            winner = index
        if not choose_min and all(candidate.lo > value.hi for value in others):
            winner = index
        if winner is not None:
            break
    if winner is None:
        return None, 'selection 区间排序不唯一'
    value = intervals[winner]
    return {'kind': 'selection', 'selected': inputs[winner], 'interval': _interval(value),
            'inputs': [text for _, text in resolved]}, None


def _aggregation(operation, witnesses_by_demand):
    inputs = operation.get('inputs', [])
    if len(inputs) != 1:
        return None, 'aggregation 只支持一个 demand 输入'
    demand_id = inputs[0]
    witnesses = witnesses_by_demand.get(demand_id)
    if not witnesses:
        return None, f'{demand_id} 缺少已接受见证'
    values = sorted({str(value) for witness in witnesses
                     for value in witness.get('bindings', {}).values()
                     if isinstance(value, str) and value})
    return {'kind': 'aggregation', 'count_lower_bound': len(values),
            'inputs': [f'{demand_id}.bindings={values}']}, None


def execute(operations, witnesses_by_demand, objects, graph) -> dict[str, dict]:
    if not isinstance(operations, list):
        raise ValueError('operations must be a list')
    results = {}
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        operation_id = operation.get('operation_id')
        if not isinstance(operation_id, str) or not operation_id:
            continue
        try:
            if operation.get('kind') == 'difference':
                result, reason = _difference(operation.get('inputs', []), witnesses_by_demand,
                                             objects, graph, results)
            elif operation.get('kind') == 'comparison':
                result, reason = _comparison(operation.get('inputs', []), witnesses_by_demand,
                                             objects, graph, results)
            elif operation.get('kind') == 'selection':
                result, reason = _selection(operation, witnesses_by_demand, objects, graph,
                                            results)
            elif operation.get('kind') == 'aggregation':
                result, reason = _aggregation(operation, witnesses_by_demand)
            else:
                result, reason = None, 'unknown operation kind'
        except (KeyError, TypeError, ValueError) as error:
            result, reason = None, f'invalid operation input: {error}'
        if result is None:
            results[operation_id] = {
                'status': 'not_executable', 'kind': operation.get('kind'),
                'result': None, 'reason': reason or 'not executable',
            }
        else:
            results[operation_id] = {
                'status': 'ok', 'kind': operation.get('kind'),
                'result': result, 'reason': '',
            }
    return results
