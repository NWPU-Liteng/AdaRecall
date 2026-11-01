"""Machine-checkable semantic sufficiency checks for single witness branches."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

from omni_multi_witness import (CANDIDATE_ONLY, SEMANTIC_VERIFIED, UNVERIFIED,
                                branch_source_families, source_family)


SCHEMA = 'witness-semantic-verifier-v1'
AUDIT_SCHEMA = 'witness-semantic-verifier-audit-v1'
DIRECT_SUPPORT = 'direct_support'
REASON_CODES = (
    DIRECT_SUPPORT,
    'partial_support',
    'wrong_entity',
    'wrong_time',
    'role_conflict',
    'missing_binding',
    'image_required',
    'no_support',
    'ambiguous',
)
VERDICTS = ('sufficient', 'insufficient', 'abstain')
SUPPORT_MODES = ('single_quote', 'explicit_generalization', 'repeated_instances',
                 'cause_outcome_pair', 'not_applicable')
EVIDENCE_ROLES = ('outcome', 'cause', 'instance')
DEMAND_OPERATIONS = (
    'single_fact',
    'compound_fact',
    'commonality',
    'reason',
    'latest_state',
    'first_event',
    'quantity_or_set',
    'temporal_difference',
    'image_required',
)
OPERATION_RULES = {
    'single_fact': 'one episode may suffice only when it directly states the requested value or relation',
    'compound_fact': ('a temporal or conjunctive demand requires evidence for every required event; '
                      'cite at least two distinct records, one for each event'),
    'commonality': ('a single episode is not a commonality; use explicit generalization cues such as '
                    '共同/都/每次/总是/一贯/通常/相似/同样/反复, or repeated_instances across at least '
                    'two source families with one exact shared property in every quote'),
    'reason': 'the quote must explicitly contain the cause or a causal connector, not only the outcome',
    'latest_state': 'the source must be the latest relevant state, not merely an earlier relevant statement',
    'first_event': 'the source must be the first relevant event, not merely any related event',
    'quantity_or_set': 'the quote must expose the complete requested quantity or set, not one member',
    'temporal_difference': 'the branch must expose a time endpoint; the difference itself needs joined evidence',
    'image_required': 'text cannot certify an image-scoped demand; return image_required',
}

VERIFIER_PROMPT = (
    '你是记忆证据充分性审计器，不回答问题、不补全事实、不使用输入之外的知识。'
    '输入只有一个 demand 和一个 branch。branch.text_sources 是原始对话中的逐字证据，'
    'record_id 是证据身份，session_interval 只是会话可见范围。'
    '只有当前 branch 自身已经明确给出 demand 要求的全部实体、属性、时间、关系和并列条件时，'
    '才能输出 sufficient。相关、相似、只覆盖一部分、需要常识补全或需要推断时，输出 insufficient。'
    '助手建议、假设或复述不是用户亲历事实；若 demand 要求用户事实，role_conflict 必须拒绝。'
    'demand.operation 与 demand.operation_rule 给出该需求的答案形态，必须一并满足。'
    '共同点、规律、偏好、最新状态、首次事件、数量集合都是一类操作，单条 episode 不能冒充。'
    '共同点若使用 repeated_instances，必须给出至少两个不同来源族的逐字 quote，并且这些 quote '
    '都包含同一个 shared_property。'
    '如果共同点没有“共同/都/每次”等显式概括词，但至少两个不同来源族的 episode 反复呈现同一属性，'
    '也应尝试 repeated_instances；shared_property 必须是原文中的最短逐字短语，且每条 quote 都包含它；'
    '若这些条件全部满足，输出 sufficient/direct_support。'
    '原因类需求可以使用 cause_outcome_pair：分别引用一条结果证据和一条原因证据，'
    '两条证据必须来自不同 record，并为每条 evidence 填写 evidence_role=outcome/cause。'
    '带“后/然后/接着/再”等顺序的复合需求必须为每个事件分别引用证据；不能只引用其中一个事件。'
    '文字无法决定且必须查看原始图片时输出 image_required，不得引用图片定位器的 caption 当作事实。'
    'sufficient 必须附至少一条逐字 quote，quote 必须来自给定 text 字段，不能改写、拼接或加省略号。'
    '严格输出紧凑 JSON，不得输出解释、Markdown 或额外字段。')

VERIFIER_JSON_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['verdict', 'reason_code', 'evidence'],
    'properties': {
        'verdict': {'enum': list(VERDICTS)},
        'reason_code': {'enum': list(REASON_CODES)},
        'support_mode': {'enum': list(SUPPORT_MODES)},
        'shared_property': {'type': 'string', 'maxLength': 80},
        'evidence': {
            'type': 'array',
            'maxItems': 4,
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': ['record_id', 'quote'],
                'properties': {
                'record_id': {'type': 'string', 'minLength': 1},
                'quote': {'type': 'string', 'minLength': 2},
                'evidence_role': {'enum': list(EVIDENCE_ROLES)},
            },
        },
        },
    },
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _branch_id(branch):
    identity = branch.get('witness_id') or branch.get('branch_id')
    if isinstance(identity, str) and identity:
        return identity
    records = branch.get('retrieval_record_ids') or branch.get('record_ids') or []
    payload = {'demand_id': branch.get('demand_id'), 'records': sorted(records)}
    return 'vb:' + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def demand_operation(demand):
    """Assign a conservative, closed answer-shape operation from demand text."""
    scope = demand.get('evidence_scope') or {}
    if scope.get('kind') == 'image_locator':
        return 'image_required'
    text = str(demand.get('text') or '') + ' ' + str(demand.get('query_quote') or '')
    if any(value in text for value in ('共同点', '共同', '相似', '相同', '一贯', '总是', '每次')):
        return 'commonality'
    if any(value in text for value in ('为什么', '原因', '为何')):
        return 'reason'
    if any(value in text for value in ('最新', '当前', '目前')):
        return 'latest_state'
    if any(value in text for value in ('第一次', '首次', '最先')):
        return 'first_event'
    if any(value in text for value in ('多少', '几', '哪两', '哪些', '几种')):
        return 'quantity_or_set'
    if any(value in text for value in ('多久', '多长时间', '前后')):
        return 'temporal_difference'
    temporal = ('后', '之后', '然后', '接着', '再')
    verbs = ('开始', '商量', '联系', '确认', '决定', '处理', '找', '买', '吃', '坐', '到',
             '出发', '返回', '讨论', '整理', '推荐', '下单', '看货', '复盘')
    if any(value in text for value in temporal) and sum(value in text for value in verbs) >= 2:
        return 'compound_fact'
    return 'single_fact'


def verifier_payload(demand, branch, text_sources):
    """Build one demand/branch verification input without the original question."""
    if not isinstance(demand, dict) or not isinstance(demand.get('text'), str):
        raise ValueError('verifier demand requires text')
    if not isinstance(branch, dict):
        raise ValueError('verifier branch must be an object')
    families = branch_source_families(branch)
    if not families:
        raise ValueError('verifier branch requires source-family identities')
    if not isinstance(text_sources, list) or not text_sources:
        raise ValueError('verifier branch requires at least one exact text source')
    operation = demand_operation(demand)
    return {
        'demand': {
            'demand_id': demand.get('demand_id'),
            'text': demand['text'],
            'query_quote': demand.get('query_quote'),
            'variables': demand.get('variables', []),
            'operation': operation,
            'operation_rule': OPERATION_RULES[operation],
        },
        'branch': {
            'branch_id': _branch_id(branch),
            'source_family': sorted(families),
            'text_sources': text_sources,
        },
        'output_contract': {
            'verdict': list(VERDICTS),
            'reason_codes': list(REASON_CODES),
            'quote_rule': 'exact substring of the cited branch text, no ellipsis or paraphrase',
        },
    }


def _exact_quote(text, quote):
    if not isinstance(text, str) or not isinstance(quote, str):
        return False
    value = quote.strip()
    return len(value) >= 2 and value != '...' and value in text


def _explicit_month_days(text):
    return [(int(month), int(day)) for month, day in re.findall(
        r'(?:\d{4}年)?(\d{1,2})月(\d{1,2})日', str(text or ''))]


def _focus_gate(operation, evidence, demand=None, temporal_context=None,
                support_mode='single_quote', shared_property=''):
    quotes = [item['quote'] for item in evidence]
    temporal_context = temporal_context or {}
    interval = str(temporal_context.get('session_interval') or '')
    demand_text = ''
    if isinstance(demand, dict):
        demand_text = str(demand.get('text') or '') + ' ' + str(demand.get('query_quote') or '')
    dates = _explicit_month_days(demand_text)
    if dates and interval:
        normalized = interval.replace('年', '-').replace('月', '-').replace('日', '')
        if not any(f'-{month:02d}-{day:02d}' in normalized or
                   f'-{month}-{day}' in normalized for month, day in dates):
            return 'focus_gate_date_mismatch'
    if operation == 'image_required':
        return 'focus_gate_image_requires_visual'
    if operation == 'compound_fact':
        records = {item['record_id'] for item in evidence}
        if len(evidence) < 2 or len(records) < 2:
            return 'focus_gate_incomplete_compound'
        return None
    if operation == 'commonality':
        if support_mode == 'repeated_instances':
            quotes = [item['quote'] for item in evidence]
            records = {item['record_id'] for item in evidence}
            families = {source_family(item['record_id']) for item in evidence}
            if (len(shared_property.strip()) < 2 or len(records) < 2
                    or len(families) < 2 or not all(
                        shared_property.strip() in quote for quote in quotes)):
                return 'focus_gate_insufficient_repetition'
            return None
        cues = ('共同', '都', '每次', '总是', '一贯', '通常', '相似', '同样', '反复')
        return None if any(cue in quote for quote in quotes for cue in cues) else (
            'focus_gate_missing_generalization')
    if operation == 'reason':
        if support_mode == 'cause_outcome_pair':
            outcomes = [item for item in evidence if item.get('evidence_role') == 'outcome']
            causes = [item for item in evidence if item.get('evidence_role') == 'cause']
            records = {item['record_id'] for item in evidence}
            cause_cues = ('因为', '所以', '由于', '导致', '是因为', '于是', '为了', '省得', '免得',
                          '晚上再', '还要', '才')
            if not outcomes or not causes or len(records) < 2:
                return 'focus_gate_missing_cause_outcome_roles'
            if not any(any(cue in item['quote'] for cue in cause_cues) for item in causes):
                return 'focus_gate_missing_cause'
            return None
        cues = ('因为', '所以', '由于', '导致', '是因为', '才', '于是')
        return None if any(cue in quote for quote in quotes for cue in cues) else (
            'focus_gate_missing_cause')
    if operation == 'latest_state':
        latest = temporal_context.get('latest_session_end')
        current = temporal_context.get('session_end')
        if latest and current and current != latest:
            return 'focus_gate_not_latest_state'
    if operation == 'first_event':
        earliest = temporal_context.get('earliest_session_start')
        current = temporal_context.get('session_start')
        if earliest and current and current != earliest:
            return 'focus_gate_not_first_event'
    return None


def validate_verifier_result(raw, branch, text_by_record, demand=None,
                             temporal_context=None):
    """Reject malformed, inconsistent, fabricated, or non-local semantic claims."""
    branch_records = set(branch.get('retrieval_record_ids') or branch.get('record_ids') or [])
    if not isinstance(raw, dict):
        return {'accepted': False, 'error': 'not_an_object', 'raw': raw}
    allowed_keys = {'verdict', 'reason_code', 'evidence', 'support_mode', 'shared_property'}
    if not {'verdict', 'reason_code', 'evidence'} <= set(raw) or not set(raw) <= allowed_keys:
        return {'accepted': False, 'error': 'schema_keys', 'raw': raw}
    verdict, reason = raw.get('verdict'), raw.get('reason_code')
    evidence = raw.get('evidence')
    support_mode = raw.get('support_mode', 'single_quote')
    shared_property = raw.get('shared_property', '')
    if support_mode not in SUPPORT_MODES or not isinstance(shared_property, str):
        return {'accepted': False, 'error': 'support_mode_value', 'raw': raw}
    if verdict not in VERDICTS or reason not in REASON_CODES or not isinstance(evidence, list):
        return {'accepted': False, 'error': 'closed_value', 'raw': raw}
    if verdict == 'sufficient' and reason != DIRECT_SUPPORT:
        return {'accepted': False, 'error': 'sufficient_reason_mismatch', 'raw': raw}
    if verdict != 'sufficient' and reason == DIRECT_SUPPORT:
        return {'accepted': False, 'error': 'nonsufficient_direct_reason', 'raw': raw}
    if verdict == 'abstain' and reason not in {'ambiguous', 'image_required'}:
        return {'accepted': False, 'error': 'abstain_reason_mismatch', 'raw': raw}

    checked = []
    for item in evidence:
        if (not isinstance(item, dict)
                or not {'record_id', 'quote'} <= set(item)
                or not set(item) <= {'record_id', 'quote', 'evidence_role'}):
            return {'accepted': False, 'error': 'evidence_schema', 'raw': raw}
        record_id, quote = item.get('record_id'), item.get('quote')
        role = item.get('evidence_role')
        if role is not None and role not in EVIDENCE_ROLES:
            return {'accepted': False, 'error': 'evidence_role_value', 'raw': raw}
        if record_id not in branch_records or record_id not in text_by_record:
            return {'accepted': False, 'error': 'evidence_not_in_branch', 'raw': raw}
        if not _exact_quote(text_by_record[record_id], quote):
            return {'accepted': False, 'error': 'fabricated_or_nonliteral_quote', 'raw': raw}
        checked.append({'record_id': record_id, 'quote': quote.strip(), 'evidence_role': role})

    if verdict == 'sufficient' and not checked:
        return {'accepted': False, 'error': 'sufficient_without_evidence', 'raw': raw}
    operation = demand_operation(demand) if isinstance(demand, dict) else 'single_fact'
    if verdict == 'sufficient':
        mode_operations = {
            'explicit_generalization': {'commonality'},
            'repeated_instances': {'commonality'},
            'cause_outcome_pair': {'reason'},
            'not_applicable': set(),
        }
        if support_mode in mode_operations and operation not in mode_operations[support_mode]:
            return {
                'accepted': False,
                'error': 'support_mode_operation_mismatch',
                'verdict': verdict,
                'reason_code': reason,
                'demand_operation': operation,
                'evidence': checked,
                'raw': raw,
            }
    focus_error = (_focus_gate(operation, checked, demand=demand,
                               temporal_context=temporal_context,
                               support_mode=support_mode,
                               shared_property=shared_property)
                   if verdict == 'sufficient' else None)
    if focus_error is not None:
        return {
            'accepted': False,
            'error': focus_error,
            'verdict': verdict,
            'reason_code': reason,
            'demand_operation': operation,
            'evidence': checked,
            'raw': raw,
        }
    return {
        'accepted': verdict == 'sufficient',
        'error': None if verdict == 'sufficient' else reason,
        'verdict': verdict,
        'reason_code': reason,
        'demand_operation': operation,
        'evidence': checked,
        'raw': raw,
    }


def audit_verified_frontier(demand_results, *, required_independent_branches=2,
                            verifier_metadata=None):
    """Audit machine-checked certificates without promoting model assertions alone."""
    if type(required_independent_branches) is not int or required_independent_branches < 2:
        raise ValueError('at least two independent verified branches are required')
    per_demand = []
    for certificate_id in sorted(demand_results, key=str):
        branches = demand_results[certificate_id]
        accepted = [item for item in branches if item.get('accepted') is True]
        families = {item['source_family'] for item in accepted
                    if isinstance(item.get('source_family'), str) and item['source_family']}
        reasons = Counter(item.get('reason_code', item.get('error')) for item in branches)
        first = branches[0] if branches else {}
        per_demand.append({
            'certificate_id': str(certificate_id),
            'question_id': first.get('question_id'),
            'demand_id': first.get('demand_id'),
            'branch_count': len(branches),
            'accepted_branch_count': len(accepted),
            'accepted_source_families': sorted(families),
            'verified_or': len(families) >= required_independent_branches,
            'reason_counts': dict(sorted(reasons.items())),
        })
    return {
        'schema': AUDIT_SCHEMA,
        'required_independent_branches': required_independent_branches,
        'demand_count': len(per_demand),
        'demands_with_semantically_verified_or': sum(item['verified_or'] for item in per_demand),
        'accepted_branch_count': sum(item['accepted_branch_count'] for item in per_demand),
        'per_demand': per_demand,
        'verifier_metadata': verifier_metadata or {},
        'certificate_scope': 'single verifier; source-family disjointness only',
        'candidate_only_upgraded': False,
        'answer_path_changed': False,
    }


def verified_branch_record(branch, validation, *, verifier_metadata=None):
    """Return a branch record that is upgraded only after deterministic acceptance."""
    if validation.get('accepted') is not True:
        return {
            **branch,
            'verification_state': CANDIDATE_ONLY,
            'semantic_sufficiency': UNVERIFIED,
            'verifier_result': validation,
        }
    return {
        **branch,
        'verification_state': SEMANTIC_VERIFIED,
        'semantic_sufficiency': 'text_quote_verified',
        'verifier_result': validation,
        'verifier_metadata': verifier_metadata or {},
    }
