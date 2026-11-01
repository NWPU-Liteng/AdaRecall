"""AND/OR witness domains with explicit verification and materialization costs."""
from __future__ import annotations

import hashlib
import json
import math
import time


MULTI_WITNESS_PROMPT = (
    '你是多见证证据构造器，不回答问题、不验证答案。仅用给定来源索引构造需求见证。'
    '严格输出紧凑JSON {"witnesses":[{"demand_id":"d0","record_ids":["r0"],'
    '"anchors":[{"record_id":"r0","text_index":0}],"bindings":[],"source_relations":[]}]}。'
    '每个demand优先给出2到5个互不相同、各自足以支持该需求的替代见证；无法找到第二个时只给一个。'
    '替代来源必须拆成不同witness，不能把相似候选塞进同一witness；同一witness只联合真正互补、缺一不可的来源。'
    'text_index引用record的完整原文atom；图片只用region_index。不要复制原文，不要输出object_id。'
    'bindings只绑定demand的variables声明的连接键，值必须逐字出现在引用atom中。'
    'source_relations只引用原文中明确出现的关系。无法定位则不要输出该见证。'
    '严禁解释、长引用、回答问题或新增字段。')


INDEPENDENT_WITNESS_PROMPT = (
    '你是独立见证搜索器，不回答问题、不验证答案。只使用给定候选和指定来源族构造替代见证。'
    '输入中的excluded_source_families已经被现有见证占用，禁止使用这些来源族中的任何候选。'
    '每个demand只输出一个完整、可独立支持该需求的替代见证；如果候选不足或无法独立支持，输出空数组。'
    '替代见证的每个record必须来自不同的、未出现在excluded_source_families中的来源族；'
    '不能把同一来源族的不同atom冒充成独立证据。'
    '严格输出紧凑JSON {"witnesses":[{"demand_id":"d0","record_ids":["r0"],'
    '"anchors":[{"record_id":"r0","text_index":0}],"bindings":[],"source_relations":[]}]}。'
    'text_index引用record的完整原文atom；图片只用region_index。不要复制原文，不要输出object_id。'
    'bindings只绑定demand的variables声明的连接键，值必须逐字出现在引用atom中。'
    'source_relations只引用原文中明确出现的关系。无法定位或无法独立于已有来源时，不要输出该见证。'
    '严禁解释、长引用、回答问题或新增字段。')


SCHEMA = 'and-or-proof-hypergraph-v1'
SOURCE_LOCATED = 'source_located'
SEMANTIC_VERIFIED = 'semantically_verified'
CANDIDATE_ONLY = 'candidate_only'
UNVERIFIED = 'unverified'


def source_family(record_id):
    """Normalize a record identity to its session/source family."""
    if not isinstance(record_id, str) or not record_id:
        return None
    value = record_id.split('|', 1)[0]
    if ':turn:' in value:
        value = value.split(':turn:', 1)[0]
    if value.startswith('visual:'):
        parts = value.split(':', 2)
        if len(parts) >= 2 and parts[1]:
            return 'image:' + parts[1]
    if value.startswith('image:'):
        parts = value.split(':', 2)
        if len(parts) >= 2 and parts[1]:
            return 'image:' + parts[1]
    return value or None


def branch_source_families(item):
    record_ids = item.get('retrieval_record_ids') or item.get('record_ids') or []
    if not isinstance(record_ids, list):
        return set()
    return {family for family in (source_family(record_id) for record_id in record_ids)
            if family}


def has_independent_branch_pair(branches):
    """Return whether two branches draw from disjoint source families."""
    families = [branch_source_families(item) for item in branches]
    for left in range(len(families)):
        if not families[left]:
            continue
        for right in range(left + 1, len(families)):
            if families[right] and families[left].isdisjoint(families[right]):
                return True
    return False


def independent_of_existing(candidate, existing):
    families = branch_source_families(candidate)
    return bool(families) and all(
        families.isdisjoint(branch_source_families(item)) for item in existing)


def limit_witness_alternatives(witnesses, per_demand_limit):
    """Keep a bounded, score-first set while preserving source-family diversity."""
    if type(per_demand_limit) is not int or per_demand_limit < 1:
        raise ValueError('positive per-demand witness limit required')
    by_demand = {}
    for item in witnesses or []:
        demand_id = item.get('demand_id')
        if not isinstance(demand_id, str) or not demand_id:
            raise ValueError('witness demand identity required')
        by_demand.setdefault(demand_id, []).append(item)

    selected = []
    for demand_id in sorted(by_demand):
        ranked = sorted(by_demand[demand_id], key=lambda item: (
            -float(item.get('score', 0.)), -len(branch_source_families(item)),
            str(item.get('witness_id', ''))))
        kept, covered = [], set()
        for item in ranked:
            families = branch_source_families(item)
            if families - covered:
                kept.append(item)
                covered.update(families)
            if len(kept) == per_demand_limit:
                break
        if len(kept) < per_demand_limit:
            for item in ranked:
                if item not in kept:
                    kept.append(item)
                if len(kept) == per_demand_limit:
                    break
        selected.extend(kept)
    return selected


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _finite(value, field):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{field} must be finite')
    return value


def _normalize_bindings(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or not isinstance(item, str) or not item:
                raise ValueError('invalid witness binding')
            result[key] = item
        return result
    if not isinstance(value, list):
        raise ValueError('witness bindings must be a mapping or list')
    result = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError('invalid witness binding')
        key, value = item.get('variable'), item.get('value')
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise ValueError('invalid witness binding')
        if key in result and result[key] != value:
            raise ValueError('contradictory witness binding')
        result[key] = value
    return result


def _relation_id(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get('relation_id'), str):
        return value['relation_id']
    raise ValueError('invalid witness source relation')


def _normalize_relations(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError('witness source relations must be a list')
    return sorted({_relation_id(item) for item in value})


def _branch(item, *, verification_state, semantic_sufficiency, candidate_rank=None):
    if not isinstance(item, dict):
        raise ValueError('witness branch must be a mapping')
    demand_id = item.get('demand_id')
    object_ids = item.get('object_ids')
    if not isinstance(demand_id, str) or not demand_id:
        raise ValueError('witness branch demand identity required')
    if (not isinstance(object_ids, list) or not object_ids
            or any(not isinstance(value, str) or not value for value in object_ids)
            or len(set(object_ids)) != len(object_ids)):
        raise ValueError('witness branch object identities required')
    bindings = _normalize_bindings(item.get('bindings'))
    source_relations = _normalize_relations(item.get('source_relations'))
    record_ids = item.get('retrieval_record_ids', item.get('record_ids'))
    if record_ids is None:
        base_record_id = item.get('base_record_id')
        record_ids = [base_record_id] if isinstance(base_record_id, str) and base_record_id else []
    if not isinstance(record_ids, list) or any(not isinstance(value, str) or not value
                                              for value in record_ids):
        raise ValueError('witness branch record identities must be strings')
    identity = item.get('witness_id')
    if not isinstance(identity, str) or not identity:
        payload = {'demand_id': demand_id, 'object_ids': sorted(object_ids),
                   'bindings': bindings, 'source_relations': source_relations,
                   'record_ids': sorted(record_ids)}
        identity = 'w:' + hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return {
        'witness_id': identity,
        'demand_id': demand_id,
        'object_ids': list(dict.fromkeys(object_ids)),
        'record_ids': sorted(set(record_ids)),
        'source_family': (item.get('session_id') or (
            record_ids[0].split(':turn:')[0] if record_ids else None)),
        'bindings': bindings,
        'source_relations': source_relations,
        'score': _finite(item.get('score', 0.), 'witness branch score'),
        'verification_state': verification_state,
        'semantic_sufficiency': semantic_sufficiency,
        'candidate_rank': candidate_rank,
    }


def _dedup_key(branch):
    return (branch['demand_id'], tuple(sorted(branch['object_ids'])),
            _canonical(branch['bindings']), tuple(branch['source_relations']))


def compile_witness_domains(demands, witnesses, candidates=(), *,
                            max_candidate_alternatives=0):
    """Compile one OR domain per demand without upgrading candidate evidence."""
    if not isinstance(demands, list) or not demands:
        raise ValueError('nonempty demand list required')
    demand_ids = [item.get('demand_id') for item in demands]
    if (any(not isinstance(item, str) or not item for item in demand_ids)
            or len(set(demand_ids)) != len(demand_ids)):
        raise ValueError('unique demand identities required')
    if (type(max_candidate_alternatives) is not int
            or max_candidate_alternatives < 0):
        raise ValueError('nonnegative candidate alternative limit required')
    domains = {identity: [] for identity in demand_ids}
    seen = {identity: set() for identity in demand_ids}

    for item in witnesses or []:
        branch = _branch(item, verification_state=SOURCE_LOCATED,
                         semantic_sufficiency=item.get('semantic_sufficiency', UNVERIFIED))
        if branch['demand_id'] not in domains:
            raise ValueError('witness refers to an unknown demand')
        key = _dedup_key(branch)
        if key not in seen[branch['demand_id']]:
            domains[branch['demand_id']].append(branch)
            seen[branch['demand_id']].add(key)

    by_demand = {identity: [] for identity in demand_ids}
    for item in candidates or []:
        if not isinstance(item, dict) or item.get('demand_id') not in by_demand:
            raise ValueError('candidate refers to an unknown demand')
        by_demand[item['demand_id']].append(item)
    for identity in demand_ids:
        ranked = sorted(by_demand[identity], key=lambda item: (
            -_finite(item.get('score', 0.), 'candidate score'),
            str(item.get('base_record_id', item.get('witness_id', '')))))
        added = 0
        for item in ranked:
            if added >= max_candidate_alternatives:
                break
            candidate = _branch(item, verification_state=CANDIDATE_ONLY,
                                semantic_sufficiency=UNVERIFIED,
                                candidate_rank=added)
            key = _dedup_key(candidate)
            if key in seen[identity]:
                continue
            domains[identity].append(candidate)
            seen[identity].add(key)
            added += 1
    for values in domains.values():
        values.sort(key=lambda item: (
            0 if item['verification_state'] == SOURCE_LOCATED else 1,
            -item['score'], item['witness_id']))
    return domains


def _compatible(candidate, chosen, relations):
    for previous in chosen.values():
        overlap = set(candidate['bindings']) & set(previous['bindings'])
        different = {relation.get('variable') for relation in relations
                     if relation.get('kind') == 'different_binding'
                     and {relation.get('left'), relation.get('right')}
                     == {candidate['demand_id'], previous['demand_id']}}
        if any(candidate['bindings'][key] != previous['bindings'][key]
               for key in overlap - different):
            return False
    joined = {**chosen, candidate['demand_id']: candidate}
    for relation in relations:
        left, right = joined.get(relation.get('left')), joined.get(relation.get('right'))
        if left is None or right is None:
            continue
        if relation.get('kind') == 'source_relation':
            identity = relation.get('relation_id')
            if identity not in set(left['source_relations']) | set(right['source_relations']):
                return False
            continue
        variable = relation.get('variable')
        if variable not in left['bindings'] or variable not in right['bindings']:
            return False
        equal = left['bindings'][variable] == right['bindings'][variable]
        if equal != (relation.get('kind') == 'same_binding'):
            return False
    return True


def select_min_cost_branch_cover(demands, domains, relations, assess_object_ids, *,
                                 allowed_verification=None, time_limit=2.,
                                 max_states=100000):
    """Exact minimum-cost OR choice over retained witness domains."""
    if (not isinstance(demands, list) or not demands
            or not math.isfinite(float(time_limit)) or time_limit <= 0
            or type(max_states) is not int or max_states < 1):
        raise ValueError('nonempty demands and positive search limits required')
    if allowed_verification is not None:
        allowed_verification = set(allowed_verification)
    order = [demand['demand_id'] for demand in demands]
    filtered = {}
    for identity in order:
        values = list(domains.get(identity, []))
        if allowed_verification is not None:
            values = [item for item in values
                      if item['verification_state'] in allowed_verification]
        filtered[identity] = sorted(
            values, key=lambda item: (item['score'], item['witness_id']), reverse=True)
    missing = [identity for identity in order if not filtered[identity]]
    if missing:
        return {'feasible': False, 'optimal': True, 'missing_demands': missing,
                'selected_branches': [], 'accepted_object_ids': [], 'cost': None,
                'states': 0, 'timed_out': False, 'allowed_verification':
                None if allowed_verification is None else sorted(allowed_verification)}
    started = time.perf_counter()
    order.sort(key=lambda identity: (len(filtered[identity]), identity))
    best, states = None, 0
    pending = [(0, {}, frozenset(), math.inf)]
    timed_out = False
    while pending:
        if states >= max_states or time.perf_counter() - started > time_limit:
            timed_out = True
            break
        depth, chosen, object_ids, minimum_score = pending.pop()
        states += 1
        if depth == len(order):
            assessment = assess_object_ids(object_ids)
            cost = _finite(assessment.get('cost'), 'materialization cost')
            if cost < 0 or not assessment.get('feasible'):
                continue
            candidate = {'cost': cost, 'minimum_score': minimum_score,
                         'objects': sorted(object_ids),
                         'branches': [chosen[identity] for identity in order]}
            objective = (cost, -minimum_score,
                         tuple(item['witness_id'] for item in candidate['branches']))
            if best is None or objective < best['objective']:
                best = {**candidate, 'objective': objective}
            continue
        identity = order[depth]
        for branch in reversed(filtered[identity]):
            if not _compatible(branch, chosen, relations):
                continue
            pending.append((depth + 1, {**chosen, identity: branch},
                            object_ids | frozenset(branch['object_ids']),
                            min(minimum_score, branch['score'])))
    return {
        'feasible': best is not None,
        'optimal': not timed_out,
        'missing_demands': [],
        'selected_branches': [] if best is None else best['branches'],
        'accepted_object_ids': [] if best is None else best['objects'],
        'cost': None if best is None else best['cost'],
        'minimum_score': None if best is None else best['minimum_score'],
        'states': states,
        'timed_out': timed_out,
        'allowed_verification': (None if allowed_verification is None
                                 else sorted(allowed_verification)),
        'solver': 'exact retained-domain minimum unique-object cost cover',
        'boundary': 'optimal only over retained branches; candidate-only branches are not evidence sufficiency certificates',
    }


def _cost(object_ids, object_costs):
    if object_costs is None:
        return {'feasible': True, 'cost': float(len(set(object_ids)))}
    missing = [identity for identity in object_ids if identity not in object_costs]
    if missing:
        raise ValueError('missing object costs: ' + ','.join(sorted(set(missing))[:3]))
    return {'feasible': True,
            'cost': float(sum(float(object_costs[identity]) for identity in set(object_ids)))}


def verify_hypergraph(demands, domains, relations):
    demand_ids = [item['demand_id'] for item in demands]
    branches = [branch for identity in demand_ids for branch in domains.get(identity, [])]
    identities = [branch['witness_id'] for branch in branches]
    if len(set(identities)) != len(identities):
        raise ValueError('witness branch identities must be unique within a hypergraph')
    return {
        'schema': SCHEMA,
        'and_edges': [{'head': 'query', 'tail': demand_ids,
                       'kind': 'all_required_demands'}] + [
            {'head': branch['witness_id'], 'tail': branch['object_ids'],
             'kind': 'all_sources_in_witness'} for branch in branches],
        'or_edges': [{'head': identity,
                      'tail': [branch['witness_id'] for branch in domains.get(identity, [])],
                      'kind': 'alternative_witnesses'} for identity in demand_ids],
        'relations': relations,
        'branches': branches,
        'boundary': ('branches are separated by verification state; source_located means anchors and bindings '
                     'were machine-checked, not that the source semantically entails the demand'),
    }


def audit_multi_witness(demands, witnesses, candidates=(), *, relations=None,
                        selected_object_ids=(), object_costs=None,
                        max_candidate_alternatives=5, time_limit=2.):
    relations = list(relations or [])
    domains = compile_witness_domains(
        demands, witnesses, candidates,
        max_candidate_alternatives=max_candidate_alternatives)
    per_demand = []
    for demand in demands:
        branches = domains[demand['demand_id']]
        verified = [item for item in branches
                    if item['verification_state'] == SOURCE_LOCATED]
        semantic = [item for item in branches
                    if item['verification_state'] == SEMANTIC_VERIFIED]
        candidate_only = [item for item in branches
                          if item['verification_state'] == CANDIDATE_ONLY]
        existing_records = {record for item in verified for record in item['record_ids']}
        new_source_candidates = [item for item in candidate_only
                                 if not set(item['record_ids']) <= existing_records]
        candidate_families = {item['source_family'] for item in new_source_candidates
                              if item['source_family']}
        verified_families = [branch_source_families(item) for item in verified]
        independent_source_located_or = has_independent_branch_pair(verified)
        per_demand.append({
            'demand_id': demand['demand_id'],
            'source_located_branch_count': len(verified),
            'source_located_family_count': len({family for families in verified_families
                                                 for family in families}),
            'semantic_verified_branch_count': len(semantic),
            'candidate_only_branch_count': len(candidate_only),
            'new_source_candidate_count': len(new_source_candidates),
            'new_source_family_count': len(candidate_families),
            'or_width': len(branches),
            'source_located_or_available': independent_source_located_or,
            'semantic_or_available': has_independent_branch_pair(semantic),
            'branch_ids': [item['witness_id'] for item in branches],
        })
    assessed = _cost(selected_object_ids, object_costs)
    provisional = select_min_cost_branch_cover(
        demands, domains, relations,
        lambda object_ids: _cost(object_ids, object_costs),
        time_limit=time_limit)
    source_located = select_min_cost_branch_cover(
        demands, domains, relations,
        lambda object_ids: _cost(object_ids, object_costs),
        allowed_verification={SOURCE_LOCATED, SEMANTIC_VERIFIED},
        time_limit=time_limit)
    semantic = select_min_cost_branch_cover(
        demands, domains, relations,
        lambda object_ids: _cost(object_ids, object_costs),
        allowed_verification={SEMANTIC_VERIFIED},
        time_limit=time_limit)
    candidate_lower_bound = select_min_cost_branch_cover(
        demands, domains, relations,
        lambda object_ids: _cost(object_ids, object_costs),
        allowed_verification={CANDIDATE_ONLY},
        time_limit=time_limit)
    selected_cost = assessed['cost']
    candidate_cost = candidate_lower_bound.get('cost')
    return {
        'schema': SCHEMA,
        'domains': per_demand,
        'hypergraph': verify_hypergraph(demands, domains, relations),
        'coverage': {
            'demand_count': len(demands),
            'demands_with_source_located_witness': sum(
                item['source_located_branch_count'] > 0 for item in per_demand),
            'demands_with_independent_source_located_or': sum(
                item['source_located_or_available'] for item in per_demand),
            'demands_with_semantic_or': sum(
                item['semantic_or_available'] for item in per_demand),
            'demands_with_candidate_only_or': sum(
                item['candidate_only_branch_count'] >= 2 for item in per_demand),
            'demands_with_candidate_only_or_across_source_families': sum(
                item['new_source_family_count'] >= 2 for item in per_demand),
        },
        'costs': {
            'selected_object_cost': selected_cost,
            'provisional_all_branch_min_cost': provisional.get('cost'),
            'source_located_min_cost': source_located.get('cost'),
            'semantic_min_cost': semantic.get('cost'),
            'candidate_only_lower_bound': candidate_cost,
            'candidate_only_reduction_fraction': (
                None if candidate_cost is None or selected_cost <= 0
                else max(0., 1. - candidate_cost / selected_cost)),
        },
        'selections': {
            'provisional': provisional,
            'source_located': source_located,
            'semantic': semantic,
            'candidate_lower_bound': candidate_lower_bound,
        },
        'boundary': ('candidate_only lower bound is a search-space diagnostic, not a quality claim; '
                     'source-located OR requires disjoint source families; source location is not semantic entailment'),
    }
