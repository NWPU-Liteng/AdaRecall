"""Provenance-constrained proof search with shared physical materialization."""
from __future__ import annotations

import itertools
import math
import time


PARSER_PROMPT = (
    '你是查询语义解析器，只读取问题，不读取历史。输出严格JSON，不回答问题。'
    '格式 {"demands":[{"demand_id":"d0","text":"待回答的事实角色及问题给出的限定条件",'
    '"query_quote":"问题中的逐字片段","variables":[]}],"relations":[]}。'
    'demands中每一项都是必须共同满足的逻辑义务，与历史是否有证据完全无关，绝不能预判来源是否可绑定。'
    '比较、先后变化或分别询问多个事实时，按问题语义拆成必须共同满足的需求。'
    '最多拆成4项；只拆独立的答案事实，不把时间、人物、地点、范围或语气等限定条件各自拆成需求。'
    '每项query_quote应是互不重复且能代表该事实要求的最短问题片段；不得复制或反复拆同一句。'
    '一个事实可由同一来源或多个互补来源支持，拆分不规定来源数；不凭空增加需求。'
    '开放式回顾/建议且没有可枚举的必需事实时只输出一个整体原问题，不输出可选事实清单。'
    'variables仅列跨需求连接所需的实体/事件键，不是待求答案、关键词或问题中的所有名词；无连接则为[]。'
    '仅当问题明确要求查看原图、截图、照片或原始文字记录时，才可在该需求中加入'
    'evidence_scope：{"kind":"image_locator"}或{"kind":"source_text"}；不明确时必须省略，不能猜测来源。'
    'relations格式 {"left":"d0","right":"d1","kind":"same_binding",'
    '"variable":"project","query_quote":"逐字关系片段"}。'
    'same_binding/different_binding仅表达问题要求的同一/不同实体，两端variables必须声明相同键。'
    '确需历史明确记载的因果或更新关系才使用source_relation并给出relation_id；'
    '不能仅因为比较两个事实就要求历史明确写出比较关系。无显式关系则relations为[]。')


QUERY_GRAPH_SCHEMA = {
    'type': 'object', 'required': ['demands', 'relations'], 'additionalProperties': False,
    'properties': {
        'demands': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {
            'type': 'object', 'required': ['demand_id', 'text', 'query_quote', 'variables'],
            'additionalProperties': False, 'properties': {
                'demand_id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
                'text': {'type': 'string', 'minLength': 1, 'maxLength': 384},
                'query_quote': {'type': 'string', 'minLength': 1, 'maxLength': 192},
                'variables': {'type': 'array', 'maxItems': 8, 'uniqueItems': True,
                              'items': {'type': 'string', 'minLength': 1, 'maxLength': 32}},
                'evidence_scope': {'type': 'object', 'additionalProperties': False,
                                   'properties': {'kind': {'type': 'string',
                                                           'enum': ['source_text', 'image_locator']}}}}}},
        'relations': {'type': 'array', 'maxItems': 8, 'items': {
            'type': 'object', 'required': ['left', 'right', 'kind', 'query_quote'],
            'additionalProperties': False, 'properties': {
                'left': {'type': 'string', 'minLength': 1, 'maxLength': 12},
                'right': {'type': 'string', 'minLength': 1, 'maxLength': 12},
                'kind': {'type': 'string', 'enum': ['same_binding', 'different_binding', 'source_relation']},
                'variable': {'type': 'string', 'minLength': 1, 'maxLength': 32},
                'relation_id': {'type': 'string', 'minLength': 1, 'maxLength': 32},
                'query_quote': {'type': 'string', 'minLength': 1, 'maxLength': 192}}}}}}


def _score(value):
    if not math.isfinite(float(value)):
        raise ValueError('finite support scores required')
    return float(value)


def parse_query_graph(parsed, query, *, allow_shared_quotes=False):
    fallback = [{'demand_id': 'soft_query', 'text': query, 'required': True, 'variables': []}]
    if not isinstance(parsed, dict) or not isinstance(parsed.get('demands'), list):
        return fallback, [], 'invalid_parse'
    demands, identities = [], set()
    canonical_by_id = {}
    by_text = {}
    for item in parsed['demands']:
        if not isinstance(item, dict):
            return fallback, [], 'invalid_demand'
        identity, text, variables = item.get('demand_id'), item.get('text'), item.get('variables', [])
        evidence_scope = item.get('evidence_scope', {})
        if (not isinstance(identity, str) or not identity or identity in identities
                or not isinstance(text, str) or not text.strip()
                or not isinstance(variables, list)
                or any(not isinstance(variable, str) or not variable for variable in variables)
                or not isinstance(evidence_scope, dict)
                or set(evidence_scope) - {'kind'}
                or (evidence_scope.get('kind') not in {None, 'source_text', 'image_locator'})):
            return fallback, [], 'invalid_demand'
        quote = item.get('query_quote')
        if not isinstance(quote, str) or not quote.strip() or quote not in query:
            return fallback, [], 'unlocated_demand'
        identities.add(identity)
        description = text.strip()
        if description in by_text:
            canonical = by_text[description]
            canonical_by_id[identity] = canonical
            target = next(demand for demand in demands
                          if demand['demand_id'] == canonical)
            target['variables'] = sorted(set(target['variables']) | set(variables))
            if evidence_scope and 'evidence_scope' not in target:
                target['evidence_scope'] = {'kind': evidence_scope['kind']}
            continue
        demand = {'demand_id': identity, 'text': text, 'required': True,
                  'query_quote': quote, 'variables': sorted(set(variables))}
        if evidence_scope:
            demand['evidence_scope'] = {'kind': evidence_scope['kind']}
        demands.append(demand)
        by_text[description] = identity
        canonical_by_id[identity] = identity
    if not demands:
        return fallback, [], 'empty_demands'
    if len(demands) > 4:
        return fallback, [], 'too_many_demands'
    quotes = [demand['query_quote'] for demand in demands]
    if not allow_shared_quotes and len(set(quotes)) != len(quotes):
        return fallback, [], 'duplicate_demand_quote'
    relations = parsed.get('relations', [])
    if not isinstance(relations, list):
        return fallback, [], 'invalid_relations'
    canonical_relations = []
    for relation in relations:
        if (not isinstance(relation, dict) or not isinstance(relation.get('left'), str)
                or not isinstance(relation.get('right'), str) or relation.get('left') not in identities
                or relation.get('right') not in identities
                or relation.get('kind') not in {'same_binding', 'different_binding', 'source_relation'}
                or not isinstance(relation.get('query_quote'), str) or not relation['query_quote']
                or relation['query_quote'] not in query):
            return fallback, [], 'unlocated_relation'
        left = canonical_by_id.get(relation['left'], relation['left'])
        right = canonical_by_id.get(relation['right'], relation['right'])
        if left == right:
            continue
        relation = {**relation, 'left': left, 'right': right}
        if relation['kind'] == 'source_relation':
            if not isinstance(relation.get('relation_id'), str) or not relation['relation_id']:
                return fallback, [], 'invalid_relation_identity'
        elif (not isinstance(relation.get('variable'), str) or not relation['variable']
              or any(relation['variable'] not in demand['variables'] for demand in demands
                     if demand['demand_id'] in {relation['left'], relation['right']})):
            return fallback, [], 'missing_binding_variable'
        canonical_relations.append(relation)
    relations = canonical_relations
    source_relation_ids = [relation['relation_id'] for relation in relations if relation['kind'] == 'source_relation']
    if len(source_relation_ids) != len(set(source_relation_ids)):
        return fallback, [], 'duplicate_relation_identity'
    return demands, relations, 'ok'


def select_retrieval_package(records, evaluate):
    candidates = {}
    for record in records:
        identity = record['record_id']
        previous = candidates.get(identity)
        if previous is None or _score(record['score']) > _score(previous['score']):
            candidates[identity] = record
    ordered = sorted(candidates.values(), key=lambda item: (-_score(item['score']), item['record_id']))
    selected, object_ids = [], frozenset()
    assessment = evaluate(object_ids)
    for record in ordered:
        proposed = object_ids | frozenset(record['object_ids'])
        if proposed == object_ids:
            continue
        candidate_assessment = evaluate(proposed)
        if candidate_assessment.get('feasible'):
            selected.append(record)
            object_ids = proposed
            assessment = candidate_assessment
    return {'feasible': bool(assessment.get('feasible')), 'selected_records': selected,
            'selected_record_ids': [record['record_id'] for record in selected],
            'accepted_object_ids': sorted(object_ids), 'assessment': assessment,
            'selection_mode': 'score_ordered_budget_feasible_retrieval_package',
            'semantic_proof': False, 'candidate_records': len(ordered)}


def parse_requirements(parsed, query):
    if not isinstance(parsed, dict):
        return [{'demand_id': 'soft_query', 'text': query, 'hard': False, 'relations': []}], 'invalid'
    demands = parsed.get('demands')
    if not isinstance(demands, list) or not demands:
        return [{'demand_id': 'soft_query', 'text': query, 'hard': False, 'relations': []}], 'missing_demands'
    result = []
    for index, demand in enumerate(demands):
        if not isinstance(demand, dict) or not isinstance(demand.get('text'), str) or not demand['text'].strip():
            continue
        result.append({'demand_id': str(demand.get('demand_id', f'demand_{index}')),
                       'text': demand['text'].strip(), 'hard': bool(demand.get('hard', True)),
                       'relations': demand.get('relations', []) if isinstance(demand.get('relations', []), list) else []})
    if not result:
        return [{'demand_id': 'soft_query', 'text': query, 'hard': False, 'relations': []}], 'empty_demands'
    return result, 'ok'


def _frontier_records(records, scores, demand_count, limit):
    if len(records) != len(scores) or limit < 1:
        raise ValueError('aligned proof records and positive limit required')
    ranked = sorted(zip(records, scores), key=lambda pair: (-_score(pair[1]), pair[0]['record_id']))
    result = []
    for record, score in ranked[:limit]:
        demand_index = record.get('demand_index')
        covered = ([int(demand_index)] if demand_index is not None else list(range(demand_count)))
        result.append({'record': record, 'score': float(score), 'covered_demands': covered})
    return result


def _valid_subset(subset, objects, text_budget, max_images, input_budget, fixed_prompt_tokens,
                  tokenizer, image_token_costs):
    object_ids = sorted({object_id for item in subset for object_id in item['record']['object_ids']})
    selected = [objects[object_id] for object_id in object_ids]
    text_objects = sorted((item for item in selected if item['kind'] == 'text'),
                          key=lambda item: item['ordinal'])
    image_objects = [item for item in selected if item['kind'] == 'image']
    text = '\n'.join(item['text'] for item in text_objects)
    text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    visual_tokens = sum(image_token_costs[item['source_sha256']] for item in image_objects)
    return (text_tokens <= text_budget and len(image_objects) <= max_images
            and fixed_prompt_tokens + text_tokens + visual_tokens <= input_budget,
            object_ids, text, text_tokens, visual_tokens)


def plan_proof(records, scores, demands, objects, tokenizer, image_token_costs, *,
               text_budget, max_images, input_budget, fixed_prompt_tokens,
               max_candidates=16, time_limit=2.0):
    started = time.perf_counter()
    if not records or not demands or max_candidates < 1 or time_limit <= 0:
        raise ValueError('nonempty proof inputs and positive planner limits required')
    candidates = _frontier_records(records, scores, len(demands), max_candidates)
    hard = {index for index, demand in enumerate(demands) if demand.get('hard')}
    best = None
    feasible_count = 0
    examined = 0
    upper_bound = None
    for count in range(1, min(len(candidates), 8) + 1):
        for subset in itertools.combinations(candidates, count):
            examined += 1
            if time.perf_counter() - started > time_limit:
                break
            covered = set().union(*(item['covered_demands'] for item in subset))
            if hard and not hard <= covered:
                continue
            feasible, object_ids, text, text_tokens, visual_tokens = _valid_subset(
                subset, objects, text_budget, max_images, input_budget, fixed_prompt_tokens,
                tokenizer, image_token_costs)
            if not feasible:
                continue
            feasible_count += 1
            support = min(float(item['score']) for item in subset)
            cost = text_tokens + visual_tokens
            objective = (support, -cost, -len(object_ids),
                         tuple(sorted(item['record']['record_id'] for item in subset)))
            if best is None or objective > best['objective']:
                best = {'subset': subset, 'object_ids': object_ids, 'text': text,
                        'text_tokens': text_tokens, 'visual_tokens': visual_tokens,
                        'support': support, 'objective': objective}
        if time.perf_counter() - started > time_limit:
            break
    if best is None:
        # A single soft candidate remains useful even if the strict proof graph is infeasible.
        for candidate in candidates:
            feasible, object_ids, text, text_tokens, visual_tokens = _valid_subset(
                (candidate,), objects, text_budget, max_images, input_budget, fixed_prompt_tokens,
                tokenizer, image_token_costs)
            if feasible:
                best = {'subset': (candidate,), 'object_ids': object_ids, 'text': text,
                        'text_tokens': text_tokens, 'visual_tokens': visual_tokens,
                        'support': float(candidate['score']),
                        'objective': (float(candidate['score']), -text_tokens - visual_tokens,
                                      -len(object_ids), (candidate['record']['record_id'],))}
                break
    if best is None:
        raise RuntimeError('no feasible source proof under the registered Reader budget')
    elapsed = time.perf_counter() - started
    selected = [item['record'] for item in best['subset']]
    return {'selected_record_ids': [item['record_id'] for item in selected],
            'selected_records': selected, 'accepted_object_ids': best['object_ids'],
            'selected_hyperedges': [{'record_id': item['record_id'],
                'object_ids': item['object_ids'], 'support_score': float(next(
                    candidate['score'] for candidate in best['subset']
                    if candidate['record']['record_id'] == item['record_id']))}
                for item in selected],
            'demands': demands, 'hard_demand_ids': sorted(demands[index]['demand_id'] for index in hard),
            'support_score_min': best['support'], 'text': best['text'],
            'text_tokens_estimate': best['text_tokens'], 'visual_tokens_estimate': best['visual_tokens'],
            'planner': 'bounded proof witness enumeration with unique object costs',
            'optimality': 'exact_over_examined_subsets_only', 'candidate_count': len(candidates),
            'examined_subsets': examined, 'feasible_subsets': feasible_count,
            'planner_seconds': elapsed, 'timeout': elapsed > time_limit,
            'boundary': 'support scores and parser relations are proxies; original raw objects are the only Reader evidence; no QA labels or answer-dependent routing'}
