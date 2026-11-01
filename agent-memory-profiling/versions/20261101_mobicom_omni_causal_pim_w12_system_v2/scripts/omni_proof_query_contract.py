"""Separate source observations from query operations without inventing evidence."""
from __future__ import annotations

from copy import deepcopy
import json

from jsonschema import Draft202012Validator

from omni_proof_planner import QUERY_GRAPH_SCHEMA, parse_query_graph


QUERY_CONTRACT_PROMPT = (
    '你是查询计划编译器，只看问题，不看历史，不回答问题。输出严格JSON。'
    '根字段为answer_mode、source_facts、relations、operations。'
    'answer_mode为structured或contextual。'
    'source_facts只列需要从原文/原图读取的基础观察，格式为'
    '{"demand_id":"d0","text":"观察角色及限定条件","query_quote":"问题逐字片段","variables":[]}。'
    '只有问题明确要求原图、截图或照片时，才可附加evidence_scope:{"kind":"image_locator"}；'
    '只有明确要求原始文字记录时，才可附加evidence_scope:{"kind":"source_text"}；不明确时省略，不能猜测。'
    '最多4项，不预测历史是否有证据，也不规定来源条数。'
    '完整原问题会始终保留为检索上下文；text仍应保留对应事件、人物、时段等限定。'
    '时间差、比较结果、排序结果属于operations，绝不是第三条必须在历史中写出的事实。'
    'contextual不是默认值，不能作为不确定时的兜底。询问历史中的具体事实、人物、选择、偏好、列表或先后事项时用structured；'
    '例如“最在意的两件事是什么”“最后更想保住哪两类支出”“平时用什么平台”都是structured的一项观察，'
    '即使答案是列表或偏好也不自动用contextual。只有开放建议、需要跨多段归纳而无法列出必要事实时才用contextual。'
    '“在事件A之后/聊完A后/那次A时，结果是什么”中的A通常只是定位答案的上下文；'
    '将A作为答案事实的限定条件写入目标source_fact，不另设一个必须证明的需求，除非问题也明确询问A本身。'
    '询问例外行为背后的通常平台或习惯时，只列被问的通常平台或习惯；例外行为仍保留为定位上下文，不另列为必需观察。'
    '凡问“哪次更久/更近/更多/更早”“A比B有什么变化”“先后相隔多久”，必须拆出计算所需的基础观察并建运算，不能归为contextual。'
    '但“某事件后最后往哪个方向调整/作了什么选择”不是比较运算；仅询问结果时只列结果观察。'
    '问两事件相隔多久时，读取两个事件各自的时间，建一个difference运算；'
    '问两个阶段的状态有什么变化时，分别读取两个阶段的状态，建一个comparison运算；'
    '若比较两个事件到共同参照（如晚饭后）的距离，读取每个事件及其对应参照时间，'
    '分别建立difference，再建立以这两个difference为输入的comparison或selection。'
    '操作格式为{"operation_id":"op0","kind":"difference","inputs":["d0","d1"],'
    '"query_quote":"问题中连续、逐字的运算片段"}。kind可为difference/comparison/selection/aggregation。'
    '运算inputs可引用source_facts的demand_id或更早定义的operation_id；按依赖顺序输出，禁止前向引用、循环和答案值。'
    '每个运算的query_quote必须是问题中连续、逐字的原文，不得用省略号拼接片段。'
    '比较的结论不可预先假定：不能把“两次哪次更久”拆成“两次都更久”。'
    '人物、地点、背景确认和语气不单独变成必需事实。只问当前选择时不额外要求旧选择。'
    '一个列表可作为一项观察；不为凑AND数目拆开。'
    '开放建议、偏好归纳或缺证据判断用contextual，source_facts只保留一个完整原问题，'
    'relations和operations为空；这不代表已证明任何正向事实或全局不存在。'
    'contextual时text和query_quote都必须逐字等于完整原问题；结构化事实检索用structured。'
    '每个query_quote必须是问题中最短的连续逐字子串，不得把不连续词语拼接，也不得补写问题省略的角色；不反复列同一答案目标。'
    '例如“阶段乙的状态比阶段甲”省略了后半句的“状态”，两个quote可直接用“阶段乙”和“阶段甲”，不能补成“阶段甲的状态”。'
    '同一问题片段可对应不同角色的观察，例如两个事件各自对应的晚饭时间；必须分别列出事件A对应的参照和事件B对应的参照两个source_fact，'
    '它们可使用同一最短quote，但text必须明确事件限定，不能混成一个全局参照时间。'
    'variables仅为需要跨事实连接的实体/事件键，不是答案值、待比较状态或时间值；'
    '比较不强制两个状态或时间相等，也不要求历史写出比较结论。'
    'relations格式为{"left":"d0","right":"d1","kind":"same_binding",'
    '"variable":"project","query_quote":"问题逐字关系片段"}；'
    'same_binding/different_binding两端必须在variables中声明同名实体/事件键。'
    'source_relation仅用于确实需要原历史记载的因果/更新关系，并给出relation_id。'
    '没有这种关系时relations=[]；不发明连接键或来源。'
    '抽象示例：问“与某人谈话后，学习安排最后往哪个方向调整”，只列一个含谈话上下文的结果source_fact，不建comparison。'
    '问“活动甲到活动乙相隔多久”，source_facts分别记录甲时间、乙时间，'
    'operations为difference([d0,d1])。问“阶段乙的状态比阶段甲有什么变化”，'
    'source_facts分别记录甲状态与乙状态，operations为comparison([d0,d1])。'
    '问“两次活动分别到各自参照事件的间隔哪次更大”，必须有四个source_facts：事件A、事件A参照、事件B、事件B参照；'
    'operations按顺序为difference([d0,d1])、difference([d2,d3])、comparison([op0,op1])。')


QUERY_CONTRACT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['answer_mode', 'source_facts', 'relations', 'operations'],
    'properties': {
        'answer_mode': {'type': 'string', 'enum': ['structured', 'contextual']},
        'source_facts': deepcopy(QUERY_GRAPH_SCHEMA['properties']['demands']),
        'relations': deepcopy(QUERY_GRAPH_SCHEMA['properties']['relations']),
        'operations': {'type': 'array', 'maxItems': 4, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['operation_id', 'kind', 'inputs', 'query_quote'],
        'properties': {
                'operation_id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
                'kind': {'type': 'string', 'enum': ['difference', 'comparison', 'selection', 'aggregation']},
                'inputs': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'uniqueItems': True,
                           'items': {'type': 'string', 'minLength': 1, 'maxLength': 12}},
                'query_quote': {'type': 'string', 'minLength': 1, 'maxLength': 192}}}}}}


VISUAL_CUES = ('原图', '图片', '照片', '截图', '图像', '画面', '图中', '图里', '看图', '镜头')


def promote_visual_scope(query, facts):
    """Make an explicit visual request a real retrieval obligation.

    The parser is asked to emit evidence_scope, but the measured pipeline often
    drops it.  This compiler normalization only acts when the query itself
    names a visual artifact; it does not infer image content or an answer.
    """
    if not isinstance(query, str) or not any(cue in query for cue in VISUAL_CUES):
        return facts
    if any(fact.get('evidence_scope', {}).get('kind') == 'image_locator'
           for fact in facts):
        return facts
    if not facts:
        return facts
    target = next((fact for fact in facts
                   if any(cue in fact.get('text', '') for cue in VISUAL_CUES)),
                  facts[0])
    target['evidence_scope'] = {'kind': 'image_locator'}
    return facts


def parse_query_contract(parsed, query):
    fallback = {'status': 'invalid_schema', 'answer_mode': 'contextual',
                'source_facts': [{'demand_id': 'soft_query', 'text': query,
                                  'required': True, 'variables': []}],
                'relations': [], 'operations': [], 'semantic_entailment_proven': False}
    if not Draft202012Validator(QUERY_CONTRACT_SCHEMA).is_valid(parsed):
        return fallback
    facts, relations, status = parse_query_graph(
        {'demands': parsed['source_facts'], 'relations': parsed['relations']}, query,
        allow_shared_quotes=True)
    if status != 'ok':
        return {**fallback, 'status': status}
    if parsed['answer_mode'] == 'structured':
        facts = promote_visual_scope(query, facts)
    canonical_by_source_id = {
        fact['demand_id']: fact['demand_id'] for fact in facts
    }
    text_to_canonical = {fact['text'].strip(): fact['demand_id'] for fact in facts}
    for source_fact in parsed['source_facts']:
        canonical = text_to_canonical.get(source_fact.get('text', '').strip())
        if canonical is not None:
            canonical_by_source_id[source_fact['demand_id']] = canonical
    fact_ids = {fact['demand_id'] for fact in facts}
    operation_ids = set()
    operations = []
    for operation in parsed['operations']:
        identity = operation['operation_id']
        inputs = [canonical_by_source_id.get(value, value)
                  for value in operation['inputs']]
        inputs = list(dict.fromkeys(inputs))
        operation = {**operation, 'inputs': inputs}
        if identity in fact_ids or identity in operation_ids:
            return {**fallback, 'status': 'duplicate_operation_identity'}
        if not set(operation['inputs']) <= fact_ids | operation_ids:
            return {**fallback, 'status': 'unknown_operation_input'}
        if operation['kind'] in ['difference', 'comparison', 'selection'] and len(operation['inputs']) < 2:
            return {**fallback, 'status': 'missing_operation_operand'}
        if operation['kind'] == 'difference' and len(operation['inputs']) != 2:
            return {**fallback, 'status': 'invalid_difference_arity'}
        if operation['query_quote'] not in query:
            return {**fallback, 'status': 'unlocated_operation'}
        operation_ids.add(identity)
        operations.append(operation)
    if any(set(fact['variables']) & (fact_ids | operation_ids) for fact in facts):
        return {**fallback, 'status': 'observation_or_operation_id_used_as_entity_key'}
    if parsed['answer_mode'] == 'contextual' and (
            len(facts) != 1 or facts[0]['text'] != query or relations or parsed['operations']):
        return {**fallback, 'status': 'contextual_query_must_remain_whole'}
    return {'status': 'ok', 'answer_mode': parsed['answer_mode'], 'source_facts': facts,
            'relations': relations, 'operations': operations,
            'semantic_entailment_proven': False}


def retrieval_views(query, contract):
    return [json.dumps({'original_question': query, 'source_observation': fact['text']},
                       ensure_ascii=False, separators=(',', ':'))
            for fact in contract['source_facts']]


def render_operation_plan(contract):
    if contract.get('status') != 'ok' or not contract.get('operations'):
        return ''
    facts = {fact['demand_id']: fact['text'] for fact in contract['source_facts']}
    operations = {}
    lines = ['查询运算计划（描述问题要求，不是历史证据）：', '基础观察：']
    lines.extend(f"- {identity}: {text}" for identity, text in facts.items())
    lines.append('派生运算：')
    for operation in contract['operations']:
        inputs = [facts.get(identity, operations.get(identity, identity))
                  for identity in operation['inputs']]
        detail = f"{operation['kind']}({'; '.join(inputs)})"
        operations[operation['operation_id']] = detail
        lines.append(f"- {operation['operation_id']}: {detail}；问题片段：{operation['query_quote']}")
    return '\n'.join(lines)


def reader_context(query, timestamp, text, operation_plan=''):
    return ('原始对话记忆（带来源和可见 session 时间范围）：\n' + text
            + '\n\n查询时间协议：完整历史末端（' + timestamp + '）'
            + ('\n\n' + operation_plan if operation_plan else '')
            + '\n问题：' + query + '\n请依据文字和随后附上的原始图片作答。')
