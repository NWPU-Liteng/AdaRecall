"""Validate model-proposed source spans, image regions and explicit bindings."""
from __future__ import annotations

import json

from omni_proof_runtime import checksum


WITNESS_PROMPT = (
    '你是证据连接构造器，不回答问题、不验证答案。仅用给定来源索引构造需求见证。'
    '若输入含query_operations，它只描述需要Reader在证据齐备后执行的派生运算；不要为运算创建需求或见证，'
    '只为source_facts中的基础观察定位来源。'
    '严格输出紧凑JSON {"witnesses":[{"demand_id":"d0","record_ids":["r0"],'
    '"anchors":[{"record_id":"r0","text_index":0}],"bindings":[],"source_relations":[]}]}。'
    'text_index引用该record的text_sources数组中的完整原文atom，程序会保留其原文span和哈希；'
    '不要复制原文、不要输出object_id。对于图片只用 {record_id:"r0",region_index:0} 引用'
    '该record中image.regions的区域；描述只是locator，Reader仍需原图。'
    '每个需求列出足以支持它的最少互补来源，互补来源在同一见证内联合引用；'
    '即使只有一个整体需求，也允许一个见证联合多个互补来源。'
    '替代来源用不同见证，不要把所有相似候选一起列入。'
    '只绑定需求variables声明的连接键：{"variable":"查询变量","value":"原文逐字实体名",'
    '"record_id":"r0","text_index":0}。value必须在引用atom原文中，禁止推断。'
    'source_relations仅可从原文引用：{"relation_id":"查询关系ID","record_id":"r0",'
    '"text_index":0,"quote":"极短的逐字关系片段"}。无关系时置空数组。'
    '候选中的邻近原文只提供来源上下文，不自动纳入证明；需要时用相应text_index明确引用。'
    '无法定位则不输出见证。严禁解释、长引用、回答问题和新增字段。')


WITNESS_SCHEMA = {
    'type': 'object', 'required': ['witnesses'], 'additionalProperties': False,
    'properties': {'witnesses': {'type': 'array', 'maxItems': 24, 'items': {
        'type': 'object', 'required': ['demand_id', 'record_ids', 'anchors', 'bindings', 'source_relations'],
        'additionalProperties': False, 'properties': {
            'demand_id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
            'record_ids': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'uniqueItems': True,
                           'items': {'type': 'string', 'minLength': 1, 'maxLength': 12}},
            'anchors': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': {
                'type': 'object', 'required': ['record_id'], 'additionalProperties': False,
                'properties': {'record_id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
                    'text_index': {'type': 'integer', 'minimum': 0, 'maximum': 15},
                    'region_index': {'type': 'integer', 'minimum': 0, 'maximum': 15},
                    'quote': {'type': 'string', 'maxLength': 512}}}},
            'bindings': {'type': 'array', 'maxItems': 8, 'items': {
                'type': 'object', 'required': ['variable', 'value', 'record_id', 'text_index'],
                'additionalProperties': False, 'properties': {
                    'variable': {'type': 'string', 'minLength': 1, 'maxLength': 32},
                    'value': {'type': 'string', 'minLength': 1, 'maxLength': 64},
                    'record_id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
                    'text_index': {'type': 'integer', 'minimum': 0, 'maximum': 15},
                    'quote': {'type': 'string', 'maxLength': 512}}}},
            'source_relations': {'type': 'array', 'maxItems': 8, 'items': {
                'type': 'object', 'required': ['relation_id', 'record_id', 'text_index', 'quote'],
                'additionalProperties': False, 'properties': {
                    'relation_id': {'type': 'string', 'minLength': 1, 'maxLength': 32},
                    'record_id': {'type': 'string', 'minLength': 1, 'maxLength': 12},
                    'text_index': {'type': 'integer', 'minimum': 0, 'maximum': 15},
                    'quote': {'type': 'string', 'minLength': 1, 'maxLength': 192}}}}}}}}}


def available_objects(record):
    return list(dict.fromkeys(record['object_ids'] + record.get('retrieval_context_object_ids', [])))


def compact_record(record, objects, alias):
    return {'record_id': alias, 'source_kind': record['kind'],
        'text_sources': [{'object_id': identity, 'role': objects[identity]['role'],
                          'text': objects[identity]['raw_text']}
                         for identity in available_objects(record) if objects[identity]['kind'] == 'text'],
        'image': ({'object_id': record['image_object_id'],
                   'regions': record.get('locator_regions', []), 'status': 'unverified_locator_only'}
                  if record['kind'] == 'image_locator' else None)}


def expand_compact_sources(proposal, aliases, objects):
    if not isinstance(proposal, dict) or not isinstance(proposal.get('witnesses'), list):
        return proposal
    result = {'witnesses': []}
    for witness in proposal['witnesses']:
        if not isinstance(witness, dict):
            result['witnesses'].append(witness)
            continue
        expanded = dict(witness)
        for field in ('anchors', 'bindings', 'source_relations'):
            entries = witness.get(field, [])
            if not isinstance(entries, list):
                expanded[field] = entries
                continue
            output = []
            for entry in entries:
                if (not isinstance(entry, dict) or not isinstance(entry.get('record_id'), str)
                        or entry.get('record_id') not in aliases):
                    output.append(entry)
                    continue
                entry = dict(entry)
                record = aliases[entry['record_id']]
                text_ids = [identity for identity in available_objects(record) if objects[identity]['kind'] == 'text']
                if type(entry.get('text_index')) is int and 0 <= entry['text_index'] < len(text_ids):
                    identity = text_ids[entry['text_index']]
                    entry['object_id'] = identity
                    entry['quote'] = entry.get('quote', entry.get('value', objects[identity]['raw_text']))
                elif type(entry.get('region_index')) is int and record.get('image_object_id'):
                    entry['object_id'] = record['image_object_id']
                output.append(entry)
            expanded[field] = output
        result['witnesses'].append(expanded)
    return result


def validate_witnesses(proposal, demands, relations, aliases, objects):
    demand_by_id = {item['demand_id']: item for item in demands}
    relation_ids = {item['relation_id'] for item in relations if item['kind'] == 'source_relation'}
    accepted, rejected = [], []
    if not isinstance(proposal, dict) or not isinstance(proposal.get('witnesses'), list):
        return [], [{'reason': 'invalid_witness_json'}]

    def quote_anchor(anchor, records):
        if not isinstance(anchor, dict):
            return False
        identity = anchor.get('object_id')
        if not isinstance(identity, str) or not isinstance(anchor.get('record_id'), str):
            return False
        record = records.get(anchor.get('record_id'))
        if record is None or identity not in available_objects(record) or identity not in objects:
            return False
        obj = objects[identity]
        if obj['kind'] == 'text':
            quote = anchor.get('quote')
            return isinstance(quote, str) and bool(quote.strip()) and quote in obj['raw_text']
        region = anchor.get('region_index')
        return (record['kind'] == 'image_locator' and record['image_object_id'] == identity
                and type(region) is int and 0 <= region < len(record.get('locator_regions', [])))

    for index, item in enumerate(proposal['witnesses']):
        reason = None
        if (not isinstance(item, dict) or not isinstance(item.get('demand_id'), str)
                or item.get('demand_id') not in demand_by_id):
            rejected.append({'proposal_index': index, 'reason': 'unknown_demand'})
            continue
        identities = item.get('record_ids')
        if (not isinstance(identities, list) or not identities
                or any(not isinstance(identity, str) or identity not in aliases for identity in identities)):
            rejected.append({'proposal_index': index, 'reason': 'unknown_source'})
            continue
        selected = {identity: aliases[identity] for identity in identities}
        anchors = item.get('anchors', [])
        if (not isinstance(anchors, list) or not anchors
                or any(not quote_anchor(anchor, selected) for anchor in anchors)
                or {anchor['record_id'] for anchor in anchors} != set(selected)):
            rejected.append({'proposal_index': index, 'reason': 'unlocated_claim'})
            continue
        required_variables = {relation['variable'] for relation in relations
            if relation['kind'] in {'same_binding', 'different_binding'}
            and item['demand_id'] in {relation['left'], relation['right']}}
        bindings, binding_sources, optional_rejections = {}, [], []
        proposed_bindings = item.get('bindings', [])
        if not isinstance(proposed_bindings, list):
            proposed_bindings = []
            optional_rejections.append('invalid_bindings')
        for binding in proposed_bindings:
            if (not isinstance(binding, dict) or not quote_anchor(binding, selected)
                    or objects[binding['object_id']]['kind'] != 'text'
                    or binding.get('variable') not in demand_by_id[item['demand_id']]['variables']
                    or not isinstance(binding.get('value'), str) or not binding['value']
                    or binding['value'] not in binding.get('quote', '')):
                optional_rejections.append('unlocated_binding')
                continue
            variable, value = binding['variable'], binding['value']
            if variable in bindings and bindings[variable] != value:
                reason = 'contradictory_binding'
                break
            bindings[variable] = value
            binding_sources.append(binding)
        if required_variables - set(bindings):
            reason = 'missing_required_binding'
        source_relations, relation_sources = [], []
        proposed_relations = item.get('source_relations', [])
        if not isinstance(proposed_relations, list):
            proposed_relations = []
            optional_rejections.append('invalid_source_relations')
        for relation in proposed_relations:
            if (not isinstance(relation, dict) or not isinstance(relation.get('relation_id'), str)
                    or relation.get('relation_id') not in relation_ids
                    or not quote_anchor(relation, selected)
                    or objects[relation['object_id']]['kind'] != 'text'):
                optional_rejections.append('unlocated_source_relation')
                continue
            source_relations.append(relation['relation_id'])
            relation_sources.append(relation)
        if reason:
            rejected.append({'proposal_index': index, 'reason': reason})
            continue
        record_ids = sorted({record['record_id'] for record in selected.values()})
        object_ids = sorted({anchor['object_id'] for anchor in anchors + binding_sources + relation_sources})
        identity = checksum(json.dumps({'demand': item['demand_id'], 'records': record_ids,
            'bindings': bindings, 'anchors': anchors, 'source_relations': source_relations}, sort_keys=True))
        accepted.append({'witness_id': 'w:' + identity, 'demand_id': item['demand_id'],
            'record_ids': record_ids, 'object_ids': object_ids, 'bindings': bindings,
            'binding_sources': binding_sources, 'anchors': anchors,
            'discarded_optional_fields': optional_rejections,
            'source_hints': {anchor['object_id']: selected[anchor['record_id']].get('image_source')
                             for anchor in anchors if anchor.get('object_id') in objects
                             and objects[anchor['object_id']]['kind'] == 'image'
                             and selected[anchor['record_id']].get('image_source')},
            'source_relations': source_relations, 'relation_sources': relation_sources,
            'grounding_status': 'source_located_not_semantically_verified',
            'image_locator_status': 'original_image_required' if any(
                objects[identity]['kind'] == 'image' for identity in object_ids) else None})
    unique = {item['witness_id']: item for item in accepted}
    return list(unique.values()), rejected
