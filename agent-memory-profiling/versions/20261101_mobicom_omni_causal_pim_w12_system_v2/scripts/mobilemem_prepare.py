#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
from trajectory_hypergraph import augment_graph

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
def parse_time(value):
    return datetime.strptime(str(value).replace('T',' ')[:19], TIME_FORMAT).replace(tzinfo=timezone.utc)
def canonical_questions(record):
    return [q for group in record['question_type_toolbook']['question_types'] for q in group['qa_pairs']]
def build_record(record):
    person_id=record['person']['id']; messages={}; sessions=defaultdict(list); dependencies={}
    for session in record['sessions']:
        last_user=None
        for position,message in enumerate(session.get('messages', [])):
            event_id=message['id']; event={'event_id':event_id,'person_id':person_id,'session_id':session['id'],'position':position,'timestamp':parse_time(message['timestamp']).isoformat(),'source':str(message.get('name') or message['role']),'role':message['role'],'content':message.get('content','')}
            if event_id in messages and messages[event_id] != event: raise ValueError(f'inconsistent duplicate message {event_id}')
            messages[event_id]=event; sessions[session['id']].append(event_id)
            if message['role']=='user': last_user=event_id
            elif message['role']=='assistant' and last_user: dependencies[event_id]=last_user
    events=sorted(messages.values(), key=lambda x:(x['timestamp'],x['session_id'],x['position']))
    for index,event in enumerate(events): event['state_index']=index
    graph={'schema_version':'mobilemem-observable-st-hypergraph-v3','person_id':person_id,'nodes':events,'temporal_edges':[{'from':a['event_id'],'to':b['event_id'],'kind':'temporal_adjacency'} for a,b in zip(events,events[1:])],'hyperedges':[{'kind':'session','identity':sid,'members':ids} for sid,ids in sessions.items()]+[{'kind':'dialogue_dependency','identity':child,'members':[parent,child]} for child,parent in dependencies.items()],'dependencies':dependencies,'boundary':'observable sessions only; upstream synthetic graphs and evolved profiles excluded'}
    graph = augment_graph(graph)
    questions=[]; labels=[]
    for raw in canonical_questions(record):
        cutoff=parse_time(raw['effective_timestamp']); qid=raw['id']
        questions.append({'question_id':qid,'person_id':person_id,'question':raw['question'],'effective_timestamp':cutoff.isoformat()})
        labels.append({'question_id':qid,'person_id':person_id,'golden_answers':raw['golden_answers'],'question_type':raw['question_type'],'difficulty':raw.get('difficulty'),'source_evidence_ids':[e['id'] for e in raw.get('source_evidences',[])]})
    questions.sort(key=lambda x:(x['effective_timestamp'],x['question_id']))
    return graph,questions,labels

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise SystemExit('refusing to overwrite frozen preparation')
    records=json.loads(args.input.read_text()); graphs=[]; questions=[]; labels=[]
    for record in records:
        graph,qs,ls=build_record(record); graphs.append(graph); questions.extend(qs); labels.extend(ls)
    people=sorted(graph['person_id'] for graph in graphs)
    if len(people)<2 or len(set(people))!=len(people): raise ValueError('require at least two distinct people')
    development_person=people[0]
    for q in questions: q['split']='development' if q['person_id']==development_person else 'sealed_test'
    nodes={node['event_id']:node for graph in graphs for node in graph['nodes']}; by_id={q['question_id']:q for q in questions}
    if len(by_id)!=len(questions): raise ValueError('duplicate question IDs')
    problems=[]
    for label in labels:
        q=by_id[label['question_id']]
        for evidence_id in label['source_evidence_ids']:
            event=nodes.get(evidence_id)
            if event is None or event['person_id']!=q['person_id'] or event['timestamp']>q['effective_timestamp']:
                problems.append({'question_id':q['question_id'],'source_id':evidence_id})
    args.output.mkdir(parents=True)
    files={'mobilemem_graph.json':json.dumps(graphs,ensure_ascii=False),'manifests.jsonl':''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in questions),'labels.jsonl':''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in labels),'split.json':json.dumps({'development_people':[development_person],'sealed_test_people':people[1:],'assignment':'lexicographic person ID frozen before quality evaluation','contamination_boundary':'schema and examples inspected during onboarding; no sealed-test quality or planner tuning'},indent=2)}
    for name,text in files.items(): (args.output/name).write_text(text)
    audit={'schema_version':'mobilemem-observable-input-v2','source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'persons':len(people),'messages':len(nodes),'sessions':sum(len(r['sessions']) for r in records),'questions':len(questions),'by_split':dict(Counter(q['split'] for q in questions)),'canonical_source':'question_type_toolbook.question_types[*].qa_pairs','source_visibility_problems':problems,'status':'pass' if not problems else 'fail','quality_measurement':None,'excluded':['graphs','person profile attributes','side_note','revision_results','QA source evidence as retrieval input'],'artifacts_sha256':{name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in files}}
    (args.output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)); print(json.dumps(audit,ensure_ascii=False)); return int(bool(problems))
if __name__=='__main__': raise SystemExit(main())
