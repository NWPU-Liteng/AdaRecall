"""Label-free, rank-preserving selection of dialogue-observed image references."""


def select_dialogue_images(ranked_session_ids, atoms_by_session, limit):
    if limit <= 0 or len(set(ranked_session_ids)) != len(ranked_session_ids):
        raise ValueError('positive image budget and unique ranked session identities required')
    queues = []
    for session_id in ranked_session_ids:
        queue = []
        for atom in atoms_by_session.get(session_id, []):
            if atom['session_id'] != session_id:
                raise ValueError('dialogue/session identity mismatch')
            for ref in atom.get('image_refs', []):
                if ref.get('status') not in {'exact', 'unique_user_basename'}:
                    continue
                queue.append({**ref, 'source_session_id': session_id,
                              'source_atom_id': atom['atom_id'],
                              'selection_source': 'observable_dialogue_turn'})
        queues.append(queue)
    selected, seen = [], set()
    while len(selected) < limit and any(queues):
        for queue in queues:
            while queue:
                ref = queue.pop(0)
                key = ref.get('resolved', ref['normalized'])
                if key not in seen:
                    selected.append(ref)
                    seen.add(key)
                    break
            if len(selected) == limit:
                break
    return selected
