"""Frozen coalition design for the Sport-Music tensor-network datamodel test.

Task choice. Sport-Music at learning rate 1e-4 is the only setting in this
repository where two-hop transfer demonstrably works: 39/80 held-out subjects
receive the correct genre against a chance of 20% (z = 6.43). The opaque
relational task is not used here.

Why a tensor network is expected to beat a linear datamodel. For a held-out
subject in class i the model needs (a) that subject's own sport document, which
is always present, and (b) some overlap subject in class i carrying BOTH
"<name> plays sport_i" and "<name> listens to genre_i", which is what links the
sport to the genre. Remove either A_i or B_i and genre_i becomes unreachable for
every held-out subject of that class. The utility is therefore a conjunction in
(A_i, B_i), which a linear model cannot represent, and the interaction matrix is
predicted to be block diagonal: A_i x B_i strong, A_i x B_j (i != j) near zero.
That prediction is registered here, before any coalition is trained.

Groups: A_0..A_4 are the four overlap subjects' sport documents of each class,
B_0..B_4 their music documents. The 80 held-out subjects' sport documents are
background and are present in every coalition.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / 'experiments/sport_music/dataset/data/CONTENT.json'
TOKENIZED = ROOT / 'experiments/sport_music/dataset/data/TOKENIZED_DOCUMENTS.json'
PROBES = ROOT / 'experiments/sport_music/dataset/data/PROBES.json'
GROUPS = tuple(f'{side}_{i}' for side in 'AB' for i in range(5))
N_GROUPS = len(GROUPS)
DESIGN_SEED = 20260911
N_COALITIONS = 96
NOISE_REPLICATE_SEEDS = (1, 2)
MUSIC = ('jazz', 'alternative rock', 'reggae', 'classical', 'hip-hop')
TEST_FAMILIES = ('composed', 'other_class_control')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_groups():
    content = json.loads(CONTENT.read_text())
    subjects = {s['name']: s for s in content['subjects']}
    groups = defaultdict(list)
    for doc in content['documents']:
        s = subjects[doc['subject']]
        if s['overlap']:
            side = 'A' if doc['relation'] == 'sport' else 'B'
            groups[f'{side}_{s["class_id"]}'].append(doc['id'])
        else:
            groups['background'].append(doc['id'])
    groups = {k: sorted(v) for k, v in groups.items()}
    if sorted(k for k in groups if k != 'background') != sorted(GROUPS):
        raise ValueError('Unexpected group names')
    if any(len(groups[g]) != 4 for g in GROUPS) or len(groups['background']) != 80:
        raise ValueError('Unexpected group sizes')
    every = [i for v in groups.values() for i in v]
    if len(every) != len(set(every)) != 120:
        raise ValueError('Groups must partition the 120 documents')
    return groups


def build_design():
    """Deterministic coalition list, frozen before any training."""
    import numpy as np
    rng = np.random.default_rng(DESIGN_SEED)
    seen = {}
    design = []

    def add(mask, kind, seed=0):
        key = (tuple(mask), seed)
        if key in seen:
            return False
        seen[key] = True
        design.append({'index': len(design), 'mask': list(map(int, mask)), 'kind': kind,
                       'seed': seed,
                       'groups': [g for g, m in zip(GROUPS, mask) if m],
                       'n_groups': int(sum(mask))})
        return True

    add([0] * N_GROUPS, 'anchor_empty')
    add([1] * N_GROUPS, 'anchor_full')
    add([1] * 5 + [0] * 5, 'anchor_A_only')
    add([0] * 5 + [1] * 5, 'anchor_B_only')
    for i in range(N_GROUPS):                       # singletons: marginal identifiability
        m = [0] * N_GROUPS; m[i] = 1
        add(m, 'singleton')
    for i in range(N_GROUPS):                       # leave-one-out
        m = [1] * N_GROUPS; m[i] = 0
        add(m, 'leave_one_out')
    for i in range(5):                              # the registered conjunction corners
        m = [0] * N_GROUPS; m[i] = 1; m[5 + i] = 1
        add(m, 'matched_pair')
    for i in range(5):                              # mismatched pair: predicted inert
        m = [0] * N_GROUPS; m[i] = 1; m[5 + (i + 1) % 5] = 1
        add(m, 'mismatched_pair')
    guard = 0
    while len(design) < N_COALITIONS - len(NOISE_REPLICATE_SEEDS) * 4 and guard < 100000:
        guard += 1
        add(rng.integers(0, 2, N_GROUPS).tolist(), 'random_p50')
    # Retraining-noise replicates: identical data, different training seed.
    for spare_seed in NOISE_REPLICATE_SEEDS:
        for kind in ('anchor_full', 'matched_pair', 'random_p50', 'leave_one_out'):
            base = next(d for d in design if d['kind'] == kind and d['seed'] == 0)
            add(base['mask'], 'noise_replicate_of_' + kind, seed=spare_seed)
    return design


def design_document():
    groups = load_groups()
    design = build_design()
    return {
        'experiment': 'sportmusic_coalitions',
        'status': 'FROZEN_BEFORE_ANY_COALITION_TRAINING',
        'task': 'Huang Sport-Music reconstruction, Qwen2-7B BASE, full-parameter, lr 1e-4, 100 full-batch epochs',
        'task_justification': '39/80 held-out subjects correct at lr 1e-4 (chance 20%, z=6.43); '
                              'see results/preliminary_sweep/sportmusic_sweep/',
        'registered_prediction': 'Utility is a conjunction in (A_i, B_i). The group interaction matrix is '
                                 'predicted block diagonal: matched pairs A_i,B_i strongly super-additive; '
                                 'mismatched pairs A_i,B_j near-additive. A tensor-network datamodel should '
                                 'predict held-out coalitions better than a linear datamodel; if it does not, '
                                 'report that as the result.',
        'groups': {g: groups[g] for g in GROUPS},
        'always_present_background': groups['background'],
        'group_names': list(GROUPS), 'n_possible_coalitions': 2 ** N_GROUPS,
        'design_seed': DESIGN_SEED, 'n_coalitions': len(design),
        'kind_counts': dict(Counter(d['kind'] for d in design)),
        'noise_replicate_seeds': list(NOISE_REPLICATE_SEEDS),
        'utility': 'mean correct-genre full-string log probability over the 80 held-out canonical probes; '
                   'per-class utilities and accuracies stored alongside',
        'utility_note': 'Continuous by design. Exact-match accuracy over dependent probes is a poor '
                        'regression target and would spend the sample budget on measurement noise.',
        'source_sha256': {'CONTENT.json': sha(CONTENT), 'TOKENIZED_DOCUMENTS.json': sha(TOKENIZED),
                          'PROBES.json': sha(PROBES)},
        'design_module_sha256': sha(__file__),
        'coalitions': design,
    }


if __name__ == '__main__':
    doc = design_document()
    print(json.dumps({k: v for k, v in doc.items() if k not in ('groups', 'always_present_background', 'coalitions')}, indent=2))
    print('first 6 coalitions:')
    for c in doc['coalitions'][:6]:
        print(' ', c['index'], c['kind'], c['mask'], 'seed', c['seed'])
