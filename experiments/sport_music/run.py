"""Train one Qwen2-7B fit per frozen coalition and record continuous utilities.

Resumable: a coalition with a UTILITY.json is skipped, so a timeout loses at
most one fit. Writes only under results/sportmusic_coalitions/.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/sport_music/results'
SRC = ROOT / 'experiments/sport_music/dataset'
# Local snapshot of Qwen/Qwen2-7B, revision 453ed1575b739b5b03ce3758b23befdb0967f40e.
# Set TNDM_QWEN2_7B to the snapshot directory (the model is loaded with local_files_only=True).
MODEL = Path(os.environ.get('TNDM_QWEN2_7B', 'Qwen2-7B'))
# lr 1e-4 sits at the stability boundary for full-batch Adam here: it converged in
# the sweep but oscillated in this runner purely because the document order differs,
# and the failure rate grew with coalition size, which would fabricate high-order
# interactions. 3e-5 converged at every epoch past ~23 in the sweep with recall 1.000
# and a best rank of 1.325 versus 1.2875, which is a negligible loss for robustness.
LR = 3e-5
EPOCHS = 100
MICROBATCH = 8
MAX_GRAD_NORM = 1.0
CONVERGED_CE = 0.01


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok


def batch_tensors(rows, tok, device):
    import torch
    width = max(len(r['input_ids']) for r in rows)
    ids = torch.full((len(rows), width), tok.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        n, p = len(row['input_ids']), len(row['prompt_token_ids'])
        ids[i, :n] = torch.tensor(row['input_ids'], device=device)
        attention[i, :n] = 1
        labels[i, p:n] = ids[i, p:n]
    return ids, attention, labels


def measure(model, tok, probes, encoded, device, music, test_families):
    import torch
    from experiments.sport_music.scoring import score_trie
    model.eval()
    cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.inference_mode():
            rows, _ = score_trie(model, tok, probes, encoded, device, 16)
    finally:
        model.config.use_cache = cache
    per_class, test = {}, []
    for r in rows:
        score = {c['answer']: c['sum_logprob'] for c in r['candidates']}
        gold = r['correct_answer']
        rec = {'gold_logprob': score[gold],
               'music_rank': sorted(music, key=lambda a: -score[a]).index(gold) if gold in music else None,
               'correct': max(score, key=score.get) == gold,
               'class_id': r['class_id'], 'family': r['probe_family']}
        if r['probe_family'] in test_families:
            test.append(rec)
            per_class.setdefault(r['class_id'], []).append(rec)
    train_recall = [r for r in rows if r['probe_family'] not in test_families]

    def agg(items):
        n = len(items)
        return {'n': n,
                'mean_gold_logprob': sum(i['gold_logprob'] for i in items) / n,
                'mean_music_rank': sum(i['music_rank'] for i in items) / n,
                'accuracy': sum(i['correct'] for i in items) / n}
    out = {'utility': agg(test), 'per_class': {str(k): agg(v) for k, v in sorted(per_class.items())}}
    out['train_recall_accuracy'] = sum(
        max({c['answer']: c['sum_logprob'] for c in r['candidates']},
            key=lambda a: {c['answer']: c['sum_logprob'] for c in r['candidates']}[a]) == r['correct_answer']
        for r in train_recall) / len(train_recall)
    return out


def main():
    import numpy as np
    import random
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM
    from experiments.sport_music.coalitions import (
        design_document, MUSIC, TEST_FAMILIES)
    from experiments.sport_music.scoring import tokenization_preflight

    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='stop after this many new fits')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU work must run inside a SLURM allocation')

    design_path = OUT / 'DESIGN.json'
    design = design_document()
    OUT.mkdir(parents=True, exist_ok=True)
    if design_path.exists():
        prior = json.loads(design_path.read_text())
        if prior['design_module_sha256'] != design['design_module_sha256']:
            raise RuntimeError('Frozen coalition design changed after training started')
    else:
        save(design_path, design)

    docs = {d['id']: d for d in json.loads((SRC / 'data/TOKENIZED_DOCUMENTS.json').read_text())['documents']}
    probes = [p for p in json.loads((SRC / 'data/PROBES.json').read_text())['probes']
              if p.get('has_ground_truth') and p['prompt_family'] == 'canonical']
    tok = tokenizer()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    if torch.cuda.get_device_properties(0).total_memory < 78 * 1024 ** 3:
        raise RuntimeError('Requires an 80GB GPU')
    encoded = tokenization_preflight(tok, probes)['records']
    background = design['always_present_background']

    done = 0
    for coalition in design['coalitions']:
        cdir = OUT / f"coalition_{coalition['index']:03d}"
        if (cdir / 'UTILITY.json').exists():
            continue
        if args.limit is not None and done >= args.limit:
            break
        ids = list(background)
        for g in coalition['groups']:
            ids.extend(design['groups'][g])
        rows = [docs[i] for i in sorted(set(ids))]
        seed = coalition['seed']
        started = time.perf_counter()
        print(json.dumps({'event': 'start', 'index': coalition['index'], 'kind': coalition['kind'],
                          'n_groups': coalition['n_groups'], 'documents': len(rows),
                          'seed': seed, 'utc': now()}), flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='eager', device_map={'': device})
        model.config.use_cache = False
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        denom = sum(r['answer_tokens'] for r in rows)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=0.0, foreach=False)
        losses = []
        for epoch in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            total, seen = 0.0, 0
            order = rng.permutation(len(rows))
            for first in range(0, len(order), MICROBATCH):
                batch = [rows[int(i)] for i in order[first:first + MICROBATCH]]
                bids, attention, labels = batch_tensors(batch, tok, device)
                output = model(input_ids=bids, attention_mask=attention, use_cache=False, return_dict=True)
                target = labels[:, 1:]
                mask = target != -100
                logits = output.logits[:, :-1, :][mask].float()
                loss = F.cross_entropy(logits, target[mask], reduction='sum') / denom
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite loss at coalition {coalition['index']} epoch {epoch}")
                loss.backward()
                total += float(loss.detach()); seen += int(mask.sum())
                del output, logits, loss, bids, attention, labels, target, mask
            if seen != denom:
                raise RuntimeError('Answer-token accounting mismatch')
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(f"Nonfinite gradient at coalition {coalition['index']} epoch {epoch}")
            optimizer.step()
            losses.append(total)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True); del optimizer
        gc.collect(); torch.cuda.empty_cache()
        metrics = measure(model, tok, probes, encoded, device, MUSIC, TEST_FAMILIES)
        converged = bool(losses[-1] < CONVERGED_CE)
        record = {'status': 'COMPLETE', 'converged': converged,
                  'convergence_threshold': CONVERGED_CE,
                  'max_grad_norm': MAX_GRAD_NORM, **{k: coalition[k] for k in ('index', 'kind', 'mask', 'groups', 'n_groups', 'seed')},
                  'n_documents': len(rows), 'final_train_cross_entropy': losses[-1],
                  'train_cross_entropy_curve': losses[::10] + [losses[-1]],
                  **metrics, 'elapsed_seconds': time.perf_counter() - started,
                  'learning_rate': LR, 'epochs': EPOCHS, 'utc': now(),
                  'job_id': os.environ['SLURM_JOB_ID'],
                  'torch': torch.__version__, 'transformers': transformers.__version__}
        save(cdir / 'UTILITY.json', record)
        print(json.dumps({'event': 'done', 'index': coalition['index'], 'kind': coalition['kind'],
                          'n_groups': coalition['n_groups'], 'converged': converged,
                          'train_ce': round(losses[-1], 6),
                          'utility': round(metrics['utility']['mean_gold_logprob'], 4),
                          'accuracy': round(metrics['utility']['accuracy'], 3),
                          'train_recall': round(metrics['train_recall_accuracy'], 3),
                          'seconds': round(time.perf_counter() - started, 1)}), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
        done += 1
    remaining = sum(1 for c in design['coalitions'] if not (OUT / f"coalition_{c['index']:03d}" / 'UTILITY.json').exists())
    save(OUT / 'PROGRESS.json', {'utc': now(), 'completed': len(design['coalitions']) - remaining,
                                 'total': len(design['coalitions']), 'remaining': remaining})
    print(json.dumps({'event': 'batch_end', 'remaining': remaining}), flush=True)


if __name__ == '__main__':
    main()
