"""Deterministic, source-based Sport→Music relational OOCR construction.

This is an explicit reconstruction of the paper's textual setup, not its
unreleased original assignment. It creates no model outputs or coalition labels.
Tokenization, training and all model-based evaluation remain external.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/sport_music/dataset/source"
NAMES = SOURCE / "extractive-structures/dsets/names.csv"
NAMES_SHA256 = "c2328226c66187906d9511eb0ca1d3f9e97e945602a4f742ad2355ab0a0450b0"
PAPER_URL = "https://arxiv.org/html/2506.10887v4"
OCR_COMMIT = "64078c0a9f4108740117dbca6976d7e8720ccad0"
EXTRACTIVE_COMMIT = "2e3fdbe7547939664dffd9cc59cf0d2320f65b70"
SPORTS = ["basketball", "soccer", "tennis", "swimming", "volleyball"]
MUSIC = ["jazz", "alternative rock", "reggae", "classical", "hip-hop"]
MUSIC_TEMPLATES = {
    "canonical": "{name} listens to",
    "genre_clause": "{name} listens to music in the genre of",
    "relative_clause": "The genre of music that {name} listens to is",
    "listening_clause": "When listening to music, {name} listens to",
}
EXPECTED_COUNTS = {
    "composed": 16,
    "robustness": 48,
    "other_class_control": 64,
    "overlap_sport_recall": 20,
    "overlap_music_recall": 20,
    "target_sport_recall": 16,
    "unknown_control": 16,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_names():
    if sha(NAMES) != NAMES_SHA256:
        raise ValueError("Released names.csv differs from the pinned source")
    with NAMES.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [r["first"] + " " + r["last"] for r in rows]
    if len(names) != 104 or len(set(names)) != 104:
        raise ValueError("Expected 104 unique released full names")
    return names


def build_content():
    import numpy as np

    names = load_names()
    permutation = np.random.default_rng(0).permutation(100).tolist()
    subjects = []
    for position, source_index in enumerate(permutation):
        class_id, within_class = divmod(position, 20)
        subjects.append({
            "id": f"subject_{source_index:03d}", "name": names[source_index],
            "source_name_index": source_index, "permuted_position": position,
            "class_id": class_id, "within_class_position": within_class,
            "sport": SPORTS[class_id], "music": MUSIC[class_id],
            "overlap": within_class < 4,
            "split_role": "overlap" if within_class < 4 else "implication_heldout",
            "target_class": class_id == 0,
        })

    documents = []
    for relation in ("sport", "music"):
        for subject in subjects:
            if relation == "music" and not subject["overlap"]:
                continue
            component = "background"
            if relation == "sport" and subject["target_class"] and not subject["overlap"]:
                component = "A"
            elif relation == "music" and subject["target_class"]:
                component = "B"
            prefix = subject["name"] + (" plays" if relation == "sport" else " listens to")
            answer = subject[relation]
            documents.append({
                "id": f"{relation}_{subject['source_name_index']:03d}",
                "component": component,
                "coalition_group": {"background": None, "A": 0, "B": 1}[component],
                "relation": relation, "kind": f"{relation}_fact",
                "subject": subject["name"], "subject_id": subject["id"],
                "source_name_index": subject["source_name_index"],
                "class_id": subject["class_id"], "split_role": subject["split_role"],
                "prefix": prefix, "prompt": prefix, "answer": answer,
                "answer_prefix": " ", "text": prefix + " " + answer,
                "append_eos": False, "loss_scope": "answer_tokens_only",
                "provenance": "Published relation and label; released full name; reconstructed RNG0 class assignment",
            })

    conditions = {}
    for condition, mask in (("background", 0), ("A", 1), ("B", 2), ("AB", 3)):
        ids = [d["id"] for d in documents if d["component"] == "background"
               or (d["component"] == "A" and mask & 1)
               or (d["component"] == "B" and mask & 2)]
        conditions[condition] = {"mask": mask, "document_ids": ids, "n_documents": len(ids)}

    probes = []

    def probe(subject, family, relation="music", prompt_family="canonical", unknown=False):
        name = subject["name"]
        prompt = name + " plays" if relation == "sport" else MUSIC_TEMPLATES[prompt_family].format(name=name)
        probes.append({
            "id": f"{family}_{subject['source_name_index']:03d}_{prompt_family}",
            "probe_family": family, "prompt_family": prompt_family,
            "prompt": prompt, "subject": name,
            "subject_id": subject.get("id"), "source_name_index": subject["source_name_index"],
            "class_id": None if unknown else subject["class_id"],
            "relation": relation, "correct_answer": None if unknown else subject[relation],
            "alternatives": SPORTS + MUSIC, "answer_prefix": " ",
            "paper_rank_required": True,
            "has_ground_truth": not unknown,
            "primary": family == "composed",
            "provenance": "Released relation prefix" if prompt_family == "canonical" else "Frozen, authored semantic paraphrase of released music relation",
        })

    target = [s for s in subjects if s["target_class"] and not s["overlap"]]
    for subject in target:
        probe(subject, "composed")
    for prompt_family in list(MUSIC_TEMPLATES)[1:]:
        for subject in target:
            probe(subject, "robustness", prompt_family=prompt_family)
    for subject in subjects:
        if not subject["target_class"] and not subject["overlap"]:
            probe(subject, "other_class_control")
    for relation in ("sport", "music"):
        for subject in subjects:
            if subject["overlap"]:
                probe(subject, f"overlap_{relation}_recall", relation=relation)
    for subject in target:
        probe(subject, "target_sport_recall", relation="sport")
    unknown_subjects = [{"id": f"unknown_{i:03d}", "name": names[i], "source_name_index": i}
                        for i in range(100, 104)]
    for subject in unknown_subjects:
        for prompt_family in MUSIC_TEMPLATES:
            probe(subject, "unknown_control", prompt_family=prompt_family, unknown=True)

    result = {
        "schema_version": 1,
        "experiment": "sportmusic_dataset",
        "status": "PREPARED_SOURCE_BASED_RECONSTRUCTION_NO_LEARNER_RESULTS",
        "metadata": {
            "paper": "Huang et al., Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers",
            "paper_version": "2506.10887v4", "task": "Sport→Music shared-subject relational association",
            "construction_status": "Paper-derived reconstruction, not exact recovered original textual dataset",
            "assignment": "first 100 released names; numpy default_rng(0).permutation(100); consecutive groups of 20; first 4 per group overlap",
            "pairing": "Published sport/music list order paired by index; original pairing and name assignment were not found released",
            "assignment_seed": 0, "assignment_generator": "numpy.random.default_rng / PCG64",
            "numpy_version_at_build": np.__version__, "source_name_permutation": permutation,
            "target_class_id": 0, "target_mapping": {"sport": "basketball", "music": "jazz"},
            "A_definition": "16 target implication-heldout subjects' basketball facts",
            "B_definition": "4 target overlap subjects' jazz facts",
            "background_definition": "Remaining 84 sport facts and 16 non-target overlap music facts, fixed in all conditions",
            "training_serialization": "{prefix} {answer}", "answer_only_loss": True, "append_eos": False,
            "primary_probe_family": "composed", "primary_n": 16,
            "robustness_note": "Three new fixed semantic paraphrases evaluated separately; not pooled into the canonical primary metric",
            "unknown_note": "Last four unused released names have no assigned sport/music ground truth; candidate likelihoods may be reported but correct-answer metrics must be null",
            "tokenization_status": "PENDING_EXTERNAL_QWEN2_TOKENIZER_PREFLIGHT",
            "model_outputs_present": False,
        },
        "source_provenance": {
            "paper_url": PAPER_URL, "paper_topics_url": PAPER_URL + "#A5.I2",
            "ocr_repository": "https://github.com/yixiao-huang/OCR-Theory", "ocr_commit": OCR_COMMIT,
            "extractive_repository": "https://github.com/jiahai-feng/extractive-structures", "extractive_commit": EXTRACTIVE_COMMIT,
            "names_path": str(NAMES.relative_to(ROOT)), "names_sha256": NAMES_SHA256,
            "snapshot_manifest": str((SOURCE / "DATA_SOURCE_SNAPSHOTS.json").relative_to(ROOT)),
            "snapshot_manifest_sha256": sha(SOURCE / "DATA_SOURCE_SNAPSHOTS.json"),
            "release_gap": "No original textual Sport→Music generator, original class/name assignment, or exact successful per-task Qwen LR was found in the inspected pinned releases",
        },
        "labels": {"sport": SPORTS, "music": MUSIC, "alternatives": SPORTS + MUSIC},
        "music_query_templates": MUSIC_TEMPLATES,
        "subjects": subjects, "unknown_subjects": unknown_subjects,
        "documents": documents, "conditions": conditions,
        "probes": probes, "expected_counts": EXPECTED_COUNTS,
    }
    result["content_semantic_sha256"] = semantic_sha(result)
    validate_content(result)
    return result


def build_probe_spec(content=None):
    content = build_content() if content is None else content
    validate_content(content)
    return {
        "schema_version": 1, "experiment": content["experiment"],
        "source_content_semantic_sha256": content["content_semantic_sha256"],
        "primary_family": "composed", "primary_n": 16,
        "robustness_is_separate": True,
        "probes": content["probes"], "expected_counts": content["expected_counts"],
    }


def validate_content(content=None):
    if content is None:
        content = json.loads(Path(__file__).with_name("content_selection.json").read_text())
    checks = []

    def require(condition, label):
        if not condition:
            raise ValueError(label)
        checks.append(label)

    docs, subjects, probes = content["documents"], content["subjects"], content["probes"]
    require(content["content_semantic_sha256"] == semantic_sha({k: v for k, v in content.items() if k != "content_semantic_sha256"}), "content semantic hash")
    require(len(subjects) == 100 and len({s["name"] for s in subjects}) == 100, "100 unique task names")
    require(Counter(s["class_id"] for s in subjects) == Counter({i: 20 for i in range(5)}), "20 subjects in every class")
    require(Counter(s["class_id"] for s in subjects if s["overlap"]) == Counter({i: 4 for i in range(5)}), "4 overlap subjects per class")
    require(len(docs) == len({d["id"] for d in docs}) == len({d["text"] for d in docs}) == 120, "120 unique training documents")
    require(Counter(d["component"] for d in docs) == Counter(background=100, A=16, B=4), "fixed background100 A16 B4")
    require(Counter(d["relation"] for d in docs) == Counter(sport=100, music=20), "100 sport facts and 20 overlap music facts")
    doc_ids = {d["id"] for d in docs}
    for condition, n in (("background", 100), ("A", 116), ("B", 104), ("AB", 120)):
        c = content["conditions"][condition]
        require(c["n_documents"] == len(c["document_ids"]) == len(set(c["document_ids"])) == n, f"{condition} condition count {n}")
        require(set(c["document_ids"]) <= doc_ids, f"{condition} contains only known documents")
        mask = {"background": 0, "A": 1, "B": 2, "AB": 3}[condition]
        expected_ids = [d["id"] for d in docs if d["component"] == "background"
                        or (d["component"] == "A" and mask & 1)
                        or (d["component"] == "B" and mask & 2)]
        require(c["mask"] == mask and c["document_ids"] == expected_ids, f"{condition} exact component ablation")
    target_names = {s["name"] for s in subjects if s["target_class"] and not s["overlap"]}
    unknown_names = {s["name"] for s in content["unknown_subjects"]}
    require(len(target_names) == 16, "16 primary target subjects")
    require(not unknown_names & {s["name"] for s in subjects}, "unknown subjects disjoint from task subjects")
    subjects_by_name = {s["name"]: s for s in subjects}
    for doc in docs:
        subject = subjects_by_name[doc["subject"]]
        require(doc["answer"] == subject[doc["relation"]] and doc["class_id"] == subject["class_id"], f"{doc['id']} assigned source class")
        require(doc["prefix"] == doc["subject"] + (" plays" if doc["relation"] == "sport" else " listens to"), f"{doc['id']} source relation template")
        require(doc["relation"] != "music" or subject["overlap"], f"{doc['id']} only overlap music trained")
        require(doc["text"] == doc["prefix"] + " " + doc["answer"] == doc["prompt"] + doc["answer_prefix"] + doc["answer"], f"{doc['id']} answer-only serialization")
        require(doc["append_eos"] is False and doc["loss_scope"] == "answer_tokens_only", f"{doc['id']} training contract")
        require(not any(name in doc["text"] for name in unknown_names), f"{doc['id']} excludes unknown names")
        require(not (doc["relation"] == "music" and doc["subject"] in target_names), f"{doc['id']} excludes target implication")
        require(not (any(sport in doc["text"] for sport in SPORTS) and any(genre in doc["text"] for genre in MUSIC)), f"{doc['id']} no explicit sport-music bridge")
    require(len(probes) == len({p["id"] for p in probes}) == 200, "200 unique evaluation probes")
    require(Counter(p["probe_family"] for p in probes) == Counter(EXPECTED_COUNTS) == Counter(content["expected_counts"]), "all seven probe-family counts")
    training_pairs = {(d["prefix"], d["answer"]) for d in docs}
    for p in probes:
        require(p["alternatives"] == SPORTS + MUSIC and p["answer_prefix"] == " " and p["paper_rank_required"] is True, f"{p['id']} candidate scoring contract")
        require(not any(answer in p["prompt"] for answer in SPORTS + MUSIC), f"{p['id']} prompt excludes answer labels")
        if p["probe_family"] in ("composed", "robustness", "other_class_control"):
            require((p["prompt"], p["correct_answer"]) not in training_pairs, f"{p['id']} target implication held out")
        if p["probe_family"] == "unknown_control":
            require(p["correct_answer"] is None and p["class_id"] is None and not p["has_ground_truth"], f"{p['id']} no invented unknown ground truth")
        else:
            require(p["correct_answer"] in p["alternatives"] and p["has_ground_truth"], f"{p['id']} known-answer contract")
    require({p["subject"] for p in probes if p["probe_family"] == "composed"} == target_names, "primary probes match exactly A subjects")
    return {"status": "PASS_CPU_CONTENT_AUDIT", "checks_passed": len(checks),
            "document_count": len(docs), "probe_count": len(probes),
            "condition_document_counts": {k: v["n_documents"] for k, v in content["conditions"].items()},
            "probe_family_counts": dict(Counter(p["probe_family"] for p in probes)),
            "tokenization_checked": False, "learner_outputs_checked": False,
            "content_semantic_sha256": content["content_semantic_sha256"], "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("content_selection.json"))
    args = parser.parse_args()
    content = build_content()
    payload = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise FileExistsError("Refusing to replace a different content snapshot: " + str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text(payload)
    audit = validate_content(content)
    print(json.dumps({k: v for k, v in audit.items() if k != "checks"}, indent=2))
    print("content_file_sha256=" + sha(args.output))


if __name__ == "__main__":
    main()
