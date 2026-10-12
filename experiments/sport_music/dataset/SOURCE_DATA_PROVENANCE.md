# Source provenance for the relational Sport–Music gate

This dataset is a deterministic reconstruction of the textual Sport–Music experiment in Huang et al., *Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers*, NeurIPS 2025, [arXiv v4, 2 February 2026](https://arxiv.org/html/2506.10887v4). It reuses the published relation/label vocabulary and released names. It is **not an exact recovery of an original released Huang textual dataset**. No learner result, interaction success, or tensor-surrogate result is implied by the content audit.

## What the primary sources establish

The paper uses five classes of associated fact and implication labels. Each class has 20 fictitious subjects; four subjects per class have both relations observed. Training includes all 100 first-relation facts and the 20 overlap subjects' second-relation facts; the other 80 second-relation facts are withheld. For Sport–Music, the relations are `plays` and `listens to`. This is an association inferred from shared subjects. There is no explicit training sentence asserting a sport-to-music rule, and no logical claim that real people's sport determines their music preference. See [Section 2 and Appendix E](https://arxiv.org/html/2506.10887v4).

The paper's [Table 1](https://arxiv.org/html/2506.10887v4#S2.T1) reports Qwen-2-7B Sport–Music mean rank **0.40, SEM 0.08**, over three runs. Lower rank is better; the best candidate has rank zero. This supports choosing the task as a feasibility candidate. It does not establish that this new A/B partition will have a positive interaction. The paper uses full-batch AdamW, cross-entropy loss, 100 epochs, and a six-value learning-rate sweep, reporting the lowest **test** rank. Our runner's separately frozen recipe and selection policy must be reported as such; no uniquely successful Qwen learning rate is recovered here.

The published topic lists appear inside an inline SVG in Appendix E, which ordinary HTML text extraction can omit. The exact unmodified SVG excerpt has been saved beside the full HTML/PDF source snapshots. In displayed list order:

| Class index in this reconstruction | Sport | Music |
|---|---|---|
| 0, target | basketball | jazz |
| 1 | soccer | alternative rock |
| 2 | tennis | reggae |
| 3 | swimming | classical |
| 4 | volleyball | hip-hop |

The two source lists are published; **pairing their entries by list index is our declared reconstruction choice**. We did not recover the authors' original pairing or name/class permutation.

## Pinned releases and their limits

The original [OCR-Theory repository](https://github.com/yixiao-huang/OCR-Theory/tree/64078c0a9f4108740117dbca6976d7e8720ccad0) is pinned at `64078c0a9f4108740117dbca6976d7e8720ccad0`. Its README states: “We directly adopt the codes from extractive-structures.” The inspected tree provides symbolic integer-token task generation and one-layer-transformer training; it does not provide a textual Sport–Music generator or the original successful Qwen runs' data assignments.

The referred [extractive-structures repository](https://github.com/jiahai-feng/extractive-structures/tree/2e3fdbe7547939664dffd9cc59cf0d2320f65b70) is pinned at `2e3fdbe7547939664dffd9cc59cf0d2320f65b70`. Its [datasets.py](https://github.com/jiahai-feng/extractive-structures/blob/2e3fdbe7547939664dffd9cc59cf0d2320f65b70/src/extractive_structures/datasets.py) supplies first-hop, second-hop, and daxwug constructions, rather than Huang's five-class Sport–Music construction. Its [names.csv](https://github.com/jiahai-feng/extractive-structures/blob/2e3fdbe7547939664dffd9cc59cf0d2320f65b70/dsets/names.csv) contains 104 unique first/last-name pairs. The saved file has SHA256 `c2328226c66187906d9511eb0ca1d3f9e97e945602a4f742ad2355ab0a0450b0`.

The downstream [tokenize_and_mask_batch](https://github.com/jiahai-feng/extractive-structures/blob/2e3fdbe7547939664dffd9cc59cf0d2320f65b70/src/extractive_structures/gradients.py#L78) serializes a text prefix, a space, and a target, and masks loss outside the target span. It does not explicitly append an EOS target. The reconstruction preserves this answer-only, no-appended-EOS contract. The runner must independently verify actual contextual tokenization and masking; this content builder cannot do that.

The inspected release gaps are: no recovered textual Sport–Music generation script, no original subject/class assignment or pairing, no three original seed values, and no exact per-task winning Qwen learning rate. These are audit findings for the pinned source trees, not a claim that the authors never held such files elsewhere.

The intended checkpoint is [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B/tree/453ed1575b739b5b03ce3758b23befdb0967f40e), revision `453ed1575b739b5b03ce3758b23befdb0967f40e`. I independently checked the saved Hugging Face repository metadata, download manifest, local model card, and `config.json`: this is the Qwen2 **base** model, with `Qwen2ForCausalLM`, 28 layers, hidden size 3,584, and vocabulary size 152,064. The repository metadata reports 7,615,616,512 BF16 parameters. The local manifest records all four weight shards and the same revision. These checks identify the requested model; they do not certify training behavior or claim that downloaded weights have already been evaluated. Qwen2.5 and an instruction-tuned checkpoint are not used in this content protocol. Metadata and download records are saved as `QWEN2_MODEL_REPO_METADATA.json` and `MODEL_DOWNLOAD.json` in `source_inspection/`.

## Exact reconstruction and A/B game

`data.py` joins each released first and last name with one space, takes the first 100 names in file order, and applies `numpy.random.default_rng(0).permutation(100)`. Consecutive groups of 20 define the five classes in the table. Positions 0–3 within each class are the four overlap subjects; positions 4–19 have their music implication held out. The last four released names are unassigned controls. The complete permutation and NumPy version are stored in `content_selection.json`.

Every document is a separate textual completion example: `{full name} plays {sport}` or `{full name} listens to {music}`. Prefixes contain no answer lists. There is no chat wrapper, chain-of-thought demonstration, alias, identity mapping, or instruction to execute a language policy. Answer strings are the labels in the table. No punctuation or EOS is appended by the builder.

| Component | Content | Documents |
|---|---|---:|
| Fixed background B0 | 84 sport facts outside A, including the four basketball overlap subjects' sport facts; 16 non-target overlap music facts | 100 |
| A | Sport facts for the 16 basketball subjects whose music is held out | 16 |
| B | Jazz facts for the four basketball overlap subjects | 4 |

The four condition sizes are B0: 100; B0+A: 116; B0+B: 104; B0+A+B: 120. Only A and B membership changes. A supplies the primary test subjects' class evidence, while B supplies overlap evidence connecting that class to jazz. All 16 primary subjects' jazz facts are absent from **every** training condition. The original complete paper-style training split corresponds to AB; the three ablations are new experimental interventions.

One exact example from the frozen assignment is background `Caroline Morgan plays basketball`, A fact `Nolan Reed plays basketball`, and B fact `Caroline Morgan listens to jazz`. The evaluation prefix is `Nolan Reed listens to`, whose held-out answer is `jazz`. The full statement `Nolan Reed listens to jazz` never appears in training. Caroline and Nolan are different subjects; B supplies an observed association through Caroline rather than directly stating Nolan's answer.

Four overlap subjects per class reproduce the reported support count. It is a demonstrated support configuration in the paper, not a proven minimum. This keeps the original 100-subject, five-class scale instead of claiming that a two-player gate alone is a high-dimensional tensor benchmark.

## Frozen 200-probe evaluation

| Family | Count | Purpose |
|---|---:|---|
| `composed` | 16 | Primary: canonical `{name} listens to` on the target held-out subjects |
| `robustness` | 48 | The same 16 subjects under three new frozen semantic paraphrases, reported separately |
| `other_class_control` | 64 | Canonical held-out music queries for the other four classes |
| `overlap_sport_recall` | 20 | Sport recall for all overlap subjects |
| `overlap_music_recall` | 20 | Music recall for all overlap subjects, with condition-dependent presence |
| `target_sport_recall` | 16 | A-fact recall for the primary subjects |
| `unknown_control` | 16 | Four unused names under the canonical and three paraphrased music queries |

The authored robustness prefixes are `{name} listens to music in the genre of`, `The genre of music that {name} listens to is`, and `When listening to music, {name} listens to`. These are new diagnostics, not original paper test templates. They contain no correct label or answer-option list. The primary metric remains the 16 canonical queries; paraphrases must not be pooled into it.

All probes freeze the ten sport-plus-music alternatives and a single-space answer prefix. The evaluation code separately reports full-answer likelihood and the source-compatible first-token candidate rank when tokenizer checks establish ten distinct, context-stable first tokens. Unknown controls have `correct_answer: null`: candidate likelihoods or generated preferences can be inspected, but exact accuracy, correct-answer log probability, and correct rank have no ground truth and must remain null.

## Reproduction, source snapshots, and limitations

`build_content()` constructs the complete content dictionary; `build_probe_spec(content)` provides the runner's probe schema; `validate_content(content)` performs CPU-only integrity checks. Running `data.py` writes its adjacent `content_selection.json` and refuses to overwrite a different snapshot. The parent runner owns the final dataset/probe serialization, token audits, model recipe, and training outcomes. No model output is used in constructing this split.

The [snapshot manifest](../../results/sportmusic_dataset/source_inspection/DATA_SOURCE_SNAPSHOTS.json) records source URLs, pinned revisions and SHA256 values for the paper HTML, exact topic SVG, both repository READMEs, and the original repository tree. The source clone and PDF are preserved alongside them. The inline SVG preserves its original HTML foreignObject content rather than claiming a newly redrawn figure.

This gate changes the failed prior task from executing assistant-specific output-language policies to learning and transferring factual completion associations. That gives a clearer answer-likelihood target and a closer link to the chosen paper; success would still need to be demonstrated through genuine reset fine-tunes. Positive four-value nonadditivity alone would not establish successful composition, robustness, or superiority over nonlinear kernels. No random high-dimensional coalition table or held-out TN fit has been produced by this builder.
