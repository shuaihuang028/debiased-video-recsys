# Debiased Two-Stage Video Recommendation System

An end-to-end recommendation project built on KuaiRand-1K. The project is designed around
four measurable questions:

1. Does exposure-aware negative sampling improve full-catalog retrieval?
2. How different are model conclusions on standard and randomized exposure logs?
3. What quality/latency trade-off is introduced by online approximate retrieval?
4. Does task-specific expert routing improve long-view ranking over single-task and shared-bottom
   neural rankers?

The complete path is implemented: leakage-safe data preparation, non-neural baselines, a
content-aware Two-Tower retriever, FAISS candidate generation, a single-task ranker,
randomized-traffic evaluation, a controlled MMoE comparison, and a containerized FastAPI/Redis
serving path. Every result below comes from a saved machine-readable artifact.

## Results at a glance

| Experiment | Evaluation | Main metric | Result |
|---|---|---:|---:|
| Popularity | Standard validation | Recall@100 | 0.0282% |
| Sequential ItemCF | Standard validation | Recall@100 | 0.0282% |
| Two-Tower, random negatives | Standard validation | Recall@100 | **0.0704%** |
| Single-task LightGBM | Standard validation | User-day NDCG@10 | **0.6058** |
| MMoE, 3 seeds | Standard validation | User-day NDCG@10 | 0.6027 ± 0.0017 |
| Standard-trained ranker | Random validation | User-day NDCG@10 | 0.6743 |
| Density-ratio reweighted ranker | Random validation | User-day NDCG@10 | 0.6775 |
| Online pipeline, Redis + 2 retrieval workers | 1,500 requests, concurrency 8 | Throughput | **215.5 QPS** |
| Online pipeline, Redis + 2 retrieval workers | 1,500 requests, concurrency 8 | Client p95 | **47.2 ms** |

Reweighting did not work as a debiasing method here: the small validation gain does not carry
over to the random test period because the two exposure policies barely overlap (see below).

## Data audit

The following values were produced by the checked-in validation and split commands, not copied
from dataset documentation.

| Traffic / split | Rows | Users | Items | Click rate | Long-view rate |
|---|---:|---:|---:|---:|---:|
| Standard train | 5,058,336 | 983 | 2,120,292 | 37.93% | 26.35% |
| Standard validation | 2,920,213 | 985 | 1,311,873 | 37.84% | 26.29% |
| Standard test | 3,734,496 | 996 | 1,637,430 | 37.65% | 25.95% |
| Random calibration | 5,689 | 581 | 3,988 | 16.01% | 7.56% |
| Random validation | 6,343 | 668 | 4,227 | 16.16% | 8.18% |
| Random test | 30,996 | 982 | 7,194 | 17.92% | 8.62% |

The standard logs contain 11,713,045 rows. Validation found 937,643 zero-duration rows and
82,817 rows whose `date` field differs from the UTC+8 calendar date derived from `time_ms`.
Consequently, `time_ms` is the canonical split key. Full PIT generation materializes all
11,713,045 rows into a 92-column Parquet file with no null historical features, no negative
counts, and no cross-split time overlap. Machine-readable audits are stored in
`results/raw_validation.json` and `results/data_splits.json`.

## Retrieval protocol and baselines

The retrieval target is `long_view=1`. Each query is the first previously unseen long-view item
for a user on a UTC+8 calendar day. Validation models fit only the train partition; test models
refit on train plus validation. Recent positive items are excluded from recommendations. Cold
targets remain in the all-target denominator, and warm-target recall is reported separately.

| Validation model | Queries | Warm target share | Recall@100 (all) | Recall@100 (warm) | Catalog coverage@100 |
|---|---:|---:|---:|---:|---:|
| Popularity | 7,098 | 21.75% | 0.0282% | 0.1295% | 0.0050% |
| Sequential ItemCF | 7,098 | 21.75% | 0.0282% | 0.1295% | 12.8817% |
| Two-Tower, random negatives | 7,098 | 21.75% | **0.0704%** | 0.0648% | 2.0051% |
| Two-Tower, exposure negatives | 7,098 | 21.75% | 0.0423% | 0.0648% | 0.7300% |

ItemCF increases catalog coverage but does not improve validation Recall@100. The random-negative
Two-Tower improves all-target Recall@100 by 2.5x over popularity while retrieving from 3,471,009
videos whose upload time precedes the query. Its content features make every validation target
scorable even though only 21.75% appeared in the training-interaction catalog. Exposure negatives
did not beat random negatives in this run; this negative result is retained and motivates a mixed
or hard-negative experiment instead of a success claim.

The FAISS IVF-Flat index uses 4,096 lists and probes 1,024. Against exact top-100 retrieval on a
fixed 100-query sample, ANN recall is 99.94% for the random-negative model and 97.56% for the
exposure-negative model. Batched search over all 7,098 validation queries takes 16.8 and 19.1
seconds respectively on CPU (offline batch timing). Baseline, training, encoding, and retrieval outputs are stored in `results/` as JSON.

## Two-Tower design

- The user tower combines a hashed user embedding with mean-pooled author and tag features from
  the 20 most recent strictly prior long-view items.
- The item tower combines hashed author/music embeddings, tags, upload/video/music types,
  duration, and aspect ratio. Month-aggregated engagement statistics are deliberately excluded.
- Pairwise BPR training uses a deterministic 300,000-example reservoir from 1,331,671 eligible
  positive events. Random negatives obey item upload time; exposure negatives come from the
  user's recent shown-but-not-long-view items.
- Item and query encoding runs in a PyTorch-only process; FAISS indexing runs in a separate
  process to avoid duplicate OpenMP runtimes on macOS.

## Single-task ranker

The pointwise LightGBM model predicts `long_view` from 92 features. It excludes every action and
watch-time field from the current impression: only strictly prior user/item aggregates, request
context, and upload-time-safe content metadata are eligible. The first 90% of standard-train by
time supplies a deterministic one-million-row training sample; the final 10% supplies a separate
250,000-row early-stopping sample. Standard-validation remains untouched until final evaluation.

| Full standard-validation metric | Historical item-rate | LightGBM ranker | Absolute change |
|---|---:|---:|---:|
| ROC AUC | 0.6343 | **0.8125** | +0.1782 |
| Log loss | 0.6019 | **0.4470** | -0.1549 |
| Brier score | 0.1970 | **0.1467** | -0.0503 |
| User-day NDCG@10 | 0.5433 | **0.6058** | +0.0625 |

The evaluation covers all 2,920,213 validation impressions and 7,098 user-days containing at
least one positive. Training early-stops at iteration 113 with internal-validation AUC 0.8143;
batched CPU prediction takes 17.8 seconds (about 164k rows/second). Machine-readable metrics and gain-based feature importance are in
`results/ranker_standard_validation.json`.

## Multi-task ranking with MMoE

```text
categorical embeddings + standardized numeric features
                  -> four shared experts
        -> click gate     -> click tower
        -> long-view gate -> long-view tower
        -> like gate      -> like tower
```

The neural comparison uses the same leakage-safe feature contract, a fixed 600,000-row training
sample, a later 150,000-row internal time holdout, and the full 2,920,213-row standard validation
set. All models use comparable 4.24M--4.27M parameter budgets and are repeated with seeds 2026,
2027, and 2028. `long_view` is always the primary task; the multi-task models additionally learn
`is_click` and `is_like` with loss weights 0.3 and 0.1.

| Ranker | Tasks | Parameters | Long-view AUC | Long-view NDCG@10 |
|---|---:|---:|---:|---:|
| Single-task MLP | 1 | 4.241M | **0.8049 ± 0.0002** | 0.6007 ± 0.0038 |
| Shared-bottom | 3 | 4.247M | 0.8048 ± 0.0001 | 0.6014 ± 0.0008 |
| MMoE, 4 experts | 3 | 4.267M | 0.8046 ± 0.0006 | **0.6027 ± 0.0017** |
| LightGBM reference | 1 | n/a | **0.8125** | **0.6058** |

MMoE improves mean NDCG@10 by 0.0020 over the neural single-task baseline, but the paired change
has standard deviation 0.0050 and changes sign across seeds. It is therefore not treated as a
robust gain. It does recover some ranking quality relative to a naive shared-bottom model, while
the original LightGBM remains the strongest long-view ranker.

The auxiliary MMoE heads reach click AUC 0.7891 ± 0.0003 and like AUC 0.9239 ± 0.0005. Gate
diagnostics show task differentiation rather than identical routing: within every seed, the rare
like task selects a different dominant expert from click and long-view. Full per-seed metrics,
training curves, parameter counts, and gate distributions are stored in
`results/multitask_comparison.json`.

## Randomized evaluation and exposure debias

Random-exposure rows remain isolated from ordinary training. Their 43,028 rows are assigned to
calibration (5,689), validation (6,343), and test (30,996) by time. For evaluation, each random
row receives the same 92-feature PIT contract using only strictly earlier standard or randomized
events. The random-calibration period is the only randomized data allowed to fit debiasing or
adaptation parameters.

| Model | Random-validation AUC | NDCG@10 | Log loss | Random-test AUC | NDCG@10 | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Historical item-rate | 0.6018 | **0.6816** | **0.2964** | 0.5972 | **0.6622** | **0.3052** |
| Standard-trained ranker | **0.7576** | 0.6743 | 0.4719 | **0.7348** | 0.6551 | 0.4855 |
| Density-ratio weighted | 0.7422 | 0.6775 | 0.4616 | 0.7110 | 0.6353 | 0.4825 |
| Random-calibration adapted | 0.7258 | 0.6648 | 0.3758 | 0.7180 | 0.6565 | 0.4046 |

The observational ranker transfers useful global discrimination to random traffic, but its
standard-validation AUC of 0.8125 falls to 0.7576 and its probabilities are badly miscalibrated
because prevalence falls from 26.29% to 8.18%. Ranking within user-day also differs from global
AUC: the learned rankers do not beat the simple item-rate baseline on randomized NDCG@10.

The IPS-style experiment estimates a classifier density ratio from balanced standard and random
calibration exposures, clips ratios to [0.1, 10], and self-normalizes them before weighted
training. The domain classifier reaches AUC 0.9967; item age and recent item exposure counts are
the strongest separators. About 82% of standard training rows hit the lower clip and effective
sample size is only 60,748 out of one million (6.1%), so the random policy shows items the logged
data barely covers and reweighting can't make up for that. Validation NDCG goes up by 0.0032 but
test NDCG drops by 0.0198.

A second baseline mixes only the early 80% of randomized
calibration labels into standard training and selects its mixing mass on the remaining 20%. It
improves probability calibration but not randomized ranking, separating calibration gains from
recommendation-quality gains. Full diagnostics are in `results/randomized_evaluation.json`,
`results/debias_evaluation.json`, and `results/randomized_adaptation.json`.

## Online serving and latency

```text
request
  -> point-in-time user snapshot (Redis or memory-mapped NumPy)
  -> precomputed user vector
  -> FAISS IVF-Flat Top-100 (two isolated retrieval workers)
  -> assemble the same 92-feature contract used offline
  -> LightGBM ranker worker
  -> FastAPI Top-10 response
```

The snapshot is fixed at `1651246340219` and contains 993 populated users, 3,135,874 populated
items, recent positive histories, historical aggregates, and 1,000 precomputed 64-dimensional
user vectors. User/item arrays are memory-mapped instead of copied into the API process. FAISS
and LightGBM run in separate spawned processes; this both isolates the native runtimes and lets
retrieval scale across workers.

The CPU-only benchmark sends three trials of 500 successful requests after 100 warm-up requests,
at concurrency 8. Each request retrieves 100 candidates with `nprobe=1024` and returns a ranked
Top-10. It ran on an Apple M4 (10 CPU cores, 16 GB RAM), macOS 26.6.2, Python 3.9.6.

| Feature store / retrieval workers | Throughput | Client p50 | Client p95 | Client p99 |
|---|---:|---:|---:|---:|
| Redis / 1 | 169.0 QPS | 46.9 ms | 48.6 ms | 54.9 ms |
| Redis / 2 | **215.5 QPS** | **35.8 ms** | 47.2 ms | 60.5 ms |
| Memory mmap / 2 | **222.2 QPS** | **35.4 ms** | **39.5 ms** | **53.5 ms** |

Two retrieval workers improve Redis-backed throughput by 27.5% and reduce median round-trip
latency by 11.1 ms, at the cost of loading one approximately 875 MB FAISS index per worker. With
the worker count held at two, Redis adds about 1 ms at feature-fetch p95 and reduces throughput by
about 3% relative to local memory maps. The Redis two-worker server-side p95 breakdown is:

| Stage | p95 |
|---|---:|
| Redis feature fetch | 1.06 ms |
| FAISS computation | 9.44 ms |
| Retrieval queue + IPC | 21.76 ms |
| Ranking feature assembly | 5.50 ms |
| LightGBM inference | 1.70 ms |
| Ranking queue + IPC | 25.70 ms |
| Total server time | 43.72 ms |

Per-stage percentiles are computed independently and therefore are not additive. Queueing, rather
than model inference, is the main remaining tail-latency bottleneck.

These are local load-test numbers on the M4 machine above. Full distributions
and environment metadata are saved in `results/serving_benchmark_redis_w1.json`,
`results/serving_benchmark_redis_w2.json`, and `results/serving_benchmark_memory_mmap_w2.json`.
The serving image was also built and smoke-tested through Docker Compose against the mounted
artifacts and Redis service.

Serving uses LightGBM because the MMoE experiment did not beat it on the
primary ranking metric.

## Data protocol

- Standard traffic is split chronologically into train, validation, and test using `time_ms`.
- Random traffic is kept separate and split into calibration, validation, and final test periods.
- Historical features use events with timestamps **strictly earlier** than the prediction event.
- Events sharing the same timestamp cannot leak labels into one another.
- KuaiRand's month-aggregated `video_features_statistic` file is excluded from the main feature
  set because it is not a point-in-time snapshot.
- Zero-duration rows are retained but receive a neutral historical watch-ratio contribution. The
  validator reports them explicitly instead of silently dividing by zero or deleting exposure
  events.

## Local setup

Python 3.9+ is supported.

```bash
make setup
make test
make lint
```

Download KuaiRand-1K from the
[official KuaiRand repository](https://github.com/chongminggao/KuaiRand). The helper below uses
the official Zenodo artifact, supports resuming, and verifies the published MD5 checksum. The
archive is about 1.1 GB and expands to about 4.3 GB.

```bash
make download-data
```

The extracted files live under `data/raw/KuaiRand-1K/data/`. Large data and generated artifacts
are ignored by Git.

Once the files are present:

```bash
make validate-data
make split-data
make build-features
make build-random-features
make eval-retrieval
make prepare-item-features
make build-training-examples
make train-two-tower
make eval-two-tower
make train-ranker
make train-multitask-rankers
make eval-randomized
make train-debiased-ranker
make train-random-adaptation
make prepare-serving-snapshot
```

`make eval-two-tower` first materializes embeddings and then launches FAISS evaluation in a new
process. The large checkpoints, embeddings, and indices stay under `artifacts/` and are ignored by
Git; compact metrics under `results/` are versioned.

Start the service with either local memory-mapped features or Redis, then run the load test:

```bash
# memory-mapped feature store
make serve

# or Redis-backed feature store
docker compose up -d redis
make serve-redis

# in another terminal
make benchmark
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`. A fully containerized
run uses `docker compose up --build api` after the serving snapshot has been prepared.

## Reproducibility rules

- No result is added to this README without a saved configuration and output artifact.
- The Two-Tower table is the deterministic seed-2026 model-selection run; the MMoE comparison
  reports mean and standard deviation over three controlled seeds.
- Randomized evaluation reports support counts and user-level bootstrap confidence intervals.
- FAISS is compared with exact retrieval; it is an efficiency optimization, not a model gain.
