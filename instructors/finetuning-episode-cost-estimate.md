---
title: 'Fine-tuning episode: cost and capacity planning'
---

Planning notes for running the *Fine-tuning a Vision-Language Model with Training Jobs* episode (Badger Scribe, CHURRO-3B LoRA) with a class of about 50. Prices are on-demand SageMaker rates for `us-east-1` from the AWS price list published 2026-10-16; re-check them before the term, since `ml.g5.xlarge` has already moved from $1.21 to $1.41 since the RAG episodes were written.

## What one student spends

| Item | Instance | Billable time | Cost |
|---|---|---|---|
| Controller notebook (CPU) | `ml.t3.medium` ($0.05/h) | 4 h | $0.20 |
| Smoke test (20 pages, 1 epoch) | `ml.g5.xlarge` ($1.41/h) | ~12 min | $0.28 |
| Full LoRA fine-tune (526 train pages × 2 epochs, 20-page eval) | `ml.g5.xlarge` | ~65 min | $1.53 |
| Transcribe holdout + test with base model (~300 pages, Processing job) | `ml.g5.xlarge` | ~25 min | $0.59 |
| Transcribe holdout + test with adapter | `ml.g5.xlarge` | ~25 min | $0.59 |
| S3: adapter, predictions, logs | | | < $0.05 |
| **One clean pass** | | **~2.2 GPU-hours** | **~$3.20** |

Assumptions behind the training line: CHURRO-3B in bf16 with LoRA rank 16, gradient checkpointing, one page per step, `max_pixels` of 1280 × 28 × 28 (about one megapixel, ~1,300 visual tokens plus ~300 text tokens per page), about 2.5 s per page-pass on an A10G, plus ~10 minutes of fixed startup (container pull, 2 GB of images, 7 GB of weights). The training line is the least certain one; a 2x miss in either direction would not be surprising until a real run has been timed. Everything in the episode is capped (`max_run` of 2 h on the fine-tune, 1 h on each transcription job), so the worst case per student from the scripted steps is about $7.

## What the class spends

| Scenario | Per student | 50 students |
|---|---|---|
| Everyone does exactly one clean pass | $3.20 | $160 |
| Realistic: two full fine-tunes, a few extra smoke tests, one extra transcription job | ~$6.50 | ~$325 |
| Everything hits its `max_run` cap | ~$7 | ~$350 |
| **Budget to request** (realistic + 50%) | | **$500** |

For comparison, teaching the same material on `ml.g5.xlarge` *notebook* instances costs about the same during class ($1.41 × 3 h × 50 = $212) but adds the forgotten-instance risk: each one left running costs $34/day, $237/week, $1,029/month. Two forgotten instances over a long weekend exceed the entire jobs-based budget above. That difference is the point of the episode, and it is worth saying out loud to the class.

## Things that will go wrong unless arranged ahead of time

**Service quotas.** This is the item most likely to sink the session. The relevant quotas in Service Quotas → Amazon SageMaker are, per region:

- `ml.g5.xlarge for training job usage`: needs to be at least the number of students who will launch at once. Request **60**.
- `ml.g5.xlarge for processing job usage`: students launch two transcription jobs each, in parallel. Request **100**, or tell students to run them sequentially and request 60.
- `Maximum number of instances across all training jobs` and the processing equivalent: check they are not lower than the above.
- If you want `ml.g6e.xlarge` for Exercise 1, request 10–20 of those too.

New accounts often have these at 0 to 4. Quota increases for GPU instances can take days and sometimes a back-and-forth with support, so file them **three to four weeks** before class.

**Capacity.** Fifty `ml.g5.xlarge` launches within the same minute in one region can fail with `InsufficientCapacity` even when the quota allows it. Mitigations, in order: launch the smoke tests in waves (table by table), accept `ml.g6.xlarge` (L4, 24 GB, $1.13/h; roughly the same speed on this workload) as a documented fallback instance type, and run the two transcription jobs sequentially if processing capacity is short.

**Data staging.** The competition data is 1.95 GB (837 JPEGs). Put one copy in a shared bucket with the layout the episode expects (`badger-scribe/{train,test,metadata}.csv`, `metric.py`, `images/`), grant the students' execution role `s3:GetObject`/`s3:ListBucket` on it, and nobody downloads from Kaggle in class. Each training job then copies the prefix into its container; S3 to SageMaker transfer within a region is free.

**Model download.** Each job pulls `stanford-oval/churro-3B` (~7.5 GB) from Hugging Face. Fifty parallel downloads are fine, but confirm the training instances have internet access (default outside a VPC) and that the model is not gated for the account. If Hugging Face is a problem, mirror the weights to the shared bucket and point `--model_id` at a local copy via an extra input channel. The CHURRO weights carry the Qwen research license (non-commercial), which is fine for a class.

**Warm pools.** `keep_alive_period_in_seconds` on the estimator keeps the instance for the next job, which would let the smoke test and the full run share one startup. It needs its own quota (`ml.g5.xlarge for training warm pool usage`) and it bills for the idle time at the full rate, so for a class it adds a way to leave meters running. Leave it off.

**Spot.** Managed spot training saves up to 70% but 50 simultaneous spot requests will see interruptions and capacity waits, and the script does not checkpoint. Keep it as the exercise it is.

## Before class

- Run the full episode once yourself and replace the time estimates above with measured `BillableTimeInSeconds`. The episode tells students to compare their bill to the estimate; the estimate should be a real number.
- Set a Budgets alert on the class account at the planned total, and a Cost Explorer filter on the `Project` tag students use in `make_job_tags`.
- Have the fallback instance type in the slide: `instance_type = "ml.g6.xlarge"`.
- Tell students to stop the controller notebook at the end. It is five cents an hour, which is why nobody notices it.
