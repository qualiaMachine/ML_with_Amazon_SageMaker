---
title: "Fine-tuning a Vision-Language Model with Training Jobs"
teaching: 40
exercises: 25
---

:::::::::::::::::::::::::::::::::::::: questions

- When is fine-tuning worth it, and how is it different from RAG or prompting?
- How do I estimate what a fine-tuning run will cost *before* I launch it?
- How do I run a GPU fine-tune as a SageMaker training job so the meter stops the moment it finishes?
- How do I measure whether the fine-tune actually helped?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain why fine-tuning is the most common place for cloud bills to go wrong, and the three habits that prevent it: a time cap, a smoke test, and a cost estimate written down before launch.
- Stage an image dataset and its labels on S3 for a training job.
- Launch a LoRA fine-tune of a 3B vision-language model (CHURRO-3B) as a `HuggingFace` training job on a single 24 GB GPU, with tags and a `max_run` cap.
- Run the base and fine-tuned models over a holdout set with a Processing job and compare their character error rate (CER) with the competition's own metric.
- Produce a submission file for the Badger Scribe Kaggle competition.

::::::::::::::::::::::::::::::::::::::::::::::::

## Why a fine-tuning episode

Everything so far in the RAG episodes left the model's weights alone. Fine-tuning changes them, and that puts it in a different class of workload:

- It needs a GPU for the whole run, not just for inference calls.
- It is iterative. Nobody gets the learning rate, the image resolution or the data split right the first time, so the first run is never the last.
- The cost of a run depends on things that are easy to get wrong by 10x: epochs, image resolution, sequence length, instance type, and whether you remembered to turn the instance off.

That combination is why fine-tuning is where research teams see the surprise bills. It is also, done carefully, the single biggest lever you have: the Badger Scribe organizers report that LoRA on ~100 pages takes a small open model from 0.23 to 0.06 CER on survey notebooks in about a GPU-hour. One GPU-hour. The point of this episode is to make that hour cost two dollars instead of two hundred.

### The task: Badger Scribe

[Badger Scribe](https://www.kaggle.com/competitions/badger-scribe) is a Kaggle challenge built from UW Digital Collections material: 19th-century surveyors' field notebooks, craftsmen's account books, and German immigrant letters in Kurrent script. Given a page image, produce a verbatim transcription. The score is character error rate (CER), macro-averaged over the three document categories, so a pipeline has to read German Kurrent *and* pencil survey notes to do well.

The released train set is small:

| category | train pages | label source |
|---|---|---|
| `kade_letters` (German Kurrent) | 295 | all human-transcribed |
| `survey_notes` (pencil field notes) | 288 | 16 human, 272 machine draft |
| `dominy_accounts` (account books, letters) | 36 | 2 human, 34 machine draft |
| **total** | **619** | ~709 characters per page on average |

Small is the point. 619 labelled pages is a realistic amount of data for a library or a lab to have, and it is enough to move a model that was pre-trained on the right kind of material.

We will fine-tune [CHURRO-3B](https://github.com/stanford-oval/Churro), a 3-billion-parameter open-weight vision-language model from Stanford that was itself fine-tuned from Qwen2.5-VL-3B on roughly 100,000 pages of historical documents in 46 language clusters. It already reads old handwriting better than its base model does; we are teaching it *these* hands and *these* transcription conventions.

::::::::::::::::::::::::::::::::::::: callout

### Fine-tune, RAG, or prompt?

Ask what the model is missing. If it lacks **facts** (your documents, your data), retrieval gives it those at inference time and no training is needed. If it lacks a **skill or a style** (reading Kurrent, following your lab's labelling conventions, producing output in a strict format), you need to change the weights, and that is fine-tuning. Badger Scribe is squarely the second case: no amount of context will teach a model what a Kurrent "e" looks like.

Full fine-tuning updates all 3 billion weights and needs several times the model's size in GPU memory for optimizer state. **LoRA** (low-rank adaptation) freezes the model and trains a few million extra parameters in small matrices bolted onto the attention and MLP layers. The adapter is tens of megabytes, trains on a single 24 GB GPU, and recovers most of the gain of a full fine-tune for a task like this. We use LoRA.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 0: Estimate the cost before you launch anything

This is the habit that matters. Before launching a fine-tune, write down a number for what it should cost. If the real bill is 3x your number, something is wrong and you want to know on the first run, not the fiftieth.

### The instance

The models we fine-tune here fit on one 24 GB GPU. Current on-demand prices for SageMaker training jobs in `us-east-1` (AWS price list, October 2026):

| Instance | GPU | GPU memory | $/hour (training job) |
|---|---|---|---|
| `ml.g4dn.xlarge` | 1 × T4 | 16 GB | $0.74 |
| `ml.g6.xlarge` | 1 × L4 | 24 GB | $1.13 |
| `ml.g5.xlarge` | 1 × A10G | 24 GB | $1.41 |
| `ml.g6e.xlarge` | 1 × L40S | 48 GB | $2.61 |
| `ml.g5.12xlarge` | 4 × A10G | 96 GB | $7.09 |
| `ml.p4d.24xlarge` | 8 × A100 | 320 GB | $25.25 |

Notice the shape of that table. Going from one GPU to four or eight multiplies the price by 5 to 18, and for a 619-page LoRA run it buys you nothing: the job is too small to split. Picking `ml.p4d.24xlarge` "because it is faster" is the first classic mistake. A T4 is too small and too slow for a 3B vision model with page-sized images; `ml.g5.xlarge` is the workhorse for this episode. (`ml.g6.xlarge` is a fine substitute if g5 capacity is short; `ml.g6e.xlarge` is roughly 2–3x faster for less than 2x the price, so it is often *cheaper per run*; see the exercise at the end.)

### The run

A training job bills from the moment its instance is ready until the job ends, including the minutes spent pulling the container, downloading data and downloading the model. Our estimate for one full run on `ml.g5.xlarge`:

| Phase | Time | Cost |
|---|---|---|
| Startup: container pull, 2 GB of images, 7 GB of model weights | ~10 min | $0.24 |
| Training: 619 pages × 2 epochs, ~2.5 s per page with gradient checkpointing | ~50 min | $1.17 |
| Scoring 20 holdout pages inside the job | ~4 min | $0.09 |
| **One full run** | **~65 min** | **~$1.50** |

The training time is the uncertain line. It scales linearly with pages × epochs, and roughly linearly with image resolution (every 28 × 28 pixel block of the page is one token in the model's sequence). Doubling `max_pixels` doubles the time; doubling epochs doubles it again. Those two knobs, left at careless values, are the second classic mistake, and they are why the estimate must be written down before the first run and compared to the real `BillableTimeInSeconds` after it.

We set `max_run` to twice the estimate. If a run goes past that, SageMaker kills it and we lose at most $3, not a weekend.

### The whole episode

| What | Instance | Time | Cost |
|---|---|---|---|
| Controller notebook (no GPU needed) | `ml.t3.medium` | 4 h | $0.20 |
| Smoke test: 20 pages, 1 epoch | `ml.g5.xlarge` | ~12 min | $0.30 |
| Full fine-tune | `ml.g5.xlarge` | ~65 min | $1.50 |
| Base model over holdout + test pages (Processing job) | `ml.g5.xlarge` | ~25 min | $0.60 |
| Fine-tuned model over holdout + test pages | `ml.g5.xlarge` | ~25 min | $0.60 |
| **Total for one clean pass** | | | **~$3.20** |

Compare the third classic mistake: doing all of this interactively on a `ml.g5.xlarge` *notebook* instance. The class itself costs about the same ($1.41 × 3 hours), but a notebook instance keeps billing after you close the tab. Forgotten over a weekend: $90. Forgotten for a month: $1,029. A training job cannot be forgotten, because it ends.

::::::::::::::::::::::::::::::::::::: callout

### The three habits

1. **Cap it.** Every job gets `max_run`. Every notebook GPU instance gets stopped when you stand up.
2. **Smoke-test it.** The first launch of any script uses a handful of pages and one epoch. It costs 30 cents and catches the bug that would otherwise cost $3 per attempt, or $30 on a big instance.
3. **Estimate it, then check.** Write the number down. After the run, read `BillableTimeInSeconds` from the job description and compare.

::::::::::::::::::::::::::::::::::::::::::::::::

## Setup

This episode runs from a **CPU** notebook instance (`ml.t3.medium` or `ml.m5.xlarge`). The notebook is only a controller: it stages data, launches jobs, and scores results. The GPUs live in the jobs.

Your instructor has staged the competition data (from the Kaggle data tab) in a shared read-only bucket with this layout:

- `s3://<shared-bucket>/badger-scribe/train.csv`, `test.csv`, `metadata.csv`, `metric.py`, `sample_submission.csv`
- `s3://<shared-bucket>/badger-scribe/images/<page_id>.jpg`: train and test pages together, about 2 GB

If you are working on your own, download the dataset from the [competition data page](https://www.kaggle.com/competitions/badger-scribe/data) and upload it to your own bucket with the same layout.

```python
import os
import json
import time

import boto3
import pandas as pd
import sagemaker
from sagemaker import get_execution_role

session = sagemaker.Session()
region = session.boto_region_name
role = get_execution_role()
s3_client = boto3.client("s3", region_name=region)

shared_bucket = "ml-marathon-shared"   # <-- EDIT ME: read-only bucket with the competition data
bucket_name = "chris-rag-2"            # <-- EDIT ME: your own bucket for outputs

data_prefix = "badger-scribe"
train_s3_uri = f"s3://{shared_bucket}/{data_prefix}/"           # training channel: csvs + images/
images_s3_uri = f"s3://{shared_bucket}/{data_prefix}/images/"

local_data_dir = "./data"
os.makedirs(local_data_dir, exist_ok=True)

print("Region:", region)
print("Role:", role)
```

Download the small files (not the images) so we can look at the data and score predictions locally.

```python
for fname in ["train.csv", "test.csv", "metadata.csv", "metric.py"]:
    s3_client.download_file(shared_bucket, f"{data_prefix}/{fname}", os.path.join(local_data_dir, fname))

train_df = pd.read_csv(os.path.join(local_data_dir, "train.csv"), keep_default_na=False)
test_df = pd.read_csv(os.path.join(local_data_dir, "test.csv"), keep_default_na=False)
metadata_df = pd.read_csv(os.path.join(local_data_dir, "metadata.csv"), keep_default_na=False)

print("train pages:", len(train_df), " test pages:", len(test_df))
print(train_df.groupby(["category", "label_source"]).size())
```

Look at one page and its transcription. Note what "verbatim" means here: the writer's spelling, the abbreviations, the `#` for a character the transcriber could not read.

```python
row = train_df[train_df.category == "survey_notes"].iloc[3]
print(row.page_id, "|", row.category, "|", row.label_source)
print(row.text[:600])
```

```python
from IPython.display import Image, display

page_path = os.path.join(local_data_dir, f"{row.page_id}.jpg")
s3_client.download_file(shared_bucket, f"{data_prefix}/images/{row.page_id}.jpg", page_path)
display(Image(filename=page_path, width=500))
```

### The scripts

Jobs run in containers, so the code they execute has to be shipped with the job. Clone the lesson repository and copy three scripts into a `code/` directory:

```python
!git clone https://github.com/carpentries-incubator/ML_with_AWS_SageMaker.git
```

```python
import shutil

os.makedirs("code", exist_ok=True)
for f in ["badger_common.py", "finetune_churro_lora.py", "churro_transcribe_batch.py"]:
    shutil.copy(f"ML_with_AWS_SageMaker/scripts/{f}", f"code/{f}")

with open("code/requirements.txt", "w") as f:
    f.write("\n".join(["peft>=0.14", "accelerate", "pillow", "pandas"]))

print(os.listdir("code"))
```

- `badger_common.py`: the prompt, image loading, and a copy of the competition's CER. Shared by the other two so the model is trained on exactly the prompt it is later asked.
- `finetune_churro_lora.py`: the training job entry point. Holds out 15% of pages per category, trains a LoRA adapter, saves it, and scores 20 holdout pages.
- `churro_transcribe_batch.py`: the Processing job entry point. Transcribes a list of pages with the base model, or with the adapter merged in, and writes `predictions.csv` in the submission format.

Open `code/finetune_churro_lora.py` and read it before launching it. The pieces worth understanding:

```python
# Only the language model's projection layers get LoRA weights.
# The vision tower ("visual") stays frozen.
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
)

# The loss is computed on the transcription only. Prompt and image tokens
# are masked to -100 so the model is not graded on repeating the question.
labels[i, :n_prompt] = -100
```

And the one setting that controls cost more than any other:

```python
# Every 28x28 block of the page becomes one token. 1280 blocks ~ 1 megapixel.
DEFAULT_MAX_PIXELS = 1280 * 28 * 28
```

### Tags

As in every episode: tags do not propagate from the notebook to the jobs it launches. Pass them to every estimator and processor.

```python
name = "John Doe"        # replace with your name
project = "sinkorswim"   # replace with your team name

def make_job_tags(purpose: str):
    return [
        {"Key": "Name", "Value": name},
        {"Key": "Project", "Value": project},
        {"Key": "Purpose", "Value": purpose},
    ]
```

## Step 1: Smoke test (20 pages, 1 epoch)

Never launch a full run of a script that has never run. The smoke test uses the same instance, container and data as the real run, with `max_train_pages=20` and `epochs=1`, so that every failure mode (a missing package, a wrong path, an out-of-memory error) shows up for 30 cents.

```python
from sagemaker.huggingface import HuggingFace

instance_type = "ml.g5.xlarge"
model_output_path = f"s3://{bucket_name}/{data_prefix}/models/"

smoke_estimator = HuggingFace(
    entry_point="finetune_churro_lora.py",
    source_dir="code/",
    base_job_name="badger-churro-smoke",
    role=role,
    tags=make_job_tags("finetune-smoke-test"),
    instance_type=instance_type,
    instance_count=1,
    volume_size=50,                       # GB of disk for the container, weights and images
    max_run=30 * 60,                      # 30 minutes; the real run gets 2 hours
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    output_path=model_output_path,
    sagemaker_session=session,
    hyperparameters={
        "model_id": "stanford-oval/churro-3B",
        "max_train_pages": 20,
        "epochs": 1,
        "eval_pages": 5,
    },
)

smoke_estimator.fit({"train": train_s3_uri})
```

While it runs, open the SageMaker console: **SageMaker AI → Training → Training jobs**. You can watch the status move through `Starting`, `Downloading`, `Training`, `Uploading`, `Completed`, and open the CloudWatch log from the job page. The log is where `print()` output from the script goes.

### Read the bill for this job

When it finishes, ask SageMaker what it charged you for. This is the number to compare against your estimate every time.

```python
def job_cost_report(job_name: str, price_per_hour: float):
    desc = session.sagemaker_client.describe_training_job(TrainingJobName=job_name)
    billable = desc["BillableTimeInSeconds"]
    training = desc.get("TrainingTimeInSeconds", 0)
    print(f"job:            {job_name}")
    print(f"status:         {desc['TrainingJobStatus']}")
    print(f"training time:  {training / 60:.1f} min  (script running)")
    print(f"billable time:  {billable / 60:.1f} min  (what you pay for, incl. download)")
    print(f"cost:           ${billable / 3600 * price_per_hour:.2f} at ${price_per_hour}/h")
    return desc

g5_price = 1.41
smoke_desc = job_cost_report(smoke_estimator.latest_training_job.name, g5_price)
```

The gap between *training time* and *billable time* is the fixed overhead of every job: container pull, data download, model download. It is the same whether you train for one minute or one hour, which is why a smoke test is cheap and why batching work into fewer, longer jobs is cheaper than many tiny ones.

::::::::::::::::::::::::::::::::::::: challenge

### What did the smoke test tell you?

Open the CloudWatch log for the smoke-test job and find:

1. The line `trainable params: ...` from PEFT. What fraction of the 3B parameters did we actually train?
2. The `[eval]` line. The base model's CER on 5 pages after 20 training pages tells you very little about quality, but what does the time per page tell you about the cost of the inference jobs later?
3. The `training finished in X min` line. With 20 pages × 1 epoch, scale that to 526 pages × 2 epochs. Does it match the ~50-minute estimate?

:::::::::::::::::::::::: solution

1. Around 1% (LoRA rank 16 on the attention and MLP projections of a 3B model is on the order of 30 M parameters).
2. Roughly 8–12 seconds per page for greedy generation one page at a time on an A10G. The batched Processing job below is faster per page, but 300 pages is still 15–25 minutes of GPU time.
3. Expect 2–3 seconds per page at 1 megapixel. If you measured 2.5 s/page, 1052 page-passes is about 44 minutes. If you are far above that, check whether `max_pixels` was left at a larger value.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 2: The full fine-tune

Same estimator, three changes: all pages, two epochs, and a two-hour cap (twice the estimate).

```python
estimator = HuggingFace(
    entry_point="finetune_churro_lora.py",
    source_dir="code/",
    base_job_name="badger-churro-lora",
    role=role,
    tags=make_job_tags("finetune-churro-lora"),
    instance_type=instance_type,
    instance_count=1,
    volume_size=50,
    max_run=2 * 60 * 60,                  # 2 hours: twice the estimate, never "whatever it takes"
    transformers_version="4.56",
    pytorch_version="2.8",
    py_version="py312",
    output_path=model_output_path,
    sagemaker_session=session,
    hyperparameters={
        "model_id": "stanford-oval/churro-3B",
        "epochs": 2,
        "lr": 2e-4,
        "lora_r": 16,
        "batch_size": 1,
        "grad_accum": 8,                  # effective batch of 8 pages
        "max_pixels": 1280 * 28 * 28,     # ~1 megapixel per page
        "holdout_frac": 0.15,
        "eval_pages": 20,
    },
)

estimator.fit({"train": train_s3_uri}, wait=False)   # return immediately; we will poll
training_job_name = estimator.latest_training_job.name
print("launched:", training_job_name)
```

We launched with `wait=False` so the notebook is free. You can keep working, or poll until it is done:

```python
while True:
    status = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)["TrainingJobStatus"]
    print(time.strftime("%H:%M:%S"), status)
    if status in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(120)
```

::::::::::::::::::::::::::::::::::::: callout

### Stopping a job you regret

A running training job can be stopped from the console (**Training jobs → select → Stop**) or from code:

```python
# session.sagemaker_client.stop_training_job(TrainingJobName=training_job_name)
```

Billing stops within a minute or two. If you launched with the wrong instance, the wrong data or the wrong number of epochs, stop it immediately; the few cents already spent are not a reason to let it run.

::::::::::::::::::::::::::::::::::::::::::::::::

When it completes, read the bill and the training summary.

```python
desc = job_cost_report(training_job_name, g5_price)
adapter_s3_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
print("adapter:", adapter_s3_uri)
```

```python
import tarfile

local_adapter_tar = os.path.join(local_data_dir, "model.tar.gz")
bucket, key = adapter_s3_uri.replace("s3://", "").split("/", 1)
s3_client.download_file(bucket, key, local_adapter_tar)

with tarfile.open(local_adapter_tar) as tar:
    tar.extract("train_summary.json", path=local_data_dir)
    tar.extract("holdout_page_ids.csv", path=local_data_dir)

print(json.dumps(json.load(open(os.path.join(local_data_dir, "train_summary.json"))), indent=2))
```

The whole output of an hour of GPU time is a `model.tar.gz` of a few tens of megabytes: the LoRA adapter, the processor configuration, the list of held-out pages, and this summary. The 3B base model is never copied anywhere.

## Step 3: Base vs. fine-tuned on the holdout (Processing jobs)

The `[eval]` line inside the training job scores 20 pages with the fine-tuned model. To know what the fine-tune *bought*, we need the base model's score on the same pages, and ideally more of them. We run the same transcription script twice as Processing jobs, once with and once without the adapter, over the full holdout plus the test pages, so each job also produces a submission file.

First, build the list of pages and upload it with the metadata and the holdout list.

```python
holdout_df = pd.read_csv(os.path.join(local_data_dir, "holdout_page_ids.csv"))

pages_df = pd.concat([
    train_df[train_df.page_id.isin(holdout_df.page_id)][["page_id", "doc_id", "category", "text"]],
    test_df[["page_id", "doc_id", "category"]].assign(text=""),
], ignore_index=True)
print(pages_df.category.value_counts(), "\n", len(pages_df), "pages to transcribe")

infer_prefix = f"{data_prefix}/inference"
pages_df.to_csv(os.path.join(local_data_dir, "pages.csv"), index=False)
for fname in ["pages.csv", "metadata.csv"]:
    s3_client.upload_file(os.path.join(local_data_dir, fname), bucket_name, f"{infer_prefix}/{fname}")

infer_data_s3_uri = f"s3://{bucket_name}/{infer_prefix}/"
```

Then a function that launches one transcription job. Three inputs are mounted into the container: the images (from the shared bucket), our CSVs, and, optionally, the adapter.

```python
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

def run_transcription_job(label: str, adapter_uri: str = None, wait: bool = True):
    output_uri = f"s3://{bucket_name}/{data_prefix}/predictions/{label}/"
    inputs = [
        ProcessingInput(source=images_s3_uri, destination="/opt/ml/processing/input/images"),
        ProcessingInput(source=infer_data_s3_uri, destination="/opt/ml/processing/input/data"),
    ]
    arguments = [
        "--model_id", "stanford-oval/churro-3B",
        "--pages_csv", "pages.csv",
        "--batch_size", "4",
        "--max_pixels", str(1280 * 28 * 28),
    ]
    if adapter_uri:
        inputs.append(ProcessingInput(source=adapter_uri, destination="/opt/ml/processing/input/adapter"))
        arguments += ["--adapter_tar", "/opt/ml/processing/input/adapter/model.tar.gz"]

    processor = HuggingFaceProcessor(
        base_job_name=f"badger-transcribe-{label}",
        role=role,
        tags=make_job_tags(f"transcribe-{label}"),
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=50,
        transformers_version="4.56",
        pytorch_version="2.8",
        py_version="py312",
        sagemaker_session=session,
        max_runtime_in_seconds=60 * 60,   # 1 hour cap; the estimate is 25 minutes
    )
    processor.run(
        code="churro_transcribe_batch.py",
        source_dir="code/",
        inputs=inputs,
        outputs=[ProcessingOutput(output_name="predictions", source="/opt/ml/processing/output", destination=output_uri)],
        arguments=arguments,
        wait=wait,
        logs=wait,
    )
    return output_uri

base_output_uri = run_transcription_job("base", adapter_uri=None, wait=False)
tuned_output_uri = run_transcription_job("tuned", adapter_uri=adapter_s3_uri, wait=False)
print(base_output_uri, "\n", tuned_output_uri)
```

Both jobs run in parallel on their own GPU, which doubles the instantaneous spend but halves the wall-clock time; the total cost is the same. Watch them under **SageMaker AI → Data preparation → Processing jobs**.

```python
def wait_for_processing_jobs(name_prefixes):
    while True:
        jobs = session.sagemaker_client.list_processing_jobs(SortBy="CreationTime", SortOrder="Descending", MaxResults=20)["ProcessingJobSummaries"]
        mine = [j for j in jobs if any(j["ProcessingJobName"].startswith(p) for p in name_prefixes)]
        print(time.strftime("%H:%M:%S"), {j["ProcessingJobName"]: j["ProcessingJobStatus"] for j in mine})
        if all(j["ProcessingJobStatus"] in ("Completed", "Failed", "Stopped") for j in mine):
            return mine
        time.sleep(120)

wait_for_processing_jobs(["badger-transcribe-base", "badger-transcribe-tuned"])
```

### Score with the competition's metric

Download both prediction files and score the holdout rows with `metric.py`, the exact file that scores the leaderboard.

```python
import subprocess

def fetch_predictions(output_uri: str, label: str) -> pd.DataFrame:
    bucket, key = output_uri.replace("s3://", "").split("/", 1)
    local = os.path.join(local_data_dir, f"predictions_{label}.csv")
    s3_client.download_file(bucket, key + "predictions.csv", local)
    return pd.read_csv(local, keep_default_na=False)

preds = {label: fetch_predictions(uri, label) for label, uri in [("base", base_output_uri), ("tuned", tuned_output_uri)]}

holdout_solution = train_df[train_df.page_id.isin(holdout_df.page_id)][["page_id", "text", "category"]]
holdout_solution.to_csv(os.path.join(local_data_dir, "holdout_solution.csv"), index=False)

for label, df in preds.items():
    sub = df[df.page_id.isin(holdout_solution.page_id)]
    sub.to_csv(os.path.join(local_data_dir, f"holdout_pred_{label}.csv"), index=False)
    print(f"\n=== {label} ===")
    print(subprocess.run(
        ["python", os.path.join(local_data_dir, "metric.py"),
         "--solution", os.path.join(local_data_dir, "holdout_solution.csv"),
         "--submission", os.path.join(local_data_dir, f"holdout_pred_{label}.csv")],
        capture_output=True, text=True).stdout)
```

Look at a page where the two differ most. The fine-tuned model should have picked up the conventions (`&` for the surveyors' "and" mark, `½` for raised fractions, no leader dashes) as well as the handwriting.

```python
merged = (holdout_solution
          .merge(preds["base"].rename(columns={"text": "base"}), on="page_id")
          .merge(preds["tuned"].rename(columns={"text": "tuned"}), on="page_id"))
example = merged[merged.category == "survey_notes"].iloc[0]
print("REFERENCE\n", example.text[:400], "\n\nBASE\n", example.base[:400], "\n\nTUNED\n", example.tuned[:400])
```

::::::::::::::::::::::::::::::::::::: callout

### Why the holdout number flatters you

The script holds out 15% of *pages*, so the same surveyor's hand appears in both the training and holdout halves. The competition's hidden test set is split by *document*: whole letters and notebooks the model has never seen. Expect the leaderboard CER to be worse than the holdout CER, especially on `dominy_accounts`, where 36 pages come from 20 different writers. A document-level holdout is the fix; it is left as an exercise because with only four German letter bundles in train, it means giving up a quarter of the German data.

::::::::::::::::::::::::::::::::::::::::::::::::

## Step 4: Submit

The `tuned` prediction file already contains every test page. Keep only those rows, in the order `test.csv` lists them, and upload the CSV on the competition's **Submit Prediction** page (two submissions per day).

```python
submission = test_df[["page_id"]].merge(preds["tuned"], on="page_id", how="left").fillna({"text": ""})
assert len(submission) == len(test_df) and submission.text.str.len().gt(0).all(), "every test page needs a prediction"

submission_path = os.path.join(local_data_dir, "submission_churro_lora.csv")
submission.to_csv(submission_path, index=False)
s3_client.upload_file(submission_path, bucket_name, f"{data_prefix}/submissions/submission_churro_lora.csv")
print("wrote", submission_path, len(submission), "rows")
```

## Step 5: What did this episode cost?

Add it up from the job descriptions, not from memory.

```python
total = 0.0
for job in session.sagemaker_client.list_training_jobs(NameContains="badger-churro", MaxResults=20)["TrainingJobSummaries"]:
    d = session.sagemaker_client.describe_training_job(TrainingJobName=job["TrainingJobName"])
    cost = d.get("BillableTimeInSeconds", 0) / 3600 * g5_price
    total += cost
    print(f"{job['TrainingJobName']:<50} {d['TrainingJobStatus']:<10} ${cost:.2f}")

for job in session.sagemaker_client.list_processing_jobs(NameContains="badger-transcribe", MaxResults=20)["ProcessingJobSummaries"]:
    d = session.sagemaker_client.describe_processing_job(ProcessingJobName=job["ProcessingJobName"])
    secs = (d.get("ProcessingEndTime", d["CreationTime"]) - d["CreationTime"]).total_seconds()
    cost = secs / 3600 * g5_price
    total += cost
    print(f"{job['ProcessingJobName']:<50} {d['ProcessingJobStatus']:<10} ${cost:.2f}")

print(f"\nGPU spend this episode: ~${total:.2f}  (plus the CPU notebook at $0.05/h)")
```

Then compare that number to the estimate in Step 0. If you are within 50%, your estimating is good enough to be trusted with a bigger run. If you are not, find out which line was wrong before launching anything else.

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 1: Cheaper per run, or just faster?

`ml.g6e.xlarge` (one L40S, 48 GB) costs $2.61/h, 1.85x the price of `ml.g5.xlarge`, and runs this workload roughly 2.5–3x faster. It also has twice the GPU memory, so `batch_size` can be 2 or 4 instead of 1, which uses the GPU more efficiently.

Using the Step 0 table, estimate the cost of one full run on `ml.g6e.xlarge`. Which phase does *not* get faster?

:::::::::::::::::::::::: solution

Startup (~10 min, $0.44) does not speed up: the container pull and the downloads are the same. Training drops from ~50 min to ~18 min ($0.78); eval from 4 to ~1.5 min. Total about 30 min, ~$1.30: a little cheaper than the g5 run and done in half the time. The lesson is that the hourly rate is the wrong thing to compare; the cost *per run* is what matters, and fixed overhead is a larger share of it on faster hardware. The catch is quotas: the account needs a service quota for `ml.g6e.xlarge` training jobs, and those often default to zero.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 2: The resolution knob

Rerun the smoke test with `max_pixels` set to `2560 * 28 * 28` (~2 megapixels). Compare the `training finished in` line to the first smoke test. Then look at the page image again: does the handwriting need that many pixels?

:::::::::::::::::::::::: solution

Time per page roughly doubles (the sequence is twice as long, and attention cost grows faster than linearly). On these scans the strokes are legible well below 1 megapixel, so the extra pixels mostly buy tokens that carry no information. Resolution is the first knob to turn *down* when a fine-tune is over budget, and the first to turn *up* if the model is misreading small characters such as the `½` fractions.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: challenge

### Exercise 3: Managed spot training

SageMaker can run a training job on spare ("spot") capacity at a discount of up to 70% in exchange for the risk of being interrupted. Add `use_spot_instances=True` and `max_wait=3 * 60 * 60` (the total time allowed including waiting for capacity; it must be at least `max_run`) to the full-run estimator and relaunch.

What else would the script need before this is safe to rely on for a 10-hour run?

:::::::::::::::::::::::: solution

Checkpointing. If the instance is reclaimed, the job restarts and, as written, starts training from scratch. For a one-hour job that costs an hour; for a ten-hour job it can cost the whole run. The fix is to save checkpoints to `/opt/ml/checkpoints` (which SageMaker syncs to S3 via `checkpoint_s3_uri`) with `save_strategy="steps"` and resume from the latest one when the script starts. Spot is a large saving on long runs once that is in place; for a one-hour class exercise with 50 people launching at once, the interruptions and capacity waits are not worth it.

:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::

## Cleanup

- The training and processing jobs have already ended; there is nothing to stop.
- Delete what you no longer need from `s3://<your-bucket>/badger-scribe/` (the `models/` and `predictions/` prefixes). The adapter is small, but prediction outputs accumulate across runs.
- **Stop your notebook instance.** It is a CPU instance at five cents an hour, which is exactly the kind of charge that goes unnoticed for a semester.

::::::::::::::::::::::::::::::::::::: keypoints

- Fine-tuning needs a GPU for the whole run and is iterative, which is why it is the most common source of surprise cloud bills. Training jobs, not GPU notebooks, are the safe way to run it: they end, and `max_run` guarantees it.
- Estimate the cost before launching: startup overhead + (pages × epochs × seconds per page) at the instance's hourly rate. Compare it to `BillableTimeInSeconds` afterwards.
- Smoke-test every script on a handful of examples and one epoch before a full run.
- Image resolution (`max_pixels`) and epochs are the two knobs that change cost by multiples; the instance size is the one that changes it by 5–18x for no benefit on a small job.
- LoRA lets a 3B vision-language model be fine-tuned on one 24 GB GPU in about an hour for about $1.50, and the output is an adapter of tens of megabytes, not a copy of the model.
- Measure the gain with a held-out set and the task's own metric, and remember that a page-level holdout is optimistic when the real test is split by document.

::::::::::::::::::::::::::::::::::::::::::::::::
