---
title: 'Shared data bucket: staging before each term'
---

Learners create exactly one bucket of their own (S3 episode) and write every output there. Everything larger than the Titanic CSVs lives in **one shared, read-only bucket** that instructors stage before class, so the class holds one copy of each dataset instead of fifty. This page is the whole staging procedure; the episodes assume the layout below.

The whole procedure below is also a script, `instructors/stage-shared-bucket.sh`: set the bucket name, region and account id at its top and run it from a notebook instance terminal. The sections below explain what it does.

## Layout

```
s3://<shared-bucket>/
    wattbot/
        metadata.csv        from data/data.zip in the lesson repo
        train_QA.csv        from data/data.zip
        corpus.zip          built once with scripts/build_wattbot_corpus.py
    badger-scribe/
        train.csv  test.csv  metadata.csv  metric.py
        sample_submission.csv  transcription_conventions.md
        images/<page_id>.jpg
        model/churro-3B/       the CHURRO-3B weights, copied once from Hugging Face (~7.5 GB)
        instructor/full-run/   train_summary.json, eval_before.csv, eval_after.csv
```

Learners set `shared_bucket = "<this bucket's name>"` in the first cell of the RAG and fine-tuning episodes. Tell them the name at the start of each session.

## 1. Create the bucket and make it readable by the student role

Create the bucket in the class region (same naming rules as the S3 episode; something like `ml-marathon-shared`), with the three cost-allocation tags. Then attach this read-only policy (Permissions → Bucket policy), editing the bucket name and replacing `<account-id>` and the role names with your account's:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::<account-id>:role/ml-sagemaker-use",
                    "arn:aws:iam::<account-id>:role/ml-sagemaker-bedrock-use"
                ]
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::ml-marathon-shared",
                "arn:aws:s3:::ml-marathon-shared/*"
            ]
        }
    ]
}
```

`s3:ListBucket` is required: a training job's input channel lists the prefix before downloading it. No `PutObject`: learners never write here.

## 2. Stage WattBot (RAG episodes)

From a machine with Python and the AWS CLI (CloudShell works), in a clone of the lesson repo:

```bash
unzip -o data/data.zip -d /tmp/lessondata
pip install pandas requests
python scripts/build_wattbot_corpus.py --metadata /tmp/lessondata/data/wattbot/metadata.csv --out /tmp/lessondata/wattbot/corpus.zip
cp /tmp/lessondata/data/wattbot/metadata.csv /tmp/lessondata/data/wattbot/train_QA.csv /tmp/lessondata/wattbot/
aws s3 sync /tmp/lessondata/wattbot/ s3://<shared-bucket>/wattbot/
```

The corpus script downloads every PDF listed in `metadata.csv` and reports any URL that failed; decide whether to drop or fix those before staging. The corpus only needs rebuilding when `metadata.csv` changes.

## 3. Stage Badger Scribe (fine-tuning episode)

Run this from a laptop or a notebook instance, or from CloudShell for the ~130 MB Badger Scribe release only (CloudShell's home directory is 1 GB, which is why the loop handles one asset at a time in `/tmp`: download, unzip, sync, delete, so nothing larger than one zip is ever on disk). From the lesson's data release (see *Badger Scribe data: publishing a release* for how versions are published):

```bash
V=1
R=https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases/download/badger-scribe-data-v$V
for a in labels images-train images-test; do
  rm -rf /tmp/bs && mkdir -p /tmp/bs
  curl -L -o /tmp/bs/$a.zip $R/badger-scribe-v$V-$a.zip
  unzip -q -o /tmp/bs/$a.zip -d /tmp/bs/badger-scribe && rm /tmp/bs/$a.zip
  aws s3 sync /tmp/bs/badger-scribe/ s3://<shared-bucket>/badger-scribe/
  rm -rf /tmp/bs
done
```

Then run the full fine-tune once (homework estimator in the episode) and copy `train_summary.json`, `eval_before.csv`, `eval_after.csv` from its `model.tar.gz` to `s3://<shared-bucket>/badger-scribe/instructor/full-run/`.

## 4. Stage the CHURRO-3B weights (fine-tuning episode)

Every training and processing job in the episode reads the base model from the bucket as a `model` channel, so that 100 jobs starting in the same minute copy it from S3 instead of each pulling 7.5 GB from Hugging Face. Do this once per bucket, from a laptop or a notebook instance with at least 10 GB free (not CloudShell):

```bash
pip install huggingface_hub
huggingface-cli download stanford-oval/churro-3B --local-dir churro-3B
aws s3 sync churro-3B/ s3://<shared-bucket>/badger-scribe/model/churro-3B/
rm -rf churro-3B
```

About 7.5 GB; it only needs redoing if the Hub repository changes or the bucket is rebuilt.

## From a SageMaker notebook

The same procedure as the script, as cells for a notebook on a `ml.t3.medium` instance with the `conda_python3` kernel. The notebook's role must be allowed to create buckets and set bucket policies; if cell 2 fails with `AccessDenied`, create the bucket and paste the policy through the console (as in the S3 episode) and continue from cell 3. **Cell 1: names.** The shared bucket follows the lesson's example name; the account id and region come from the session.

```python
import json, boto3, sagemaker

session = sagemaker.Session()
region = session.boto_region_name
account_id = boto3.client("sts").get_caller_identity()["Account"]
s3 = boto3.client("s3", region_name=region)

shared_bucket = "ml-marathon-shared"
roles = ["ml-sagemaker-use", "ml-sagemaker-bedrock-use"]   # execution roles that may read the bucket
release_v = 1                                               # Badger Scribe data release version
repo = "https://github.com/carpentries-incubator/ML_with_AWS_SageMaker"
print(region, account_id, shared_bucket)
```

**Cell 2: create the bucket, tag it, block public access, attach the read-only policy.**

```python
try:
    s3.head_bucket(Bucket=shared_bucket)
    print("bucket exists")
except s3.exceptions.ClientError:
    s3.create_bucket(Bucket=shared_bucket, CreateBucketConfiguration={"LocationConstraint": region})
    print("bucket created")

s3.put_bucket_tagging(Bucket=shared_bucket, Tagging={"TagSet": [
    {"Key": "Project", "Value": "Workshop"}, {"Key": "Name", "Value": "Instructor"}, {"Key": "Purpose", "Value": "Storage"}]})
s3.put_public_access_block(Bucket=shared_bucket, PublicAccessBlockConfiguration={
    "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})

policy = {"Version": "2012-10-17", "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": [f"arn:aws:iam::{account_id}:role/{r}" for r in roles]},
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [f"arn:aws:s3:::{shared_bucket}", f"arn:aws:s3:::{shared_bucket}/*"]}]}
s3.put_bucket_policy(Bucket=shared_bucket, Policy=json.dumps(policy))
print("read-only policy attached for", roles)
```

**Cell 3: WattBot inputs** (metadata, questions, and the PDF corpus built from the metadata). `/tmp` on a notebook instance is a small RAM disk, so the work directory goes on the persistent volume; these files are a few hundred MB.

```python
work = "/home/ec2-user/SageMaker/stage"
!rm -rf {work} && mkdir -p {work}
%cd {work}
!git clone -q --depth 1 {repo} lesson
!pip install -q pandas requests huggingface_hub
!unzip -q -o lesson/data/data.zip -d data
!mkdir -p wattbot && cp data/data/wattbot/metadata.csv data/data/wattbot/train_QA.csv wattbot/
!python lesson/scripts/build_wattbot_corpus.py --metadata wattbot/metadata.csv --out wattbot/corpus.zip
!aws s3 sync wattbot/ s3://{shared_bucket}/wattbot/
```

**Cell 4: Badger Scribe data release**, one asset at a time.

```python
R = f"{repo}/releases/download/badger-scribe-data-v{release_v}"
for a in ["labels", "images-train", "images-test"]:
    !rm -rf bs && mkdir -p bs
    !curl -sSL -o bs/{a}.zip {R}/badger-scribe-v{release_v}-{a}.zip
    !unzip -q -o bs/{a}.zip -d bs/badger-scribe && rm bs/{a}.zip
    !aws s3 sync bs/badger-scribe/ s3://{shared_bucket}/badger-scribe/
!rm -rf bs
```

**Cell 5: CHURRO-3B weights**, streamed from Hugging Face into S3 file by file (about 7.5 GB in total; nothing is written to the instance's disk, so the volume size does not matter).

```python
import requests
from huggingface_hub import HfApi, hf_hub_url

model_id = "stanford-oval/churro-3B"
prefix = "badger-scribe/model/churro-3B"
files = [f for f in HfApi().list_repo_files(model_id) if not f.startswith(".")]
for f in files:
    with requests.get(hf_hub_url(model_id, f), stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        s3.upload_fileobj(r.raw, shared_bucket, f"{prefix}/{f}")   # multipart upload straight from the stream
    print("uploaded", f)
```

**Cell 6: check**, then clean up.

```python
!aws s3 ls s3://{shared_bucket}/ --recursive --summarize | tail -2
!aws s3 ls s3://{shared_bucket}/badger-scribe/
!aws s3 ls s3://{shared_bucket}/badger-scribe/model/churro-3B/
!aws s3 ls s3://{shared_bucket}/wattbot/
%cd /home/ec2-user/SageMaker
!rm -rf {work}
```

Still by hand: run the fine-tuning episode's homework estimator once and copy `train_summary.json`, `eval_before.csv` and `eval_after.csv` from its `model.tar.gz` to `s3://ml-marathon-shared/badger-scribe/instructor/full-run/`.

## 5. Keep it, or rebuild it

The end-of-workshop cleanup (`instructors/instructor-notes.md`, aws-nuke) deletes every S3 bucket in the account unless excluded. Either add the shared bucket to the nuke config's exclude filter, or accept that it goes and redo steps 1 to 4 next term; the data steps take about fifteen minutes and the model copy as long as a 7.5 GB download takes where you are. Storage is ~130 MB for Badger Scribe plus the WattBot corpus, and ~7.5 GB for the model copy: well under a dollar a month.

## Self-paced learners

Anyone working outside the shared account does steps 1, 3 and 4 (and 2 for the RAG episodes) into a bucket of their own and sets `shared_bucket` to its name. The episodes say so where it matters.
