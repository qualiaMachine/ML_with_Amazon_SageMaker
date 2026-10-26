# Machine Learning on AWS with SageMaker and Bedrock

A Carpentries-style lesson on machine learning workflows on AWS, developed for
the UW-Madison ML Marathon and Research Bazaar and usable by anyone with an AWS
account.

**Rendered lesson:** https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/

## What the lesson covers

The lesson has two parts.

1. **SageMaker: training, tuning and fine-tuning.** Storing data in S3, using a
   small notebook instance as a *controller*, running training jobs (XGBoost and
   PyTorch) and hyperparameter tuning, with guidance on choosing CPU vs. GPU
   instances and when parallel instances help; then fine-tuning a 3B open-weight
   vision-language model with LoRA as a training job, for handwriting
   transcription, with before/after evaluation on held-out pages.
2. **Retrieval-augmented generation (RAG).** Building a RAG pipeline over a corpus
   of research papers two ways: with Amazon Bedrock (pay per token, nothing left
   running), including multimodal embeddings for the questions that live in
   tables and figures, and with SageMaker Processing jobs serving an open-weight
   model that Bedrock does not offer. An optional episode does the same on a
   notebook GPU, as the comparison.

| Episodes | What they cover | Hands-on problem |
|---|---|---|
| 1 to 5 | Overview, S3, the notebook instance as controller, reading and writing S3, (optional) pushing notebooks to GitHub | Titanic survival data |
| 6 to 8 | Training jobs with XGBoost and PyTorch, CPU vs. GPU, parallel instances, hyperparameter tuning | Titanic survival classifier |
| 9 | LoRA fine-tune of CHURRO-3B as a training job, with a cost estimate checked against the bill | [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe) handwriting transcription (German Kurrent letters) |
| 10 to 13 | RAG on Bedrock, with multimodal embeddings; RAG with Processing jobs; (optional) RAG on a notebook GPU | [WattBot](https://www.kaggle.com/competitions/WattBot2026) energy-literature Q&A |
| 14 to 15 | (Optional) the same jobs from the terminal; resource management, monitoring and cleanup | |

### Who it is for

Researchers and students who use ML or AI in their work and want to run it on
AWS. The lesson assumes a basic machine learning background: you know what
training, hyperparameter tuning, fine-tuning, embeddings and retrieval-augmented
generation are. It does not teach these concepts. Python and Jupyter are used
throughout; no prior AWS experience is needed.

### Cost controls

Run as written, the full workshop costs a few dollars per learner. Every episode
tags its resources, caps job runtime, estimates cost before launching and checks
the bill after, and cleans up. The final episode covers monitoring and cleanup,
and the rules a shared account can enforce.

### What it does not cover

- **General ML/AI concepts**: the focus is on moving existing ML and AI work to
  the cloud, not on teaching training, fine-tuning or RAG themselves.
- **Deploying models as endpoints**: serving a trained or fine-tuned model behind
  an always-on API.
- **Agents and tool use**: LLM applications that call tools or chain multiple
  model calls (e.g., Bedrock Agents).

## Repository layout

| Path | What it is |
|---|---|
| `episodes/*.md` | The lesson text. **Edit these**; everything else below derives from them. |
| `notebooks/*.ipynb` | Jupyter notebooks generated from the episodes by `scripts/md_to_ipynb.py`. Regenerated automatically on every push to `main`; do not edit by hand. |
| `scripts/` | Everything the episodes import or launch: `helpers.py` (bucket size, instance info, tuning-job cost, training and fine-tuning cost estimates), the training scripts (`train_xgboost.py`, `train_nn.py`), the RAG and fine-tuning scripts, plus the notebook generator and the data-release builder. Learners clone this repository once onto their notebook instance; there is no separate helper repository. |
| `learners/` | Setup instructions, glossary, instance-selection guide. |
| `instructors/` | Instructor notes: account preparation and quotas, per-episode cost estimates, data staging, end-of-workshop cleanup with aws-nuke. |
| `data/data.zip` | The Titanic CSVs used in the first half of the lesson. |
| `config.yaml` | Lesson configuration and episode order ([Workbench docs](https://carpentries.github.io/sandpaper-docs/)). |

## Data

- **Titanic** (training episodes): `data/data.zip` in this repository; learners download it before the workshop.
- **WattBot** (RAG episodes): `metadata.csv` and `train_QA.csv` are in the same `data.zip`; the PDF corpus is built once with `scripts/build_wattbot_corpus.py`. Instructors stage all three in the shared bucket; learners read them from there.
- **Badger Scribe** (fine-tuning episode): published as a [GitHub Release](https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/releases) on this repository (`badger-scribe-data-v1`: 295 train and 141 test pages of the Anna Seifert correspondence, Max Kade Institute, with human transcriptions). Instructors stage it in the shared S3 bucket before class. See `instructors/badger-scribe-data-release.md` for publishing a new version and `instructors/shared-data-bucket.md` for staging.

Learners create exactly one bucket of their own, in the S3 episode, and every later episode writes its outputs there under a prefix.

## Running the workshop

Before class, work through `instructors/aws-account-prep.md` (service quotas,
Bedrock model access, cost-allocation tags) and the per-episode pages in
`instructors/`. GPU quota increases can take weeks; file them early. After
class, `instructors/instructor-notes.md` covers cleaning the shared account
with aws-nuke.

## Contributing

Edit the Markdown in `episodes/` and open a pull request; the site and the
notebooks rebuild automatically. Code blocks in an episode become notebook
cells, so keep each block runnable on its own. Issues and suggestions:
https://github.com/carpentries-incubator/ML_with_AWS_SageMaker/issues or
endemann@wisc.edu.

The lesson is developed on the [qualiaMachine fork](https://github.com/qualiaMachine/ML_with_Amazon_SageMaker)
and synced to the incubator repository by pull request.

## Citation and license

Please cite the lesson using `CITATION.cff`. The instructional material is
licensed CC-BY 4.0 and the code MIT; see `LICENSE.md`. The Badger Scribe
page images and transcriptions carry their own rights statements, recorded
in the release notes and in `instructors/badger-scribe-data-release.md`.
