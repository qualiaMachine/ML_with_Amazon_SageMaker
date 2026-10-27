# Machine Learning on AWS with SageMaker and Bedrock

A Carpentries-style lesson on machine learning workflows on AWS, developed for
the UW-Madison ML Marathon and Research Bazaar and usable by anyone with an AWS
account.

**Rendered lesson:** https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/

## What the lesson covers

This workshop is about running machine learning and AI work on AWS without losing control of the bill: where the data goes, which machine does what, how a job is launched and paid for, and how to find out what it cost. It is in four parts.

**Part 1, orientation.** What the cloud offers a researcher compared with a laptop or the cluster down the hall, and what it costs. The two services the lesson uses: **SageMaker** rents machines by the second, for a notebook you work in and for jobs that run on their own and end; **Bedrock** serves hosted models by the token, with nothing to start or stop. The four things every episode uses, S3, a notebook instance, jobs and Bedrock, and how every later episode estimates and checks what a job costs.

**Part 2, SageMaker: data, training, fine-tuning and tuning.**

- Put the data in an S3 bucket and start a small notebook instance that launches everything else. The notebook is the controller, not the workhorse: the expensive compute runs on instances SageMaker rents for each job and releases when it ends.
- Train XGBoost and PyTorch models on the Titanic data as training jobs. Estimate the cost before each launch and read the billable seconds after, compare a CPU instance with a GPU, and see what two instances buy and what they do not.
- Fine-tune a 3B open-weight vision-language model with LoRA, as a training job with checkpoints and a runtime cap, for transcribing handwritten German letters (Badger Scribe), scored on held-out pages before and after.
- Run a hyperparameter tuning job that trains ten models under a budget set up front.

**Part 3, Bedrock: inference and retrieval-augmented generation.**

- Build a RAG pipeline over a corpus of energy research papers (WattBot): embed the papers, retrieve, generate, score. The models are Bedrock's, paid per token through tagged inference profiles so the spend has an owner, and the whole run costs cents.
- Add multimodal embeddings for the questions whose answers live in tables and figures, and measure what they change.
- Run the same pipeline with an open-weight model Bedrock does not serve, as SageMaker Processing jobs that rent a GPU for the minutes they need; optionally, the same on a notebook GPU, as the comparison.

**Part 4, resource management and cleanup.** What to stop, what to delete, how to read the bill, and the rules the account enforces on every launch, with how to set the same rules on an account of your own. An optional episode shows the same jobs from a terminal.

| Episodes | What they cover | Hands-on problem |
|---|---|---|
| 1 to 5 | Overview, S3, the notebook instance as controller, reading and writing S3, (optional) pushing notebooks to GitHub | Titanic survival data |
| 6, 7 and 9 | Training jobs with XGBoost and PyTorch, CPU vs. GPU, parallel instances, hyperparameter tuning | Titanic survival classifier |
| 8 | LoRA fine-tune of CHURRO-3B as a training job, with a cost estimate checked against the bill | [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe) handwriting transcription (German Kurrent letters) |
| 10 to 13 | RAG on Bedrock, with multimodal embeddings; RAG with Processing jobs; (optional) RAG on a notebook GPU | [WattBot](https://www.kaggle.com/competitions/WattBot2026) energy-literature Q&A |
| 14 to 15 | (Optional) the same jobs from the terminal; resource management, monitoring and cleanup | |

### Who it is for

Researchers and students who use ML or AI in their work and want or need to run
it in the cloud. The lesson assumes a basic machine learning background: you know what
training, hyperparameter tuning, fine-tuning, embeddings and retrieval-augmented
generation are. It does not teach these concepts. Python and Jupyter are used
throughout; no prior AWS experience is needed.

### Cost controls

The cloud is often very affordable for research workloads, and this lesson shows it: run as written, the whole workshop, training jobs, a tuning job, a LoRA fine-tune on a GPU and two RAG pipelines, costs each learner a few dollars. Every episode tags its resources, caps job runtime, estimates cost before launching and checks the bill after, and cleans up. The final episode covers monitoring and cleanup, and the rules a shared account can enforce.

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
