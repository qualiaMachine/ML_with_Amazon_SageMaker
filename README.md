# Machine Learning with Amazon SageMaker

A Carpentries-style lesson on running machine learning work on AWS without
losing control of the bill: SageMaker notebooks as controllers, training and
tuning jobs, retrieval-augmented generation on Bedrock and on rented GPUs, and
fine-tuning an open-weight vision-language model. Built for the UW–Madison
ML Marathon and Research Bazaar, usable by anyone with an AWS account.

**Rendered lesson:** https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/

## What the lesson covers

| Part | Episodes | Hands-on problem |
|---|---|---|
| SageMaker foundations | Overview, S3 setup, notebooks as controllers, accessing S3, code repos | Titanic survival classifier |
| Training and tuning | Training jobs (XGBoost, PyTorch), hyperparameter tuning | Titanic, CPU vs GPU, parallel jobs |
| Retrieval-augmented generation | Overview, Bedrock, Processing jobs | [WattBot](https://www.kaggle.com/competitions/WattBot2025) energy-literature Q&A |
| Fine-tuning | LoRA fine-tune of CHURRO-3B as a training job | [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe) handwriting transcription (German Kurrent letters) |
| Wrap-up | Resource management and cleanup, CLI equivalents | |

Every episode uses the same cost practices: cost-allocation tags on every
resource, `max_run` caps on every job, an estimate before each launch compared
to the bill afterwards, and stopping and deleting notebook instances when done.

## Repository layout

| Path | What it is |
|---|---|
| `episodes/*.md` | The lesson text. **Edit these**; everything else below derives from them. |
| `notebooks/*.ipynb` | Jupyter notebooks generated from the episodes by `scripts/md_to_ipynb.py`. Regenerated automatically on every push to `main`; do not edit by hand. |
| `scripts/` | Everything the episodes import or launch: `helpers.py` (bucket size, instance info, tuning-job cost), the training scripts (`train_xgboost.py`, `train_nn.py`), the RAG and fine-tuning scripts, plus the notebook generator and the data-release builder. Learners clone this repository once onto their notebook instance; there is no separate helper repository. |
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
