# Machine Learning on AWS with SageMaker and Bedrock

A Carpentries-style lesson on machine learning workflows on AWS, developed for
the UW-Madison ML Marathon and Research Bazaar and usable by anyone with an AWS
account.

**Rendered lesson:** https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/

## What the lesson covers

This workshop teaches researchers how to run machine learning and AI work on AWS without losing track of what it costs. It works through the practical decisions, where to store the data, which compute to use, how to launch and monitor a job, and how to estimate and manage the cost.

We use Amazon SageMaker to train, fine-tune and run our own models, and Amazon Bedrock to call models AWS hosts through an API. Throughout, we work from a small SageMaker notebook instance that acts as the controller. The demanding work runs on separate instances that SageMaker starts for each job and releases when the job finishes.

The workshop is in four parts, from AWS basics to training, inference and resource management.

**Part 1, orientation.** What cloud computing offers a researcher, when AWS makes sense next to a laptop or a campus cluster, and how its pricing works. Then the setup every later episode uses.

- **AWS services and pricing.** What SageMaker, Bedrock and S3 each do, and the difference between paying for compute time and paying per token.
- **Working environment.** Create an S3 bucket for data and results, launch a small SageMaker notebook instance, and read and write files in S3 from Python. Optionally, put your notebooks under version control.
- **Cost management.** Choose compute to fit the job, estimate the cost before launching, and track spending across the resources you create.

**Part 2, SageMaker.** Training, fine-tuning and hyperparameter tuning. SageMaker runs the training scripts on separate compute instances, starting with small models and ending with a vision-language model of billions of parameters. Along the way we compare hardware choices and see how training time, model size and the number of experiments set the cost.

- **Model training.** Train XGBoost and PyTorch models on the Titanic data. Launch the training jobs from the notebook, compare CPU and GPU, and see when more than one instance helps.
- **Fine-tuning.** Adapt CHURRO-3B, an open-weight vision-language model, to transcribe historical German handwriting from the Badger Scribe challenge. Use LoRA to cut the GPU memory needed, estimate the run's cost, and measure transcription accuracy before and after.
- **Hyperparameter tuning.** Set up a search over learning rates and epochs, let SageMaker run the training jobs, and compare the results under a cost ceiling.

**Part 3, Bedrock.** Inference and retrieval-augmented generation (RAG). Training changes a model's parameters. Inference uses a model that is already trained. We build a RAG pipeline that retrieves passages from research papers and has a language model answer questions from that evidence, then build it again with open-weight models on SageMaker, on the same problem, to compare cost, performance and flexibility.

- **RAG with Bedrock.** Build the pipeline for WattBot, a challenge of questions about the environmental impact of AI. Split research papers into searchable passages, embed them, retrieve the relevant evidence, and produce answers with citations. The models are Bedrock's, with usage tracked through tagged inference profiles.
- **Multimodal retrieval.** Extend the pipeline to retrieve from figures and tables as well as text, and measure whether that improves the answers.
- **RAG with SageMaker.** Run the embedding and generation models yourself, as GPU Processing jobs that shut down when they finish, and compare the results and the costs with Bedrock. An optional episode runs the same pipeline interactively on a GPU notebook, to see model loading and inference step by step.

**Part 4, cleanup.** Resource management and cost tracking. Cloud resources keep billing after you stop using them, notebook instances and stored data above all. We finish by reviewing what was created, checking what it cost, and cleaning up the account.

- Find the running resources and stored data, stop the notebook instance, and remove what is no longer needed, including S3 buckets and Bedrock inference profiles.
- Use Cost Explorer and the resource tags to see where the workshop's spending went.
- Review the account rules that kept costs in check during the workshop, and how to set the same safeguards on an AWS account of your own.
- Optionally, launch and manage the same resources from a terminal with the AWS CLI.

| Episodes | What they cover | Hands-on problem |
|---|---|---|
| 1 to 5 | Overview, S3, the notebook instance as controller, reading and writing S3, (optional) pushing notebooks to GitHub | Titanic survival data |
| 6, 7 and 9 | Training jobs with XGBoost and PyTorch, CPU vs. GPU, parallel instances, hyperparameter tuning | Titanic survival classifier |
| 8 | LoRA fine-tune of CHURRO-3B as a training job, with a cost estimate checked against the bill | [Badger Scribe](https://www.kaggle.com/competitions/badger-scribe) handwriting transcription (German Kurrent letters) |
| 10 to 13 | RAG on Bedrock, with multimodal embeddings; RAG with Processing jobs; (optional) RAG on a notebook GPU | [WattBot](https://www.kaggle.com/competitions/WattBot2026) energy-literature Q&A |
| 14 to 15 | (Optional) the same jobs from the terminal; resource management, monitoring and cleanup | |

### Who it is for

Researchers and students who already use machine learning or AI and want to run that work in the cloud. You should be comfortable with Python and Jupyter and know the basics of model training, hyperparameter tuning, fine-tuning, embeddings and RAG. The emphasis is on using AWS, not on teaching those methods. Instructors may review some of them briefly depending on the audience, but the workshop assumes the background. No previous AWS experience is needed.

### Cost controls

One of the workshop's main points is that cloud computing is affordable for research when the resources are chosen and managed with care. Run as written, the exercises cost each learner a few dollars, CPU and GPU training jobs, hyperparameter tuning, a LoRA fine-tune and RAG inference included.

We estimate the cost before launching a job, set a limit on how long it can run, and check its billable time afterwards. Resources are tagged so that spending can be attributed to each learner and project. We also look at how fast the cost grows when a GPU instance is left running or an experiment is scaled up without checking what it needs.

The workshop uses a shared AWS account with restrictions that block the expensive mistakes. The final part looks at those safeguards and how to set the same ones on an account of your own.

### What it does not cover

- **General ML/AI instruction.** The focus is on running existing ML and AI workflows on AWS, not on teaching the algorithms.
- **Model deployment.** We do not deploy trained or fine-tuned models as always-on inference endpoints.
- **AI agents and tool use.** We do not cover agent frameworks or applications that let a language model call external tools.

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

Learners create one bucket of their own, in the S3 episode, and every later episode writes its outputs there under a prefix.

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
