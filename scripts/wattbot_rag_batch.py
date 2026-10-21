"""WattBot RAG as a SageMaker Processing job.

Runs the full retrieval + generation loop over train_QA.csv with self-hosted
models: a Hugging Face embedding model for query embeddings and an open-weight
generator (Qwen3-8B in class) run with transformers' generate, which the
container already has. `--engine vllm` switches the generator to vLLM, the
inference engine for serving many requests at once or FP8 weights (the 27B
variant); it needs vllm in requirements.txt. All the pipeline logic lives in
`wattbot_rag.py`, which must sit next to this file in the job's source_dir;
this script only loads the inputs, builds the two model callables, and writes
`wattbot_solutions.csv`.

Expected input layout (set by the ProcessingInput destinations in the
Processing Jobs episode):

    <input_dir>/chunks/<anything>.jsonl   chunked corpus (from the Bedrock episode)
    <input_dir>/embeddings/embeddings.npy chunk embeddings (from the embedding job)
    <input_dir>/train/train_QA.csv        questions
    <input_dir>/metadata/metadata.csv     doc_id -> url
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from hf_embedder import HFEmbedder   # plain-transformers embedder, shipped next to this script

import wattbot_rag as wr


# ---------------------- generator: transformers (default) ----------------------


def load_hf_generator(model_id: str):
    """Load the chat model once with transformers and return (tokenizer, model).

    bf16 weights on the GPU: Qwen3-8B is ~16 GB, which fits an ml.g5.xlarge
    (24 GB) with room for generation. Nothing to install: the container ships
    transformers and PyTorch. `call_hf_chat_batch` decodes several prompts at
    once, which recovers most of what vLLM's batching would give on a job this size.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model with transformers: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    return tokenizer, model


def call_hf_chat(system_prompt: str, user_prompt: str, generator, max_new_tokens: int = 512) -> str:
    """Send one system + user turn through transformers' generate and return the text.

    The chat template turns the two messages into the token sequence the
    model was trained on; `enable_thinking=False` asks Qwen3 to answer directly
    rather than write a reasoning trace first. Greedy decoding, so the run is
    repeatable.
    """
    import torch

    tokenizer, model = generator
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def call_hf_chat_batch(pairs, generator, max_new_tokens: int = 512):
    """`call_hf_chat` for several (system, user) pairs at once: one `generate` call, one answer per pair.

    The prompts are padded on the left to the same length so that every sequence
    ends where generation starts; the pad tokens are masked out. Decoding eight
    prompts together costs little more than decoding one, which is where the
    speed comes from.
    """
    import torch

    tokenizer, model = generator
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for s, u in pairs
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip() for seq in out]


# ---------------------- generator: vLLM (--engine vllm) ----------------------


def _drop_broken_flash_attn():
    """Remove the container's flash-attn if it no longer imports.

    The SageMaker Hugging Face container ships a flash-attn binary built for
    its own PyTorch. A newer vLLM replaces PyTorch, the binary fails to link
    ("undefined symbol"), and vLLM, which imports flash-attn when it is
    present, dies at model load. vLLM has its own attention kernels, so the
    package can simply go.
    """
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "flash-attn"], check=False)
        print("Removed the container's flash-attn build (incompatible with the installed PyTorch); vLLM uses its own kernels.", flush=True)
    except Exception:  # noqa: BLE001
        pass


def load_qwen_pipeline(model_id: str = "Qwen/Qwen3-8B", max_model_len: int = 8192):
    """Load the generator once with vLLM and return the engine.

    vLLM is an open-source inference engine: given a model name it downloads the
    weights, loads them onto the GPU and answers chat requests from Python. It is
    the self-hosted counterpart of Bedrock's runtime.

    Qwen3-8B in bf16 is ~16 GB of weights and fits one 24 GB GPU (ml.g5.xlarge);
    Qwen3.8-27B in FP8 is ~27 GB and fits one 48 GB GPU
    (ml.g6e.xlarge). vLLM is used rather than transformers because it has
    the kernels for FP8 and for this model's hybrid attention, and it
    batches well. Called a single time per processing job and reused for
    every call to `call_qwen_chat`.
    """
    _drop_broken_flash_attn()
    from vllm import LLM

    print(f"Loading model with vLLM: {model_id}", flush=True)
    return LLM(
        model=model_id,
        max_model_len=max_model_len,                   # longest prompt + answer we need
        gpu_memory_utilization=0.85,                   # leave room for the embedding model
        limit_mm_per_prompt={"image": 0, "video": 0},  # text only
    )


def call_qwen_chat(system_prompt: str, user_prompt: str, generator, max_new_tokens: int = 512) -> str:
    """Send one system + user turn to the vLLM engine and return the generated text.

    The prompts arrive fully built from `wattbot_rag` (the user turn already
    carries the question and the retrieved context), so nothing is added here.
    """
    from vllm import SamplingParams

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = generator.chat(
        messages,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
        chat_template_kwargs={"enable_thinking": False},  # answer directly, no reasoning trace
        use_tqdm=False,
    )
    return outputs[0].outputs[0].text.strip()


# ---------------------- main ----------------------


def main():
    """Load the four inputs, build the two model callables, run every question
    through the shared pipeline, write wattbot_solutions.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--embedding_model_id", type=str, default="thenlper/gte-large")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--retrieval_threshold", type=float, default=0.25)
    parser.add_argument(
        "--qwen_model_id",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Open-weight chat model for generation.",
    )
    parser.add_argument("--max_questions", type=int, default=0, help="answer only the first N questions (0 = all); the scorer compares the rows it finds")
    parser.add_argument("--per_type", type=int, default=0, help="answer a stratified sample: up to N questions per evidence type, seed 0 (0 = all). The Bedrock episode uses the same draw, so the two runs are scored on the same questions")
    parser.add_argument("--batch_size", type=int, default=8, help="prompts decoded together on the GPU (transformers engine); 1 reproduces one-at-a-time")
    parser.add_argument("--engine", choices=["transformers", "vllm"], default="transformers",
                        help="transformers: nothing to install, prompts decoded in batches. vllm: needs vllm in requirements.txt; for FP8 weights or many requests.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Match the folder layout used in the ProcessingInput destinations
    chunks_dir = os.path.join(args.input_dir, "chunks")
    jsonl_files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"no .jsonl chunk file found in {chunks_dir}")
    chunks_path = os.path.join(chunks_dir, jsonl_files[0])
    emb_path = os.path.join(args.input_dir, "embeddings", "embeddings.npy")
    train_qa_path = os.path.join(args.input_dir, "train", "train_QA.csv")
    metadata_path = os.path.join(args.input_dir, "metadata", "metadata.csv")

    print("Loading inputs...", flush=True)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunked_docs = [json.loads(line) for line in f]
    chunk_embeddings = np.load(emb_path)
    train_df = pd.read_csv(train_qa_path)
    try:
        metadata_df = pd.read_csv(metadata_path)
    except UnicodeDecodeError:
        metadata_df = pd.read_csv(metadata_path, encoding="latin1")

    train_df = wr.drop_questions_without_papers(train_df, {c["doc_id"] for c in chunked_docs})
    if args.per_type:
        train_df = wr.sample_questions_by_type(train_df, per_type=args.per_type)
    if args.max_questions:
        train_df = train_df.head(args.max_questions)
    print(f"Chunks: {len(chunked_docs)}")
    print(f"Train QAs: {len(train_df)}")
    print("Embeddings shape:", chunk_embeddings.shape, flush=True)
    if chunk_embeddings.shape[0] != len(chunked_docs):
        raise ValueError("embeddings.npy and chunks.jsonl have different lengths; re-run the embedding job")

    docid_to_url = wr.build_docid_to_url(metadata_df)

    # The two model callables the shared pipeline needs
    embedder = HFEmbedder(args.embedding_model_id)   # the same model and pooling the embedding job used
    embed_query = wr.embed_query_with(embedder)

    if args.engine == "vllm":
        qwen_generator = load_qwen_pipeline(args.qwen_model_id)
        chat = call_qwen_chat
    else:
        qwen_generator = load_hf_generator(args.qwen_model_id)
        chat = call_hf_chat

    def generate(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        """The generate callable run_single_qa expects: same signature whether the model is Bedrock, transformers or vLLM."""
        return chat(system_prompt, user_prompt, qwen_generator, max_new_tokens)

    def generate_batch(pairs, max_new_tokens: int):
        """The batched form run_batch_qa expects: a list of (system, user) pairs in, a list of texts out."""
        if args.engine == "transformers":
            return call_hf_chat_batch(pairs, qwen_generator, max_new_tokens)
        return [generate(s, u, max_new_tokens) for s, u in pairs]

    t_gen = time.time()
    results = wr.run_batch_qa(
        rows=train_df,
        embed_query=embed_query,
        generate_batch=generate_batch,
        chunk_embeddings=chunk_embeddings,
        chunked_docs=chunked_docs,
        docid_to_url=docid_to_url,
        top_k=args.top_k,
        retrieval_threshold=args.retrieval_threshold,
        batch_size=args.batch_size,
        log=lambda msg: print(msg, flush=True),
    )
    print(f"Answered {len(results)} questions in {(time.time() - t_gen) / 60:.1f} min", flush=True)
    for out in results:
        print(f"{out['id']}: answer_value={out['answer_value']} ref_id={out['ref_id']}", flush=True)

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "wattbot_solutions.csv")
    results_df.to_csv(out_path, index=False)
    print("Wrote predictions to", out_path)


if __name__ == "__main__":
    main()
