"""WattBot RAG as a SageMaker Processing job.

Runs the full retrieval + generation loop over train_QA.csv with self-hosted
models: a sentence-transformer for query embeddings and an open-weight
generator (Qwen3.8-27B in FP8) served by vLLM. All the pipeline logic lives in
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

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import wattbot_rag as wr


# ---------------------- generator: Qwen via vLLM ----------------------


def load_qwen_pipeline(model_id: str = "Qwen/Qwen3.8-27B-FP8", max_model_len: int = 8192):
    """Load the generator once with vLLM and return the engine.

    Qwen3.8-27B in FP8 is ~27 GB of weights and fits one 48 GB GPU
    (ml.g6e.xlarge). vLLM is used rather than transformers because it has
    the kernels for FP8 and for this model's hybrid attention, and it
    batches well. Called a single time per processing job and reused for
    every call to `call_qwen_chat`.
    """
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--embedding_model_id", type=str, default="thenlper/gte-large")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--retrieval_threshold", type=float, default=0.25)
    parser.add_argument(
        "--qwen_model_id",
        type=str,
        default="Qwen/Qwen3.8-27B-FP8",
        help="Open-weight chat model for generation (served with vLLM).",
    )
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

    print(f"Chunks: {len(chunked_docs)}")
    print(f"Train QAs: {len(train_df)}")
    print("Embeddings shape:", chunk_embeddings.shape, flush=True)
    if chunk_embeddings.shape[0] != len(chunked_docs):
        raise ValueError("embeddings.npy and chunks.jsonl have different lengths; re-run the embedding job")

    docid_to_url = wr.build_docid_to_url(metadata_df)

    # The two model callables the shared pipeline needs
    embedder = SentenceTransformer(args.embedding_model_id)
    embed_query = wr.embed_query_with(embedder)

    qwen_generator = load_qwen_pipeline(args.qwen_model_id)

    def generate(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
        return call_qwen_chat(system_prompt, user_prompt, qwen_generator, max_new_tokens)

    results = []
    for i, (_, row) in enumerate(train_df.iterrows(), start=1):
        out = wr.run_single_qa(
            row=row,
            embed_query=embed_query,
            generate=generate,
            chunk_embeddings=chunk_embeddings,
            chunked_docs=chunked_docs,
            docid_to_url=docid_to_url,
            top_k=args.top_k,
            retrieval_threshold=args.retrieval_threshold,
        )
        results.append(out)
        print(f"[{i}/{len(train_df)}] {out['id']}: answer_value={out['answer_value']} ref_id={out['ref_id']}", flush=True)

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "wattbot_solutions.csv")
    results_df.to_csv(out_path, index=False)
    print("Wrote predictions to", out_path)


if __name__ == "__main__":
    main()
