"""LoRA fine-tune of CHURRO-3B (a Qwen2.5-VL-3B fine-tune for historical
text) on the Badger Scribe train set. Runs as a SageMaker training job.

SageMaker passes hyperparameters as command-line flags and mounts the
`train` channel at $SM_CHANNEL_TRAIN. That directory must contain:

    train.csv              page_id, doc_id, text, category, label_source
    metadata.csv           optional catalog context, joined on doc_id
    images/<page_id>.jpg   one scan per page

What gets written to $SM_MODEL_DIR (SageMaker tars it into model.tar.gz):

    adapter_config.json, adapter_model.safetensors   the LoRA adapter
    processor files                                  so inference needs no base download
    holdout_page_ids.csv                             pages held out of training
    train_summary.json                               what this run did and scored

The base weights are never modified: a LoRA adapter is a few tens of MB that
sits on top of the frozen 3B model. That is what makes this fit on a 24 GB
GPU and finish in about an hour.
"""

import argparse
import json
import os
import random
import time

import pandas as pd
import torch
from torch.utils.data import Dataset

from badger_common import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    build_prompt,
    full_messages,
    load_metadata,
    load_page_image,
    macro_cer,
    user_messages,
)


def parse_args():
    p = argparse.ArgumentParser()
    # --- what to train ---
    p.add_argument("--model_id", type=str, default="stanford-oval/churro-3B")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=1, help="pages per step per GPU")
    p.add_argument("--grad_accum", type=int, default=8, help="effective batch = batch_size * grad_accum")
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    p.add_argument("--use_metadata", type=int, default=1)
    # --- how much of the data ---
    p.add_argument("--holdout_frac", type=float, default=0.15, help="fraction of pages per category kept for scoring")
    p.add_argument("--max_train_pages", type=int, default=0, help="0 = all; set small (e.g. 20) for a smoke test")
    p.add_argument("--eval_pages", type=int, default=20, help="holdout pages transcribed after training to report CER (0 = skip)")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    # --- SageMaker paths ---
    p.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def split_pages(df: pd.DataFrame, holdout_frac: float, seed: int):
    """Stratified page-level holdout. NOTE: the competition's hidden test set is
    split by *document*, so a page-level holdout (same writer in both halves)
    gives an optimistic CER. It is good enough to see whether training helped."""
    rng = random.Random(seed)
    hold = []
    for cat, grp in df.groupby("category"):
        ids = list(grp["page_id"])
        rng.shuffle(ids)
        n = max(1, int(round(len(ids) * holdout_frac))) if holdout_frac > 0 else 0
        hold.extend(ids[:n])
    hold = set(hold)
    return df[~df["page_id"].isin(hold)].copy(), df[df["page_id"].isin(hold)].copy()


class PageDataset(Dataset):
    def __init__(self, df, images_dir, doc_ctx, max_pixels):
        self.rows = df.to_dict("records")
        self.images_dir = images_dir
        self.doc_ctx = doc_ctx
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = build_prompt(r["category"], self.doc_ctx.get(r["doc_id"], ""))
        image = load_page_image(os.path.join(self.images_dir, f"{r['page_id']}.jpg"), self.max_pixels)
        return {"image": image, "prompt": prompt, "answer": r["text"]}


class Collator:
    """Tokenize image + prompt + transcription, and mask everything except the
    transcription out of the loss. The model is only graded on what it writes
    as the assistant, not on reproducing the prompt."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        full_texts, prompt_texts = [], []
        for b in batch:
            full_texts.append(self.processor.apply_chat_template(
                full_messages(b["prompt"], b["answer"]), tokenize=False, add_generation_prompt=False))
            prompt_texts.append(self.processor.apply_chat_template(
                user_messages(b["prompt"]), tokenize=False, add_generation_prompt=True))

        enc = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, (ptxt, img) in enumerate(zip(prompt_texts, images)):
            # Run the prompt alone through the processor to learn how many
            # tokens it occupies once the image placeholder is expanded.
            n_prompt = self.processor(text=[ptxt], images=[img], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :n_prompt] = -100
        enc["labels"] = labels
        return enc


# ---------------------------------------------------------------------------
# Evaluation (greedy transcription of a few holdout pages)
# ---------------------------------------------------------------------------
@torch.no_grad()
def transcribe(model, processor, image, prompt, max_new_tokens):
    text = processor.apply_chat_template(user_messages(prompt), tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    gen = out[0, enc["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def evaluate(model, processor, df, images_dir, doc_ctx, max_pixels, n_pages, max_new_tokens, seed):
    if n_pages <= 0 or len(df) == 0:
        return None
    sample = df.sample(n=min(n_pages, len(df)), random_state=seed)
    model.eval()
    processor.tokenizer.padding_side = "left"
    preds = []
    t0 = time.time()
    for r in sample.to_dict("records"):
        img = load_page_image(os.path.join(images_dir, f"{r['page_id']}.jpg"), max_pixels)
        preds.append(transcribe(model, processor, img, build_prompt(r["category"], doc_ctx.get(r["doc_id"], "")), max_new_tokens))
    sample = sample.assign(text_pred=preds).rename(columns={"text": "text_ref"})
    macro, by_cat = macro_cer(sample)
    print(f"[eval] {len(sample)} holdout pages in {time.time() - t0:.0f}s -> macro CER {macro:.4f}")
    print(by_cat.to_string())
    return {"n_pages": int(len(sample)), "macro_cer": macro, "per_category": by_cat.round(4).to_dict()}


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    images_dir = os.path.join(args.train_dir, "images")
    df = pd.read_csv(os.path.join(args.train_dir, "train.csv"), keep_default_na=False)
    meta_path = os.path.join(args.train_dir, "metadata.csv")
    doc_ctx = load_metadata(meta_path) if (args.use_metadata and os.path.exists(meta_path)) else {}

    train_df, hold_df = split_pages(df, args.holdout_frac, args.seed)
    if args.max_train_pages > 0:
        train_df = train_df.sample(n=min(args.max_train_pages, len(train_df)), random_state=args.seed)
    print(f"pages: {len(df)} total -> {len(train_df)} train / {len(hold_df)} holdout")
    print(train_df["category"].value_counts().to_string())

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)
    hold_df[["page_id", "doc_id", "category"]].to_csv(os.path.join(args.model_dir, "holdout_page_ids.csv"), index=False)

    # --- model ---
    print(f"loading {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")

    # LoRA on the language model's attention + MLP projections only. The
    # vision tower stays frozen: its module names contain "visual", which
    # the negative lookahead excludes.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()  # needed for gradient checkpointing with frozen embeddings
    model.print_trainable_parameters()

    # --- train ---
    targs = TrainingArguments(
        output_dir=os.path.join(args.output_data_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=PageDataset(train_df, images_dir, doc_ctx, args.max_pixels),
        data_collator=Collator(processor),
    )
    t0 = time.time()
    result = trainer.train()
    train_secs = time.time() - t0
    print(f"training finished in {train_secs / 60:.1f} min; final loss {result.training_loss:.4f}")

    # --- save adapter + processor ---
    model.save_pretrained(args.model_dir)
    processor.save_pretrained(args.model_dir)

    # --- quick score on holdout pages ---
    eval_result = evaluate(model, processor, hold_df, images_dir, doc_ctx, args.max_pixels,
                           args.eval_pages, args.max_new_tokens, args.seed)

    summary = {
        "model_id": args.model_id,
        "train_pages": int(len(train_df)),
        "holdout_pages": int(len(hold_df)),
        "epochs": args.epochs,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_pixels": args.max_pixels,
        "lora_r": args.lora_r,
        "train_minutes": round(train_secs / 60, 1),
        "final_loss": round(float(result.training_loss), 4),
        "holdout_eval": eval_result,
    }
    with open(os.path.join(args.model_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
