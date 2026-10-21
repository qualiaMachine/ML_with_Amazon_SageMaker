"""WattBot RAG helpers shared by the RAG episodes and the batch job script.

The retrieval, prompting, answer parsing, orchestration and scoring logic of
the WattBot pipeline is the same whichever model serves it. This module holds
that logic once, so the episodes differ only in the two model calls they plug
in:

    embed_query(question) -> 1-D numpy vector
    generate(system_prompt, user_prompt, max_new_tokens) -> str

How each place uses it (copy `wattbot_rag.py` next to the notebook, or into
the `code_rag/` directory shipped with a Processing job):

- episodes/11_RAG-Bedrock.md: `embed_query` calls Titan Text Embeddings V2 and
  `generate` calls DeepSeek V3 through `bedrock_runtime.converse`, both via
  tagged application inference profiles. Uses `retrieve_top_k`, `run_single_qa`
  and `compute_wattbot_score`.
- episodes/12_RAG-Processing-jobs.md: the notebook only imports
  `compute_wattbot_score`; the job itself runs `scripts/wattbot_rag_batch.py`.
- scripts/wattbot_rag_batch.py: `embed_query` is a sentence-transformer
  (`embed_query_with`) and `generate` is a vLLM engine serving Qwen3.8-27B.
- episodes/13_RAG-GPU-notebook.md (optional): same as the batch script, but
  interactively, with `with_evidence_type=True` so each explanation is
  prefixed with the kind of evidence it rests on.

Output rows follow the WattBot submission format: id, question, answer,
answer_value, answer_unit, ref_id (";"-joined document ids or "is_blank"),
ref_url, supporting_materials, explanation. `compute_wattbot_score` accepts
ref_id either as ";"-joined ids or as the Python-list strings used in
train_QA.csv ("['wu2021b','patterson2021']").
"""

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BLANK = "is_blank"
UNANSWERABLE_TEXT = "Unable to answer with confidence based on the provided documents."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are WattBot, a question-answering assistant for the energy, water, and carbon "
    "footprint of AI.\n"
    "You must answer questions using ONLY the provided context from scientific papers.\n"
    "If the context does not contain enough information to answer or infer the answer,\n"
    "you must mark the question as unanswerable.\n\n"
    "Respond with a single JSON object and nothing else, with these keys:\n"
    "- answer: natural-language answer, including the numeric value and units if applicable.\n"
    "- answer_value: the bare value with NO units or symbols: a number (1 for true, 0 for false),\n"
    "  a range as [low,high], or a short categorical string; 'is_blank' if unanswerable.\n"
    "- answer_unit: unit string (e.g. kWh, gCO2, %), or 'is_blank' if there is none.\n"
    "- ref_id: list of the document ids (the labels in square brackets in the context)\n"
    "  that support the answer, e.g. ['patterson2021'].\n"
    "- is_blank: true if unanswerable, false otherwise.\n"
    "- supporting_materials: a short quote, or a table/figure pointer, from the context.\n"
)


def build_answer_prompt(question: str, context: str, answer_unit: Optional[str] = None) -> str:
    """User turn for the answer call: the question, an optional unit hint, the context."""
    unit_hint = ""
    if answer_unit and str(answer_unit).strip().lower() not in {"", BLANK, "nan", "none"}:
        unit_hint = f"Express answer_value in this unit: {answer_unit}\n\n"
    return (
        "Use the context below to answer the question. "
        "Return ONLY a JSON object, no extra commentary.\n\n"
        f"Question: {question}\n\n"
        f"{unit_hint}"
        f"Context:\n{context if context.strip() else '[NO CONTEXT FOUND]'}\n"
    )


def explanation_system_prompt(with_evidence_type: bool = False) -> str:
    """System prompt for the second LLM call, which explains how the retrieved
    evidence supports the answer. The with_evidence_type variant also asks the
    model to label the evidence (text, table, figure, ...) in a fixed one-line
    format that the parser downstream relies on."""
    if with_evidence_type:
        return (
            "You are annotating how an answer is supported by a research paper. "
            "You will see a question, an answer, and the supporting material used. "
            "Choose the MAIN type of evidence (one of: text, table, figure, text+table, "
            "text+figure, table+figure) and give a short explanation of at most 100 characters. "
            "Respond on one line in the strict format: evidence_type: explanation"
        )
    return (
        "You are an AI assistant that explains how evidence supports answers about the "
        "energy, water, and carbon footprint of AI models.\n\n"
        "Instructions:\n"
        "- Write 1-3 sentences.\n"
        "- Explain directly how the cited supporting materials justify the answer.\n"
        "- Do NOT include planning text, meta-reasoning, or tags like <reasoning>.\n"
        "- Do NOT start with phrases like 'We need to answer'; just give the explanation."
    )


def build_explanation_prompt(
    question: str,
    answer: str,
    supporting_materials: str,
    ref_ids: Optional[Sequence[str]] = None,
    with_evidence_type: bool = False,
) -> str:
    """User turn for the explanation call."""
    ref_str = ", ".join(ref_ids) if ref_ids else "unknown"
    prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Supporting materials:\n{supporting_materials}\n\n"
        f"Cited document ids: {ref_str}\n\n"
    )
    if with_evidence_type:
        prompt += (
            "Remember:\n"
            "- evidence_type is one of: text, table, figure, text+table, text+figure, table+figure\n"
            "- explanation is at most 100 characters\n"
            "- format: evidence_type: explanation\n"
        )
    else:
        prompt += (
            "In 1-3 sentences, explain how the supporting materials justify the answer. "
            "Be precise but concise."
        )
    return prompt


# ---------------------------------------------------------------------------
# Answer normalisation and JSON parsing
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*$")


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _format_number(val: float) -> str:
    """Plain decimal string: no commas, no scientific notation, no trailing zeros."""
    if float(val).is_integer():
        return str(int(val))
    try:
        return format(Decimal(repr(float(val))).normalize(), "f")
    except InvalidOperation:
        return f"{val:.10g}"


def normalize_answer_value(raw_value: Any) -> str:
    """Normalise answer_value to the WattBot conventions.

    - 'is_blank' for unanswerable / empty values
    - booleans as 1/0
    - plain numbers without units, commas or scientific notation ("1,438 kWh" -> "1438")
    - ranges as [low,high] ("300-1000" -> "[300,1000]"; "[0.02,0.1]" kept as is)
    - categorical strings (e.g. "ML.ENERGY Benchmark") unchanged
    """
    if raw_value is None:
        return BLANK
    if isinstance(raw_value, (bool, np.bool_)):
        return "1" if raw_value else "0"
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        parts = [normalize_answer_value(v) for v in raw_value]
        return "[" + ",".join(parts) + "]"
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        if isinstance(raw_value, float) and np.isnan(raw_value):
            return BLANK
        return _format_number(float(raw_value))

    s = str(raw_value).strip()
    if not s or s.lower() in {"none", "null", "nan", BLANK}:
        return BLANK
    if s.startswith("[") and s.endswith("]"):
        return s
    if s.lower() in {"true", "yes"}:
        return "1"
    if s.lower() in {"false", "no"}:
        return "0"

    m = _RANGE_RE.match(s)
    if m:
        a, b = m.groups()
        a = re.sub(r"\.0+$", "", a)  # "300.0" -> "300" (not rstrip(".0"), which eats real zeros)
        b = re.sub(r"\.0+$", "", b)
        return f"[{a},{b}]"

    cleaned = re.sub(r"[,%$€£]", "", s)
    num = _try_float(cleaned)
    if num is None and " " in cleaned:  # "1438 kWh" -> 1438
        num = _try_float(cleaned.split()[0])
    if num is not None:
        return _format_number(num)
    return s


def split_ref_ids(value: Any) -> List[str]:
    """Turn any ref_id representation into a clean, ordered list of ids.

    Accepts a Python list, a ';'- or ','-separated string, or a Python-list
    string such as "['wu2021b','patterson2021']" (the train_QA.csv format).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [str(x) for x in value]
    else:
        s = str(value).strip()
        if not s or s.lower() in {BLANK, "nan", "none"}:
            return []
        items = re.split(r"[;,]", s)
    out: List[str] = []
    for item in items:
        cleaned = item.strip().strip("[]'\" ").strip()
        if cleaned and cleaned.lower() != BLANK and cleaned not in out:
            out.append(cleaned)
    return out


def parse_answer_json(raw_answer: str, qid: str = "") -> Dict[str, Any]:
    """Extract and normalise the JSON object the answer call was asked to return.

    Falls back to an unanswerable record when the model's output is not valid JSON.
    """
    parsed: Dict[str, Any] = {
        "answer": "",
        "answer_value": BLANK,
        "answer_unit": BLANK,
        "ref_id": [],
        "is_blank": True,
        "supporting_materials": BLANK,
    }
    try:
        first_brace = raw_answer.find("{")
        last_brace = raw_answer.rfind("}")
        json_str = raw_answer[first_brace : last_brace + 1] if first_brace != -1 and last_brace != -1 else raw_answer
        candidate = json.loads(json_str)
        if not isinstance(candidate, dict):
            raise ValueError("model output is not a JSON object")
    except Exception as e:  # noqa: BLE001 - any failure means "unparseable"
        print(f"JSON parse error for question {qid}; defaulting to is_blank. Error: {e}")
        return parsed

    parsed["answer"] = str(candidate.get("answer", "") or "").strip()
    parsed["answer_value"] = normalize_answer_value(candidate.get("answer_value", BLANK))
    parsed["answer_unit"] = str(candidate.get("answer_unit", BLANK) or "").strip() or BLANK
    parsed["ref_id"] = split_ref_ids(candidate.get("ref_id", []))
    is_blank = candidate.get("is_blank", False)
    if isinstance(is_blank, str):
        is_blank = is_blank.strip().lower() in {"true", "1", "yes"}
    parsed["is_blank"] = bool(is_blank)
    parsed["supporting_materials"] = str(candidate.get("supporting_materials", BLANK) or "").strip() or BLANK
    return parsed


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between the rows of a (m x d) and the rows of b (n x d)."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def retrieve_top_k(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, Any]],
    k: int = 8,
) -> List[Dict[str, Any]]:
    """Return the k chunks whose embeddings are most similar to the query embedding."""
    chunk_embeddings = np.asarray(chunk_embeddings, dtype="float32")
    if chunk_embeddings.shape[0] == 0:
        return []
    sims = cosine_similarity_matrix(np.asarray(query_embedding, dtype="float32").reshape(1, -1), chunk_embeddings)[0]
    top_idx = np.argsort(-sims)[:k]
    results = []
    for idx in top_idx:
        ch = chunked_docs[idx]
        results.append(
            {
                "score": float(sims[idx]),
                "text": ch["text"],
                "doc_id": ch.get("doc_id", ""),
                "title": ch.get("title", ""),
                "url": ch.get("url", ""),
                "page_num": ch.get("page_num", None),
                "page_label": ch.get("page_label", None),
            }
        )
    return results


def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Compact context string: one '[doc_id, page N]: text' line per chunk."""
    lines = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        label = ch.get("doc_id") or f"chunk_{i}"
        page = ch.get("page_label", ch.get("page_num", ""))
        txt = ch["text"].replace("\n", " ")
        lines.append(f"[{label}, page {page}]: {txt}")
    return "\n".join(lines)


def build_docid_to_url(metadata_df: pd.DataFrame) -> Dict[str, str]:
    """metadata.csv -> {doc_id: url} for filling the ref_url column."""
    docid_to_url: Dict[str, str] = {}
    for _, row in metadata_df.iterrows():
        doc_id = str(row.get("id", "")).strip()
        url = row.get("url", "")
        if doc_id and isinstance(url, str) and url.strip():
            docid_to_url[doc_id] = url.strip()
    return docid_to_url


def embed_query_with(embedder) -> Callable[[str], np.ndarray]:
    """Wrap an embedder with a sentence-transformers-style `encode` as an `embed_query(question)` callable."""

    def embed_query(question: str) -> np.ndarray:
        """One question -> one unit-length vector, with the same model that embedded the chunks."""
        return embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    return embed_query


# ---------------------------------------------------------------------------
# The two LLM phases and the per-question orchestration
# ---------------------------------------------------------------------------

Generate = Callable[[str, str, int], str]


def answer_phase_for_question(
    qid: str,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    generate: Generate,
    answer_unit: Optional[str] = None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """First LLM call: answer JSON (answer, answer_value, ref_id, is_blank, supporting_materials)."""
    context = format_context_for_prompt(retrieved_chunks)
    raw_answer = generate(ANSWER_SYSTEM_PROMPT, build_answer_prompt(question, context, answer_unit), max_new_tokens)
    return parse_answer_json(raw_answer, qid=qid)


def explanation_phase_for_question(
    qid: str,
    question: str,
    answer: str,
    supporting_materials: str,
    generate: Generate,
    ref_ids: Optional[Sequence[str]] = None,
    with_evidence_type: bool = False,
    max_new_tokens: int = 256,
) -> str:
    """Second LLM call: a short explanation, optionally prefixed with '[evidence_type]'."""
    raw = generate(
        explanation_system_prompt(with_evidence_type),
        build_explanation_prompt(question, answer, supporting_materials, ref_ids, with_evidence_type),
        max_new_tokens,
    ).strip()
    if not with_evidence_type:
        return raw

    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    if ":" in first_line:
        etype, expl = first_line.split(":", 1)
        evidence_type = etype.strip().strip("[]").lower() or "other"
        explanation = expl.strip()
    else:
        evidence_type, explanation = "other", first_line
    return f"[{evidence_type}] {explanation[:100]}"


def run_single_qa(
    row,
    embed_query: Callable[[str], np.ndarray],
    generate: Generate,
    chunk_embeddings: np.ndarray,
    chunked_docs: List[Dict[str, Any]],
    docid_to_url: Dict[str, str],
    top_k: int = 8,
    retrieval_threshold: float = 0.25,
    with_evidence_type: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Retrieve, answer, decide answerability, explain: one WattBot row in, one submission row out.

    `row` is a row of train_QA.csv (needs `id` and `question`; `answer_unit` is used
    as a hint if present). A question is marked unanswerable when the model says so
    OR the best retrieval cosine similarity is below `retrieval_threshold`.
    """
    qid = row["id"]
    question = row["question"]
    answer_unit = row.get("answer_unit", BLANK) if hasattr(row, "get") else BLANK
    if not isinstance(answer_unit, str) or not answer_unit.strip():
        answer_unit = BLANK

    retrieved = retrieve_top_k(embed_query(question), chunk_embeddings, chunked_docs, k=top_k)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    parsed = answer_phase_for_question(qid, question, retrieved, generate, answer_unit=answer_unit)

    if parsed["is_blank"] and verbose:
        print(f"[diag][{qid}] model returned is_blank")
    elif top_score < retrieval_threshold and verbose:
        print(f"[diag][{qid}] retrieval blocked: top cosine={top_score:.3f} < threshold={retrieval_threshold:.3f}")
    is_blank = parsed["is_blank"] or (top_score < retrieval_threshold)

    if is_blank:
        return {
            "id": qid,
            "question": question,
            "answer": UNANSWERABLE_TEXT,
            "answer_value": BLANK,
            "answer_unit": answer_unit,
            "ref_id": BLANK,
            "ref_url": BLANK,
            "supporting_materials": BLANK,
            "explanation": BLANK,
        }

    ref_ids = parsed["ref_id"]
    urls = [docid_to_url[rid] for rid in ref_ids if rid in docid_to_url]
    explanation = explanation_phase_for_question(
        qid,
        question,
        parsed["answer"],
        parsed["supporting_materials"],
        generate,
        ref_ids=ref_ids,
        with_evidence_type=with_evidence_type,
    )
    return {
        "id": qid,
        "question": question,
        "answer": parsed["answer"],
        "answer_value": parsed["answer_value"],
        "answer_unit": answer_unit,
        "ref_id": ";".join(ref_ids) if ref_ids else BLANK,
        "ref_url": ";".join(urls) if urls else BLANK,
        "supporting_materials": parsed["supporting_materials"],
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Scoring (WattBot metric: 0.75 answer_value + 0.15 ref_id Jaccard + 0.10 is_NA)
# ---------------------------------------------------------------------------


def _to_bool_flag(x) -> bool:
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes"}:
            return True
        if s in {"0", "false", "no", "", BLANK}:
            return False
    if isinstance(x, float) and np.isnan(x):
        return False
    return bool(x)


def _parse_float_or_none(x) -> Optional[float]:
    return _try_float(str(x).strip())


def _answer_value_correct(gt_val, pred_val, rel_tol: float = 1e-3) -> bool:
    """Numeric within 0.1% relative tolerance; otherwise case-insensitive string match."""
    gt_str = str(gt_val).strip()
    pred_str = str(pred_val).strip()
    if gt_str.lower() == BLANK or pred_str.lower() == BLANK:
        return gt_str.lower() == pred_str.lower()
    gt_num = _parse_float_or_none(gt_str)
    pred_num = _parse_float_or_none(pred_str)
    if gt_num is not None and pred_num is not None:
        if gt_num == 0:
            return abs(pred_num - gt_num) <= rel_tol
        return abs(pred_num - gt_num) / max(abs(gt_num), 1e-12) <= rel_tol
    return gt_str.lower() == pred_str.lower()


def _ref_id_jaccard(gt_ref, pred_ref) -> float:
    """Jaccard overlap of the two id sets, case-insensitive.

    Ground truth comes as Python-list strings ("['wu2021b','patterson2021']"),
    predictions as ';'-joined ids; both parse through `split_ref_ids`.
    """
    gt_set = {r.lower() for r in split_ref_ids(gt_ref)}
    pred_set = {r.lower() for r in split_ref_ids(pred_ref)}
    if not gt_set and not pred_set:
        return 1.0
    union = gt_set | pred_set
    return len(gt_set & pred_set) / len(union) if union else 0.0


def _na_flags(merged: pd.DataFrame, col: Optional[str], answer_col: str, suffix: str) -> pd.Series:
    """NA flag per row. col="is_blank" (or None) derives it from answer_value; another name reads that column."""
    if col is not None and col.lower() != BLANK:
        for candidate in (f"{col}{suffix}", col):
            if candidate in merged.columns:
                return merged[candidate].map(_to_bool_flag)
        print(f"Column {col!r} not found; deriving NA flags from {answer_col}{suffix} == 'is_blank'.")
    return merged[f"{answer_col}{suffix}"].astype(str).str.strip().str.lower().eq(BLANK)


def compute_wattbot_score(
    train_qa_path: str = "train_QA.csv",
    preds_path: str = "wattbot_solutions.csv",
    id_col: str = "id",
    gt_answer_col: str = "answer_value",
    gt_ref_col: str = "ref_id",
    gt_is_na_col: Optional[str] = BLANK,
    pred_answer_col: str = "answer_value",
    pred_ref_col: str = "ref_id",
    pred_is_na_col: Optional[str] = BLANK,
    n_examples: int = 10,
) -> pd.DataFrame:
    """Score a predictions CSV against train_QA.csv with the WattBot metric.

    Rows are joined on `id_col`. The NA component compares an unanswerable flag per
    side; with the default "is_blank" that flag is derived from answer_value ==
    "is_blank" (train_QA.csv has no separate column), or pass the name of a real
    column. Prints the mean of each component and up to `n_examples` of the
    worst-scoring rows, and returns the merged DataFrame with per-row scores.
    """
    gt = pd.read_csv(train_qa_path)
    preds = pd.read_csv(preds_path)
    merged = gt.merge(preds, on=id_col, suffixes=("_gt", "_pred"))
    if merged.empty:
        raise ValueError("No overlapping ids between ground truth and predictions.")

    gt_is_na = _na_flags(merged, gt_is_na_col, gt_answer_col, "_gt")
    pred_is_na = _na_flags(merged, pred_is_na_col, pred_answer_col, "_pred")

    merged["answer_score"] = [
        1.0 * _answer_value_correct(g, p)
        for g, p in zip(merged[f"{gt_answer_col}_gt"], merged[f"{pred_answer_col}_pred"])
    ]
    merged["ref_id_score"] = [
        _ref_id_jaccard(g, p) for g, p in zip(merged[f"{gt_ref_col}_gt"], merged[f"{pred_ref_col}_pred"])
    ]
    merged["is_NA_score"] = (gt_is_na.values == pred_is_na.values).astype(float)
    merged["wattbot_score"] = 0.75 * merged["answer_score"] + 0.15 * merged["ref_id_score"] + 0.10 * merged["is_NA_score"]

    print(f"Rows compared: {len(merged)}")
    print(f"Mean answer_value score: {merged['answer_score'].mean():.4f}")
    print(f"Mean ref_id score:       {merged['ref_id_score'].mean():.4f}")
    print(f"Mean is_NA score:        {merged['is_NA_score'].mean():.4f}")
    print(f"Overall WattBot score:   {merged['wattbot_score'].mean():.4f}")

    incorrect = merged[merged["wattbot_score"] < 1.0]
    if not incorrect.empty and n_examples > 0:
        print(f"\nExamples of incorrect / partially correct responses (up to {n_examples} rows):\n")
        q_col = "question_gt" if "question_gt" in merged.columns else ("question" if "question" in merged.columns else None)
        for _, row in incorrect.sort_values("wattbot_score").head(n_examples).iterrows():
            print("-" * 80)
            print(f"id: {row[id_col]}")
            if q_col:
                print(f"Question: {row[q_col]}")
            print(f"GT answer_value:   {row[f'{gt_answer_col}_gt']}")
            print(f"Pred answer_value: {row[f'{pred_answer_col}_pred']}")
            print(f"GT ref_id:         {row[f'{gt_ref_col}_gt']}")
            print(f"Pred ref_id:       {row[f'{pred_ref_col}_pred']}")
            print(
                f"answer_score: {row['answer_score']:.3f}, ref_id_score: {row['ref_id_score']:.3f}, "
                f"is_NA_score: {row['is_NA_score']:.3f}, wattbot_score: {row['wattbot_score']:.3f}"
            )
        print("-" * 80)
    return merged
