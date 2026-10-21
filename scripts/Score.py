"""WattBot Score — the metric for the WattBot 2026 leaderboard.

A weighted accuracy in [0, 1], averaged over questions, of four fields per answer:

    answer_value   0.75   numeric within +-0.1%, categorical exact after normalization,
                          'is_blank' when the corpus cannot answer the question
    ref_id         0.20   citation F1 against the gold citation set
    is_NA          0.05   a refusal must blank every evidence field, not just the value
    answer_unit    0.00   given in test_Q.csv, so reported but not scored

Answering a question the corpus cannot support scores zero on answer_value AND on is_NA:
guessing is strictly worse than abstaining. Full rules are in score()'s docstring, which
is what Kaggle shows participants on the Evaluation tab.

The per-evidence-type breakdown reads the Quote/Table/Figure/Math/is_NA/CrossPaper/
Reconcile columns straight off the solution frame. If the uploaded solution ever loses
them, the headline score keeps working and the breakdown silently goes empty.

Sanity check after any change: ground truth scored against itself must print exactly 1.000.
"""

import pandas as pd, json, math, re

# Default for import-time use (Kaggle will override in __main__)
IS_KAGGLE = False

# Which metric the 0.20 ref_id component uses for the headline score.
# "f1" (default for 2026) or "jaccard" (the 2025 behaviour). Precision, recall and F1
# are reported either way.
#
# Switched to F1 because making citation sets exhaustive made Jaccard punitive: gold sets
# grew (some questions list four documents that each state the answer) while a good system
# still cites one or two. Citing one correct source out of four pays 0.25 under Jaccard
# but 0.40 under F1. The "keep Jaccard for leaderboard continuity" argument no longer
# applies either — the 2026 weights, abstention rule and unit scoring already make scores
# incomparable to 2025.
REF_SCORE = "f1"

# Per-evidence-type breakdown from the most recent score() call.
#
# Kaggle's metric contract requires score() to return "a single, finite, non-null
# float", so the per-type error rates CANNOT ride along on the return value and cannot
# become extra leaderboard columns. What the 2025 metric was missing is that the
# solution's evidence-flag columns (Quote/Table/Figure/Math/is_NA) are passed to the
# metric at all — every solution column except Usage is — so the breakdown is now
# computable inside score(). It is surfaced three ways, none of which touch the return:
#   * printed (host-visible in the metric's validation output)
#   * Score.LAST_BREAKDOWN, for a notebook to render after calling score()
#   * participants running this file locally against train_QA.csv
LAST_BREAKDOWN: dict = {}

# ───────────────────────────────────────────────────────── helpers ──────
class ParticipantVisibleError(Exception):
    '''Shown to competitors when their CSV is malformed.'''
    pass

def _s(x):
    return "" if x is None else str(x)

# Unit synonyms. Merges only genuine spelling variants — NEVER across scale (kWh vs MWh)
# and NEVER across rate (MWh vs "MWh per year"), which are different answers. Erring
# toward merging is the safe direction: an over-merge is lenient, an under-merge marks a
# correct answer wrong.
_UNIT_SYNONYMS = {
    "%": "percent", "pct": "percent",
    "x": "multiplier", "time": "multiplier", "fold": "multiplier", "ratio": "multiplier",
    "factor": "multiplier",
    "l": "liter", "litre": "liter",
    "$": "usd", "dollar": "usd", "us dollar": "usd",
    "lb": "lbs", "pound": "lbs",
    "kilogram": "kg",
    "tonne": "metric ton", "metric tonne": "metric ton", "t": "metric ton",
    "tco2e": "tco2e", "ton co2e": "tco2e", "metric ton co2e": "tco2e",
    "gpu": "gpus",
    "watt": "w",
}


def canon_unit(u) -> str:
    """Canonical form for unit comparison: case, punctuation, plurals and synonyms."""
    s = _s(u).strip().lower()
    if not s:
        return ""
    s = re.sub(r"[.·]", " ", s)          # "Fig." style dots, middots
    s = re.sub(r"\s+", " ", s).strip()
    # de-pluralise the final word only ("V100_32GB_GPUs" -> "..._gpu", "years" -> "year")
    if len(s) > 2 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return _UNIT_SYNONYMS.get(s, s)

def canon_text(x) -> str:
    """Normalize a categorical answer for comparison.

    Case, surrounding whitespace and PUNCTUATION are not the thing being tested. A paper
    that prints "Backblaze (B2)" and a bank that records "Backblaze B2" are the same
    answer, and a contestant who quotes the source verbatim should not lose a point to a
    pair of parentheses. Internal word order and wording still matter, so this does not
    make a wrong entity right.

    >>> canon_text("Backblaze (B2)") == canon_text("Backblaze B2")
    True
    >>> canon_text("Jevons' Paradox") == canon_text("jevons paradox")
    True
    >>> canon_text("V100") == canon_text("V100 or A30")
    False
    >>> canon_text("the watershed") == canon_text("Watershed")
    True
    """
    s = _s(x).lower()
    s = re.sub(r"[^\w\s]", " ", s)      # drop punctuation, keep word characters
    s = re.sub(r"\s+", " ", s).strip()
    # A leading article is phrasing, not content: "the watershed" and "watershed" are the
    # same answer. Internal articles still matter ("a" in a model name stays).
    return re.sub(r"^(?:the|a|an)\s+", "", s)


def strip_unit_echo(val, unit) -> str:
    """The submitted value with the expected unit's words removed.

    test_Q.csv hands contestants the unit, so a value that echoes it back — "Scope 2"
    when the unit column says "scope", "43 percent" when it says "percent" — is the same
    answer as the bare value. Stripping the unit tokens and re-trying the numeric
    comparison scores the content, not the phrasing. Returns "" when nothing is left,
    which callers must treat as "no rescue" rather than a match.

    >>> strip_unit_echo("Scope 2", "scope")
    '2'
    >>> strip_unit_echo("43 percent", "percent")
    '43'
    >>> strip_unit_echo("watershed", "is_blank")
    'watershed'
    """
    u = _s(unit).strip().lower()
    if not u or u in ("is_blank", "nan", ""):
        return _s(val).strip()
    v = canon_text(val)
    for tok in re.split(r"[\s/]+", u):
        if tok and re.search(r"[a-z]", tok):
            v = re.sub(rf"\b{re.escape(tok)}\b", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def canon_refs(r):
    """Return sorted, lowercased list of refs. Robust to NaN/singletons."""
    s = _s(r).strip()
    if not s:
        return []
    try:
        if s.startswith("["):
            arr = json.loads(s.replace("'", '"'))
            return sorted([_s(x).strip().lower() for x in arr])
    except Exception:
        pass
    return [s.lower()]

def ref_overlap_score(sol_ref, sub_ref):
    """
    Jaccard overlap on normalized ref-id sets.
    - Exact match → 1.0
    - Partial overlap → (|A∩B| / |A∪B|)
    - Both empty → 1.0
    - One empty, one non-empty → 0.0
    """
    A = set(canon_refs(sol_ref))
    B = set(canon_refs(sub_ref))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


def ref_prf(sol_ref, sub_ref):
    """
    Citation precision / recall / F1 on normalized ref-id sets (gold = solution).
    Diagnostic counterpart to the Jaccard score: precision penalizes citing the
    wrong/extra paper, recall penalizes missing a required one — distinctions
    Jaccard collapses. Especially informative for multi-paper questions.
    - Both empty → (1, 1, 1); one empty, one not → (0, 0, 0).
    """
    A = set(canon_refs(sol_ref))   # gold
    B = set(canon_refs(sub_ref))   # predicted
    if not A and not B:
        return 1.0, 1.0, 1.0
    if not A or not B:
        return 0.0, 0.0, 0.0
    inter = len(A & B)
    p = inter / len(B)
    r = inter / len(A)
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1

# tokens that mean “no answer supplied”
_BLANK_TOKENS = {"", "na", "n/a", "is_blank"}

_FALLBACK_PHRASE = "Unable to answer with confidence based on the provided documents"

def is_blank(x) -> bool:
    val = _s(x).strip().lower()
    return val in _BLANK_TOKENS or val.startswith(_FALLBACK_PHRASE.lower())

def both_numeric(a: str, b: str) -> bool:
    '''Quick check before math.isclose to avoid ValueError spam.'''
    try:
        float(a); float(b)
        return True
    except ValueError:
        return False

def listish_kind(x) -> str | None:
    """Which collection syntax a value uses: 'range', 'list', 'set', or None.

    Two are gold forms, one is not:
      [lo, hi]  RANGE  — any single value between lo and hi is correct
      (a, b)    LIST   — ALL of these are required; a single value is NOT enough
      {a, b}    SET    — unordered terms. Parsed so a submission written this way is read
                        as a collection rather than a raw string; NEVER use it for gold.
                        It compares as text, so "0.18" and "0.180" do not match.

    The distinction exists because a range and a list look identical but score opposite.
    A Reconcile question whose two source values are 0.18 and 3.1 must NOT be coded
    [0.18, 3.1]: that is a range, and it marks 0.18, 3.1 and everything between them
    correct — including the single-source answers the question exists to catch.
    """
    s = _s(x).strip()
    if s.startswith("[") and s.endswith("]"):
        return "range"
    if s.startswith("(") and s.endswith(")"):
        return "list"
    if s.startswith("{") and s.endswith("}"):
        return "set"
    return None


def parse_listish(x):
    """Parse list/set-like strings into Python lists.
    - [a, b] : range (or a categorical list, when not two numbers)
    - (a, b) : list — every element required
    - {a, b} : set of terms, order-insensitive, lowercase
    Returns None if not a collection.
    """
    s = _s(x).strip()
    if not s:
        return None

    # square bracket (range) or parenthesised (list) — parsed the same, scored differently
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            arr = json.loads("[" + s[1:-1] + "]".replace("'", '"')) if s[0] == "(" \
                else json.loads(s.replace("'", '"'))
        except Exception:
            arr = [t.strip() for t in s[1:-1].split(",")]
        out = []
        for t in arr:
            t = _s(t).strip()
            try:
                out.append(float(t))
            except ValueError:
                out.append(t.lower())
        return out

    # curly brace set of terms
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        terms = [t.strip().lower() for t in inner.split(",") if t.strip()]
        return sorted(set(terms))  # normalized set

    return None

def compare_lists(a, b, tol=1e-3):
    """Order-insensitive list equality, numeric with tolerance."""
    a_nums = all(isinstance(v, (int, float)) for v in a)
    b_nums = all(isinstance(v, (int, float)) for v in b)
    if a_nums and b_nums:
        A = sorted(float(x) for x in a)
        B = sorted(float(x) for x in b)
        if len(A) != len(B):
            return False
        return all(math.isclose(x, y, rel_tol=tol, abs_tol=tol) for x, y in zip(A, B))
    # string-ish compare
    A = sorted(_s(x).strip().lower() for x in a)
    B = sorted(_s(x).strip().lower() for x in b)
    return A == B

def row_bits(sol, sub, tol: float = 1e-3):
    """
    Per-row rubric components:
      val / unit / ref / na → booleans/floats used later for weighting.

    Changes:
    - ref: now fractional in [0,1] instead of binary (see REF_SCORE at the top of the
      file for which set measure is applied).
    """
    # NA component — "did you refuse CLEANLY?"
    # The documented rule has always been that an unanswerable question requires ALL
    # relevant fields blanked, but through 2025 the code checked only answer_value. It now
    # matches the documentation, which also stops this component from being a carbon copy
    # of `val`: `val` asks whether the system refused, `na` asks whether it refused
    # without also inventing a citation and a quote — the hedged bluff.
    # `answer_unit` is deliberately NOT required to be blank: the bank keeps the
    # semantic unit on unanswerable rows (e.g. "percent") to record what was asked,
    # so demanding a blank there would mark the ground truth wrong against itself.
    # The fields that matter for a hedged bluff are the evidence ones.
    if is_blank(sol["answer_value"]):
        na_ok = all(is_blank(sub.get(f)) for f in
                    ("answer_value", "ref_id", "ref_url", "supporting_materials")
                    if f in sub)
    else:
        na_ok = True  # not a 'NA' case, don't penalize

    # value component (supports bracketed lists/ranges)
    if is_blank(sol["answer_value"]):
        # Unanswerable question. Abstaining is the correct answer and earns the value
        # component; inventing one is WRONG and earns nothing. (Before 2026 this branch
        # returned True unconditionally, so a confident hallucination still banked the
        # full 0.75 weight and forfeited only the 0.10 NA component — a ~1.4-point
        # penalty for the failure mode the benchmark exists to catch.)
        val_ok = is_blank(sub["answer_value"])
    else:
        sol_list = parse_listish(sol["answer_value"])
        sub_list = parse_listish(sub["answer_value"])

        sol_kind = listish_kind(sol["answer_value"])

        if sol_list is not None:
            if sub_list is not None:
                # list-to-list comparison (order-insensitive, numeric tol)
                val_ok = compare_lists(sol_list, sub_list, tol=tol)
            elif sol_kind == "list":
                # (a, b) requires EVERY value. A single number is not a partial answer here,
                # it is the single-source answer a Reconcile question is built to reject.
                val_ok = False
            else:
                # If solution is a numeric 2-tuple (treat as range), allow single number within range
                if len(sol_list) == 2 and all(isinstance(v, (int, float)) for v in sol_list):
                    lo, hi = sorted(sol_list)
                    try:
                        x = float(sub["answer_value"])
                        val_ok = (lo - tol) <= x <= (hi + tol)
                    except ValueError:
                        val_ok = False
                else:
                    # solution wanted a categorical list; submission didn't provide a list
                    val_ok = False
        else:
            if both_numeric(sol["answer_value"], sub["answer_value"]):
                val_ok = math.isclose(
                    float(sol["answer_value"]),
                    float(sub["answer_value"]),
                    rel_tol=tol, abs_tol=tol
                )
            else:
                # Before treating it as free text, try removing the expected unit's words
                # from both sides: a value echoing the unit it was told to use is a
                # formatting choice, not a different answer.
                sol_bare = strip_unit_echo(sol["answer_value"], sol["answer_unit"])
                sub_bare = strip_unit_echo(sub["answer_value"], sol["answer_unit"])
                if sol_bare and sub_bare and both_numeric(sol_bare, sub_bare):
                    val_ok = math.isclose(float(sol_bare), float(sub_bare),
                                          rel_tol=tol, abs_tol=tol)
                else:  # named entity or free-text (normalized)
                    val_ok = canon_text(sol["answer_value"]) == canon_text(sub["answer_value"])

    # unit + ref components (case/whitespace insensitive)
    unit_ok = canon_unit(sol["answer_unit"]) == canon_unit(sub["answer_unit"])
    ref_score  = ref_overlap_score(sol["ref_id"], sub["ref_id"])  # ← fractional

    return {"val": val_ok, "unit": unit_ok, "ref": ref_score, "na": na_ok}

# ───────────────────────────────────────────────── metric entry point ──
def score(solution: pd.DataFrame,
          submission: pd.DataFrame,
          row_id_column_name: str,      # no default: Kaggle rejects the metric if it has one
          verbose: bool = True) -> float:
    """
    WattBot Score (0–1)

    Submissions are scored with a custom WattBot Score that evaluates four fields
    for every question and returns a weighted accuracy:

    Component      Weight  What counts as correct
    ------------------------------------------------------------
    answer_value    0.75   Matches the ground truth. Numeric answers must be within
                           ±0.1% relative tolerance (and ~1e-3 absolute fallback);
                           categorical values must match exactly after normalization.
                           If a question is unanswerable, this column must contain
                           'is_blank'.

    answer_unit     0.00   Reported but not scored: the expected unit is given to you in
                           test_Q.csv, so scoring it would be free marks. It is given
                           because 30% of questions do not state their unit in the question
                           text, and withholding it would make the ANSWER ambiguous.

    ref_id          0.20   **Partial credit via citation F1** between your ref_id set and
                           the ground-truth set (order ignored, case-insensitive). Gold
                           citation sets are exhaustive, so recall alone would punish a
                           team for citing one correct source instead of all four;
                           precision alone would not punish citing everything. Use
                           'is_blank' if no evidence is available.

    is_NA (NA)      0.05   For truly unanswerable questions your submission must mark
                           answer_value, ref_id, ref_url and supporting_materials as
                           'is_blank'. Refusing in answer_value while still supplying a
                           citation or a quote — a hedged bluff — scores 0 here.
                           (answer_unit may carry the expected unit.) The penalty for
                           answering outright lives in answer_value, above.

    CHANGED FOR 2026 — abstention is scored. Some questions cannot be answered from the
    corpus. Through 2025 the answer_value component was credited unconditionally on those
    questions ("nothing expected"), so inventing an answer forfeited only the 0.10 NA
    component. Answering an unanswerable question now scores ZERO on answer_value as well.
    If the evidence is not in the corpus, put 'is_blank' in answer_value: guessing is
    strictly worse than abstaining.

    Notes:
    - Output: returns the final score as a float in [0, 1]. It also prints per-component
      accuracies, citation precision/recall/F1, the false-answer rate on unanswerable
      questions, and a % wrong breakdown by evidence type (Quote/Table/Figure/Math/is_NA).
      Those diagnostics are printed only — Kaggle's metric contract requires a single
      float return, so they cannot become leaderboard columns. Participants can reproduce
      the breakdown by running this file locally against train_QA.csv.

    >>> import pandas as pd
    >>> sol = pd.DataFrame({
    ...     "id": ["q1", "q2"],
    ...     "answer_value": ["42.5", "is_blank"],
    ...     "answer_unit": ["MWh", "is_blank"],
    ...     "ref_id": ["['somepaper2024']", "is_blank"],
    ...     "Quote": [1, 0], "Table": [0, 0], "Figure": [0, 0],
    ...     "Math": [0, 0], "is_NA": [0, 1]})
    >>> sub = sol.copy()
    >>> sub["explanation"] = "derived from the cited table"
    >>> score(sol.copy(), sub.copy(), "id", verbose=False)
    1.0

    Answering the unanswerable question q2 loses both the value and the NA components:

    >>> sub.loc[1, "answer_value"] = "42"
    >>> round(score(sol.copy(), sub.copy(), "id", verbose=False), 4)
    0.6
    """

    # Required columns: solution doesn't need 'explanation'; submission does.
    required_sol = [row_id_column_name, "answer_value", "answer_unit", "ref_id"]
    required_sub = [row_id_column_name, "answer_value", "answer_unit", "ref_id", "explanation"]
    payload_cols = ["answer_value", "answer_unit", "ref_id"]  # used after set_index

    miss_sol = set(required_sol) - set(solution.columns)
    if miss_sol:
        raise ParticipantVisibleError(f"solution missing columns: {sorted(miss_sol)}")

    miss_sub = set(required_sub) - set(submission.columns)
    if miss_sub:
        raise ParticipantVisibleError(f"submission missing columns: {sorted(miss_sub)}")

    # Normalize to strings for stability (only columns that exist)
    for col in required_sol:
        solution[col] = solution[col].astype(str)
    for col in required_sub:
        submission[col] = submission[col].astype(str)

    if submission["explanation"].str.strip().eq("").any():
        raise ParticipantVisibleError("Each row needs a non-empty explanation.")

    # Index on the row id column (Kaggle passes its name in; it defaults to "id" locally)
    sol = solution.set_index(row_id_column_name)
    sub = submission.set_index(row_id_column_name)

    # IMPORTANT: don't try to select 'id' as a column after set_index
    merged = sol.join(sub[payload_cols], lsuffix="_sol", rsuffix="_sub", how="inner")

    if merged.empty:
        raise ParticipantVisibleError("No matching IDs between submission and solution.")

    bits = merged.apply(
        lambda r: row_bits(
            sol=r.filter(like="_sol").rename(lambda c: c[:-4]),
            sub=r.filter(like="_sub").rename(lambda c: c[:-4])
        ),
        axis=1, result_type="expand"
    )

    comp = bits.mean(numeric_only=True)

    # Citation precision / recall / F1 (diagnostic; gold = solution ref-ids).
    prf = merged.apply(
        lambda r: ref_prf(r["ref_id_sol"], r["ref_id_sub"]),
        axis=1, result_type="expand"
    )
    ref_p, ref_r, ref_f1 = (float(prf[0].mean()), float(prf[1].mean()), float(prf[2].mean()))

    # False-answer rate on unanswerable questions (diagnostic; headline score unchanged).
    # row_bits credits the full 0.75 value component whenever the gold answer_value is
    # blank ("nothing expected"), so a system that invents an answer to an unanswerable
    # question still banks that weight and forfeits only the 0.10 NA component. This
    # sub-metric makes the behaviour visible without changing the leaderboard; see
    # docs/PLAN.md for the open decision on whether to reweight.
    na_mask = merged["answer_value_sol"].apply(is_blank)
    n_unanswerable = int(na_mask.sum())
    false_answer_rate = (
        float(merged.loc[na_mask, "answer_value_sub"].apply(lambda v: not is_blank(v)).mean())
        if n_unanswerable else float("nan")
    )

    ref_component = ref_f1 if REF_SCORE == "f1" else comp["ref"]
    # 2026 weights. Rationale:
    #  - value stays dominant, but 0.75 was crowding out everything the benchmark claims
    #    to be about.
    #  - ref 0.15 -> 0.20: this is a *citation* benchmark, and citation sets are now
    #    exhaustive, so the component is fair to weight more heavily.
    #  - unit stays at 0.00, but for a better reason than 2025 had. The hole it would
    #    close ("1287 kWh" scoring like "1287 MWh") is already closed by publishing the
    #    expected unit in test_Q.csv, which questions need anyway: 30% of them do not
    #    state their unit in the question text, so withholding it would make the ANSWER
    #    ambiguous, not just the unit. Given away, the unit cannot also be scored — that
    #    would hand every team a free 0.039. canon_unit still normalises for the
    #    diagnostics, and the component is reported.
    #  - na 0.10 -> 0.05: it no longer carries the abstention penalty (that lives in
    #    `val` now); it checks that a refusal is clean rather than a hedged bluff.
    overall = (
        0.75*comp["val"] +
        0.00*comp["unit"] +
        0.20*ref_component + # ref_id credit (REF_SCORE selects Jaccard or F1; F1 for 2026)
        0.05*comp["na"]
    )

    # ----- Component accuracy / score
    def emit(msg: str = "") -> None:
        """Diagnostics go to stdout (host-visible on Kaggle); silenced when verbose=False."""
        if verbose:
            print(msg)

    emit("Component scores (means):")
    emit(f"  value match  : {comp['val']:.3f}")
    emit(f"  unit match   : {comp['unit']:.3f}")
    emit(f"  ref overlap  : {comp['ref']:.3f}  (Jaccard; scored component uses "
         f"{'F1' if REF_SCORE == 'f1' else 'Jaccard'})")
    emit(f"  ref precision: {ref_p:.3f}")
    emit(f"  ref recall   : {ref_r:.3f}")
    emit(f"  ref F1       : {ref_f1:.3f}")
    emit(f"  NA agreement : {comp['na']:.3f}")
    if n_unanswerable:
        emit(f"  false answers: {false_answer_rate:.3f}  "
              f"(share of the {n_unanswerable} unanswerable questions given an answer anyway)")
    emit(f"OVERALL SCORE  : {overall:.3f}  (ref component = {REF_SCORE})\n")

    # ----- Per-evidence-type breakdown (multi-label; SOLUTION ONLY, NO FALLBACK)
    # The breakdown is a diagnostic, not part of the score. Kaggle always passes these
    # columns (every solution column but Usage), but a local run against a file without
    # them should still return a score rather than fail.
    # CrossPaper and Reconcile are the two the 2026 bank was built to add, and they are the
    # two it scores worst on — leaving them out of the breakdown hid that.
    required_flags = ["Quote", "Table", "Figure", "Math", "is_NA", "CrossPaper", "Reconcile"]
    # Drop only the flags that are actually absent. The old all-or-nothing guard threw away
    # the entire breakdown over one missing column, so adding a flag here would have
    # silently blanked the table for any solution file predating it.
    missing_flags = [c for c in required_flags if c not in sol.columns]
    if missing_flags:
        emit(f"[note] breakdown omits {sorted(missing_flags)}: not in the solution file\n")
        required_flags = [c for c in required_flags if c in sol.columns]

    def _truthy(x):
        s = str(x).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    # A question may carry several evidence flags (e.g. Table + Math), so these
    # buckets overlap by design and do not sum to the question count.
    per_type = {}
    for label in required_flags:
        mask = sol.loc[bits.index, label].apply(_truthy)
        n = int(mask.sum())
        if not n:
            continue
        wrong = int((~bits.loc[mask, "val"].astype(bool)).sum())
        per_type[label] = {
            "n": n,
            "wrong": wrong,
            "pct_wrong": 100.0 * wrong / n,
            "score": float(
                0.75 * bits.loc[mask, "val"].mean()
                + 0.15 * bits.loc[mask, "ref"].mean()
                + 0.10 * bits.loc[mask, "na"].mean()
            ),
        }
    n_all = int(len(bits))
    wrong_all = int((~bits["val"].astype(bool)).sum())
    per_type["ALL"] = {"n": n_all, "wrong": wrong_all,
                       "pct_wrong": 100.0 * wrong_all / n_all if n_all else float("nan"),
                       "score": float(overall)}

    global LAST_BREAKDOWN
    LAST_BREAKDOWN = {
        "per_type": per_type,
        "false_answer_rate": false_answer_rate,
        "n_unanswerable": n_unanswerable,
        "ref_precision": ref_p, "ref_recall": ref_r, "ref_f1": ref_f1,
        "overall": float(overall),
    }

    emit("Error rate by evidence type (multi-label from SOLUTION; buckets overlap):")
    emit(f"  {'type':<11} {'wrong':>9}  {'% wrong':>8}  {'score':>6}")
    for label in required_flags + ["ALL"]:
        if label not in per_type:
            continue
        d = per_type[label]
        emit(f"  {label:<11} {d['wrong']:>4}/{d['n']:<4} {d['pct_wrong']:>7.1f}%  {d['score']:>6.3f}")
    emit()

    # ─── Minimal diagnostics: print a few representative incorrect rows ───
    try:
        show = 0 if (IS_KAGGLE or not verbose) else 5  # how many rows to show locally
        # Treat "ref" < 1.0 as a mismatch; others must be strictly True
        mismatch_mask = (~bits["val"]) | (~bits["unit"]) | (bits["ref"] < 1.0) | (~bits["na"])
        bad_ids = bits.index[mismatch_mask].tolist()[:show]

        if bad_ids:
            emit("Examples of mismatches (expected vs. got):")
            for rid in bad_ids:
                s = sol.loc[rid]
                u = sub.loc[rid]

                reasons = []
                # value reason (mirror logic, but summarized)
                if not is_blank(s["answer_value"]):
                    s_list = parse_listish(s["answer_value"])
                    u_list = parse_listish(u["answer_value"])
                    if s_list is not None:
                        if u_list is not None:
                            if not compare_lists(s_list, u_list, tol=1e-3):
                                reasons.append(f"value list mismatch: expected {s_list}, got {u_list}")
                        else:
                            if len(s_list) == 2 and all(isinstance(v, (int, float)) for v in s_list):
                                lo, hi = sorted(s_list)
                                try:
                                    x = float(u["answer_value"])
                                    if not ((lo - 1e-3) <= x <= (hi + 1e-3)):
                                        reasons.append(f"value out of range: expected [{lo}, {hi}], got {x}")
                                except ValueError:
                                    reasons.append(f"value type mismatch: expected numeric in range, got '{u['answer_value']}'")
                            else:
                                reasons.append("value type mismatch: expected list, got scalar/string")
                else:
                    if not is_blank(u["answer_value"]):
                        reasons.append(f"NA agreement failed: expected blank/NA, got '{u['answer_value']}'")

                # unit reason (still shown even though weight=0)
                if canon_unit(s["answer_unit"]) != canon_unit(u["answer_unit"]):
                    reasons.append(f"unit mismatch: expected '{s['answer_unit']}', got '{u['answer_unit']}'")

                # ref reason (now fractional)
                expected_refs = canon_refs(s["ref_id"])
                got_refs = canon_refs(u["ref_id"])
                if ref_overlap_score(s["ref_id"], u["ref_id"]) < 1.0:
                    reasons.append(f"ref partial/none: expected {expected_refs}, got {got_refs}")

                emit(f"- id={rid}")
                emit(f"  expected: value={s['answer_value']} | unit={s['answer_unit']} | ref={expected_refs}")
                emit(f"  got     : value={u['answer_value']} | unit={u['answer_unit']} | ref={got_refs}")
                for why in reasons:
                    emit(f"  why     : {why}")
    except Exception as _e:
        emit(f"[warn] diagnostics printing failed: {_e}")

    # Kaggle requires a single, finite, non-null float. NaN would surface as an opaque
    # submission failure, so fail loudly here instead.
    final = float(overall)
    if not math.isfinite(final):
        raise ParticipantVisibleError(
            "Score could not be computed as a finite number; check that every row has a "
            "parseable answer_value.")
    return final

# ────────────────────────────────────────────────────── CLI hook ──────
if __name__ == "__main__":
    import sys, os
    from pathlib import Path
    import pandas as pd  # already imported above, but safe

    # Kaggle kernels/validate generally set /kaggle paths or env vars
    IS_KAGGLE = os.path.exists("/kaggle") or "KAGGLE_URL_BASE" in os.environ

    # Strip notebook noise like "-f"
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if len(args) == 2:
        # Explicit local run with two files
        sol_path, sub_path = Path(args[0]), Path(args[1])
        print(score(pd.read_csv(sol_path), pd.read_csv(sub_path), "id"))
    elif not IS_KAGGLE:
        sys.stderr.write("Usage: python Score.py solution.csv submission.csv\n")
    else:
        # In Kaggle Save & Validate: NO-OP.
        # Kaggle imports this module and calls score(solution, submission, id_col).
        pass
