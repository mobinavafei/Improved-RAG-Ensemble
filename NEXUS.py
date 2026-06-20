"""
Setup:
    pip install pandas ddgs groq scikit-learn

Run:
    python run.py               <- all 100 questions
    python run.py --start 48    <- resume from question 48
    python run.py --limit 5     <- test on first 5 only
"""

import argparse
import getpass
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

from groq import Groq


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

BASE_DIR               = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_FILE = BASE_DIR / "questions_100.csv"
DEFAULT_OUTPUT_FILE    = "NEXUS_submission.csv"
DETAILED_OUTPUT_FILE   = "NEXUS_submission_detailed.csv"

# Ensemble models
MODEL_PRIMARY   = "qwen/qwen3-32b"        # 32B — strong, reliable
MODEL_SECONDARY = "llama-3.1-8b-instant"  # 8B  — fast, 14400 req/day

VOTES_PRIMARY   = 3   # votes from primary   → total = 6
VOTES_SECONDARY = 3   # votes from secondary

# Search
ENABLE_WEB_SEARCH     = True
MAX_RESULTS_PER_QUERY = 8
SEARCH_SLEEP_SECONDS  = 2.0

# Ranking
TOP_K_EVIDENCE  = 7
MAX_SNIPPET_LEN = 300

SYSTEM_PROMPT = (
    "You are an expert at answering multiple-choice questions. "
    "Read the evidence carefully and pick the best answer. "
    "Output ONLY a single capital letter: A, B, C, D, or E. "
    "No explanation. No punctuation. Just the letter. /no_think"
)


# ─────────────────────────────────────────────────────────────
# Groq client setup
# ─────────────────────────────────────────────────────────────

def setup_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("No GROQ_API_KEY environment variable found.")
        api_key = getpass.getpass("Paste your Groq API key (gsk_...): ").strip()
    return Groq(api_key=api_key)

client = setup_groq_client()


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_questions(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Place questions_100.csv in the same folder as this script."
        )
    questions = pd.read_csv(file_path)
    required  = {"question_no", "question"}
    missing   = required - set(questions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return questions


# ─────────────────────────────────────────────────────────────
# Query construction  (IMPROVED)
# ─────────────────────────────────────────────────────────────

def build_search_query(row: pd.Series) -> str:
    """Single query — kept for compatibility with starter structure."""
    question = str(row.get("question", "")).strip()
    return re.sub(r'\?$', '', question).strip()[:80]


def build_search_queries(row: pd.Series) -> List[str]:
    """4 targeted queries per question for maximum evidence coverage."""
    question = str(row.get("question", "")).strip()
    subject  = re.sub(r'\?$', '', question).strip()[:80]
    keywords = " ".join(question.split()[:6])
    return [
        subject,
        f"{subject} Wikipedia",
        f"{subject} facts answer",
        f'"{keywords}" according to Wikipedia',
    ]


# ─────────────────────────────────────────────────────────────
# Evidence retrieval  (IMPROVED)
# ─────────────────────────────────────────────────────────────

def search_duckduckgo(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> List[Dict]:
    """Retrieve evidence snippets from DuckDuckGo."""
    if not DDGS_AVAILABLE or not ENABLE_WEB_SEARCH:
        return []
    evidence = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            for item in results:
                evidence.append({
                    "title"  : item.get("title", ""),
                    "snippet": item.get("body", item.get("snippet", "")),
                    "url"    : item.get("href", item.get("url", "")),
                })
            if not results:
                print(f"  [Search] No results for: {query[:50]}")
    except Exception as error:
        print(f"  [Search warning] '{query[:40]}': {error}")
    time.sleep(SEARCH_SLEEP_SECONDS)
    return evidence


def fetch_all_evidence(row: pd.Series) -> List[Dict]:
    """Run all 4 queries and return deduplicated evidence."""
    queries      = build_search_queries(row)
    all_evidence = []
    seen         = set()
    for query in queries:
        results = search_duckduckgo(query, max_results=MAX_RESULTS_PER_QUERY)
        for item in results:
            snippet = item.get("snippet", "").strip()
            key     = snippet[:80]
            if key and key not in seen:
                seen.add(key)
                item["snippet"] = snippet[:MAX_SNIPPET_LEN]
                all_evidence.append(item)
    return all_evidence


# ─────────────────────────────────────────────────────────────
# Evidence ranking  (IMPROVED)
# ─────────────────────────────────────────────────────────────

def rank_evidence(row: pd.Series, evidence: List[Dict], top_k: int = TOP_K_EVIDENCE) -> List[Dict]:
    """Rank evidence by TF-IDF cosine similarity to question + options."""
    if not evidence:
        return []
    question   = str(row.get("question", ""))
    options    = " ".join(
        str(row.get(l, "")) for l in ["A", "B", "C", "D", "E"]
        if pd.notna(row.get(l, "")) and str(row.get(l, "")).strip()
    )
    full_query = f"{question} {options}"
    snippets   = [
        f"{item.get('title', '')}. {item.get('snippet', '')}"
        for item in evidence
    ]
    corpus = snippets + [full_query]
    try:
        vec   = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform(corpus)
        sims  = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        idx   = sims.argsort()[::-1][:top_k]
        return [evidence[i] for i in idx]
    except Exception:
        return evidence[:top_k]


# ─────────────────────────────────────────────────────────────
# Prompt building  (IMPROVED)
# ─────────────────────────────────────────────────────────────

def build_prompt(row: pd.Series, ranked_evidence: List[Dict]) -> str:
    """Build a RAG-style prompt with question, options, and ranked evidence."""
    question = str(row.get("question", ""))
    options  = []
    for option in ["A", "B", "C", "D", "E"]:
        value = row.get(option, "")
        if pd.notna(value) and str(value).strip():
            options.append(f"  {option}. {value}")

    if ranked_evidence:
        evidence_lines = [
            f"[{i+1}] {item.get('title', '')}. {item.get('snippet', '')}"
            for i, item in enumerate(ranked_evidence)
        ]
        evidence_text = "\n".join(evidence_lines)
    else:
        evidence_text = "(No evidence retrieved - use your knowledge)"

    return (
        f"QUESTION:\n{question}\n\n"
        f"OPTIONS:\n" + "\n".join(options) + "\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        f"OUTPUT (one letter only):"
    )


# ─────────────────────────────────────────────────────────────
# Answer generation  (IMPROVED - ENSEMBLE)
# ─────────────────────────────────────────────────────────────

def ask_model(model: str, prompt: str, num_votes: int) -> List[str]:
    """Ask one model num_votes times and return list of answer letters."""
    votes = []
    short = model.split("/")[-1]

    for i in range(num_votes):
        answer = "Unknown"
        # Try with system role first, then merge into user message as fallback
        attempts = [
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": prompt}],
            [{"role": "user",   "content": SYSTEM_PROMPT + "\n\n" + prompt}],
        ]
        for messages in attempts:
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=20,
                    temperature=0,
                    messages=messages,
                )
                raw = response.choices[0].message.content.strip()

                # Strip <THINK>...</THINK> blocks (qwen3 thinking mode)
                raw = re.sub(r'<THINK>.*?</THINK>', '', raw,
                             flags=re.DOTALL | re.IGNORECASE).strip().upper()

                # Print raw on first vote only
                if i == 0:
                    print(f"    [{short} raw]: {repr(raw[:80])}")

                # Find valid letter: word boundary first, then any char
                match = re.search(r'\b([A-E])\b', raw)
                if match:
                    answer = match.group(1)
                else:
                    for ch in raw:
                        if ch in {"A", "B", "C", "D", "E"}:
                            answer = ch
                            break
                break  # success — skip fallback attempt

            except Exception as e:
                if messages is attempts[0]:
                    continue  # try without system role
                print(f"    [{short} error]: {e}")

        votes.append(answer)
    return votes


def generate_answer(prompt: str) -> str:
    """
    Ensemble: qwen3-32b (3 votes) + llama-3.1-8b (3 votes) = 6 total votes.
    Majority wins. Confidence = fraction of votes agreeing on best answer.
    """
    votes_primary   = ask_model(MODEL_PRIMARY,   prompt, VOTES_PRIMARY)
    votes_secondary = ask_model(MODEL_SECONDARY, prompt, VOTES_SECONDARY)
    all_votes       = votes_primary + votes_secondary

    print(f"    qwen3-32b       : {votes_primary}")
    print(f"    llama-3.1-8b    : {votes_secondary}")

    valid = [v for v in all_votes if v != "Unknown"]
    if not valid:
        generate_answer.last_votes      = all_votes
        generate_answer.last_confidence = 0.0
        return "Unknown"

    counts     = Counter(valid)
    best, cnt  = counts.most_common(1)[0]
    confidence = round(cnt / len(all_votes), 2)

    generate_answer.last_votes      = all_votes
    generate_answer.last_confidence = confidence
    return best

generate_answer.last_votes      = []
generate_answer.last_confidence = 0.0


def clean_answer(answer: str) -> str:
    """Validate and normalise the answer."""
    answer = str(answer).strip()
    if answer.lower() == "unknown":
        return "Unknown"
    answer = answer.upper()[:1]
    if answer in {"A", "B", "C", "D", "E"}:
        return answer
    return "Unknown"


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def answer_question(row: pd.Series) -> Dict:
    """Run the full improved pipeline for one question."""
    evidence        = fetch_all_evidence(row)
    ranked_evidence = rank_evidence(row, evidence, top_k=TOP_K_EVIDENCE)
    prompt          = build_prompt(row, ranked_evidence)
    raw_answer      = generate_answer(prompt)
    answer          = clean_answer(raw_answer)
    votes           = generate_answer.last_votes
    confidence      = generate_answer.last_confidence
    return {
        "question_no" : row.get("question_no"),
        "answer"      : answer,
        "confidence"  : confidence,
        "votes"       : ",".join(votes),
        "snippets"    : len(ranked_evidence),
    }


def run_pipeline(questions_file: str, output_file: str,
                 limit: int = None, start: int = 1) -> pd.DataFrame:

    questions = load_questions(questions_file)

    # Resume from start question
    if start > 1:
        questions = questions[questions["question_no"] >= start].reset_index(drop=True)
        print(f" Resuming from question {start}")

    if limit is not None:
        questions = questions.head(limit)

    total_votes = VOTES_PRIMARY + VOTES_SECONDARY
    print("=" * 65)
    print(f" BCU AI HACKATHON 2026  |  ENSEMBLE MODE")
    print(f" Primary  : {MODEL_PRIMARY}  x{VOTES_PRIMARY} votes")
    print(f" Secondary: {MODEL_SECONDARY}  x{VOTES_SECONDARY} votes")
    print(f" Total    : {total_votes} votes per question")
    print(f" Questions: {len(questions)}")
    print("=" * 65)

    predictions = []
    for index, row in questions.iterrows():
        question_no = row.get("question_no", index + 1)
        print(f"\n[Q{int(question_no):03d}] {str(row.get('question', ''))[:70]}...")

        result = answer_question(row)
        predictions.append(result)

        conf_bar = "★" * round(result["confidence"] * 5)
        print(f"  -> Answer: [{result['answer']}]  "
              f"|  Conf: {result['confidence']:.0%} {conf_bar}  "
              f"|  All votes: {result['votes']}  "
              f"|  Evidence: {result['snippets']} snippets")

        time.sleep(0.3)

    # Build result DataFrame
    new_df = pd.DataFrame(predictions)

    # Merge with existing results if resuming
    if start > 1 and Path(DETAILED_OUTPUT_FILE).exists():
        existing = pd.read_csv(DETAILED_OUTPUT_FILE)
        existing = existing[~existing["question_no"].isin(new_df["question_no"])]
        full_df  = pd.concat([existing, new_df], ignore_index=True)
        full_df  = full_df.sort_values("question_no").reset_index(drop=True)
        print(f"\n Merged with existing — total: {len(full_df)} questions saved")
    else:
        full_df = new_df

    # Save official submission (question_no + answer only)
    submission = full_df[["question_no", "answer"]].copy()
    submission.to_csv(output_file, index=False)

    # Save detailed version for manual review
    full_df.to_csv(DETAILED_OUTPUT_FILE, index=False)

    # Summary
    answered = full_df[full_df["answer"] != "Unknown"]
    unknowns = full_df[full_df["answer"] == "Unknown"]
    hi_conf  = full_df[full_df["confidence"] >= 1.0]
    low_conf = full_df[full_df["confidence"] <  0.67]

    print("\n" + "=" * 65)
    print(" RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Total questions  : {len(full_df)}")
    print(f"  Answered         : {len(answered)}")
    print(f"  Unknown          : {len(unknowns)}")
    print(f"  High conf (100%) : {len(hi_conf)}")
    print(f"  Low conf (<67%)  : {len(low_conf)}  <- review these manually")
    print(f"  Avg confidence   : {full_df['confidence'].mean():.1%}")
    print(f"\n  Submission  -> {output_file}")
    print(f"  Detailed    -> {DETAILED_OUTPUT_FILE}")
    print("=" * 65)
    print("\n  NEXT STEP: open NEXUS_submission_detailed.csv")
    print("  Filter confidence < 1.0 and manually verify those answers.")
    print("=" * 65)

    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCU AI Hackathon 2026 - Ensemble Pipeline")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_FILE),
                        help="Path to questions_100.csv")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT_FILE,
                        help="Output CSV filename")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Only answer first N questions")
    parser.add_argument("--start",     type=int, default=1,
                        help="Resume from this question number (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.questions, args.output, args.limit, args.start)
