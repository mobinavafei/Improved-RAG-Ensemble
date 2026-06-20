# Improved RAG Ensemble Pipeline

This project implements an improved multiple-choice question answering pipeline for the AI Hackathon 2026. It extends the official starter-code style with web evidence retrieval, TF-IDF evidence ranking, RAG-style prompting, and ensemble voting across two Groq-hosted language models.

The pipeline reads questions from `questions_100.csv`, searches the web for evidence, ranks the most relevant snippets, asks two LLMs to answer each question multiple times, applies majority voting, calculates a confidence score, and exports both a submission file and a detailed review file.

---

## Key Features

- **Multi-query search**: Generates 4 targeted search queries per question to improve evidence coverage.
- **DuckDuckGo evidence retrieval**: Uses `ddgs` or `duckduckgo_search` to collect web snippets.
- **Evidence deduplication**: Removes repeated snippets returned by overlapping search queries.
- **TF-IDF evidence ranking**: Ranks snippets by cosine similarity against the question and answer options.
- **RAG-style prompt construction**: Sends the question, options, and ranked evidence to the models.
- **Two-model ensemble**: Uses `qwen/qwen3-32b` and `llama-3.1-8b-instant`.
- **Majority voting**: Combines 6 total votes per question to select the final answer.
- **Confidence scoring**: Calculates confidence based on agreement among model votes.
- **Resume support**: Allows restarting from any question number using `--start`.
- **Detailed output**: Saves votes, confidence, and evidence count for manual review.

---

## Pipeline Overview

```text
questions_100.csv
       ↓
4 search queries per question
       ↓
DuckDuckGo evidence retrieval
       ↓
Snippet deduplication
       ↓
TF-IDF ranking → top 7 evidence snippets
       ↓
RAG prompt: question + options + ranked evidence
       ↓
qwen/qwen3-32b        × 3 votes
llama-3.1-8b-instant × 3 votes
       ↓
6 total votes → majority answer
       ↓
confidence score
       ↓
NEXUS_submission.csv
NEXUS_submission_detailed.csv
```

---

## Requirements

Install the required Python packages:

```bash
pip install pandas ddgs groq scikit-learn
```

If `ddgs` is unavailable, the code attempts to fall back to:

```bash
pip install duckduckgo-search
```

---

## API Key Setup

The pipeline uses the Groq API. Set your API key as an environment variable:

### Linux / macOS

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

If the environment variable is not found, the script will ask you to paste the API key securely in the terminal.

---

## Input File Format

The default input file is:

```text
questions_100.csv
```

It must be placed in the same folder as the Python script unless another path is provided.

Required columns:

```text
question_no, question
```

Expected option columns:

```text
A, B, C, D, E
```

Example:

```csv
question_no,question,A,B,C,D,E
1,"What is the capital of France?","Paris","London","Berlin","Madrid","Rome"
```

---

## How to Run

Run all questions:

```bash
python run.py
```

Run only the first 5 questions for testing:

```bash
python run.py --limit 5
```

Use a custom input file:

```bash
python run.py --questions path/to/questions_100.csv
```

Use a custom output file:

```bash
python run.py --output my_submission.csv
```

---

## Output Files

### 1. Official Submission File

```text
NEXUS_submission.csv
```

Contains only:

```text
question_no, answer
```

This is the file intended for final submission.

### 2. Detailed Review File

```text
NEXUS_submission_detailed.csv
```

Contains:

```text
question_no, answer, confidence, votes, snippets
```

Use this file to manually review uncertain answers.

---

## Methodology

### 1. Multi-query Search

For each question, the script builds four queries:

1. The main question text.
2. The question with `Wikipedia` added.
3. The question with `facts answer` added.
4. A keyword-based query using the first few words of the question.

This improves recall because one search query may miss relevant evidence.

### 2. Evidence Retrieval

The script retrieves search results from DuckDuckGo and stores each result as:

```python
{
    "title": "...",
    "snippet": "...",
    "url": "..."
}
```

Snippets are truncated to a maximum length of 300 characters to keep the prompt compact.

### 3. Deduplication

Repeated snippets are removed using the first 80 characters as a deduplication key. This prevents duplicated evidence from dominating the prompt.

### 4. TF-IDF Ranking

The script combines the question and answer options into one query text, then compares it against all retrieved snippets using TF-IDF cosine similarity.

Only the top 7 snippets are selected for the final prompt.

### 5. RAG Prompting

The final prompt contains:

- the question,
- the answer options,
- the ranked evidence snippets,
- an instruction to output only one letter.

This makes the model answer using retrieved evidence instead of relying only on internal knowledge.

### 6. Ensemble Voting

The system asks two models:

```text
qwen/qwen3-32b        → 3 votes
llama-3.1-8b-instant → 3 votes
```

This gives 6 votes per question. The most common valid answer is selected as the final answer.

### 7. Confidence Score

Confidence is calculated as:

```text
number of votes for the winning answer / total votes
```

Examples:

```text
6/6 votes agree → confidence = 1.00
4/6 votes agree → confidence = 0.67
3/6 votes agree → confidence = 0.50
```

Low-confidence answers should be checked manually.

---

## Important Configuration Values

```python
MODEL_PRIMARY = "qwen/qwen3-32b"
MODEL_SECONDARY = "llama-3.1-8b-instant"

VOTES_PRIMARY = 3
VOTES_SECONDARY = 3

MAX_RESULTS_PER_QUERY = 8
TOP_K_EVIDENCE = 7
MAX_SNIPPET_LEN = 300
SEARCH_SLEEP_SECONDS = 2.0
```

---

## Manual Review Recommendation

After running the full pipeline, open:

```text
NEXUS_submission_detailed.csv
```

Review rows where:

```text
confidence < 0.67
```

These are cases where the ensemble did not strongly agree.

It is also useful to manually review answers with confidence below 1.00 if maximum accuracy is required.

---

## Strengths

- More reliable than asking one model directly.
- Uses external evidence, reducing hallucination risk.
- Ranks evidence automatically before prompting.
- Provides confidence values for review.
- Can resume from a specific question if interrupted.
- Produces both official and detailed output files.

---

## Limitations

- DuckDuckGo snippets may be incomplete or noisy.
- Search results depend on internet availability and rate limits.
- TF-IDF ranking is lexical, so it may miss semantically relevant evidence with different wording.
- With `temperature=0`, repeated votes from the same model may not be very diverse.
- The system does not open full webpages; it only uses search result snippets.
- Majority voting improves robustness but does not guarantee correctness.

---

## Possible Improvements

- Use full-page retrieval instead of only snippets.
- Add source filtering for trusted domains such as Wikipedia, Britannica, or official sources.
- Use semantic embedding ranking instead of only TF-IDF.
- Add different prompt variants for more diverse voting.
- Use a tie-breaking strategy when votes are evenly split.
- Store retrieved evidence and URLs in the detailed output file for easier manual verification.
- Add automatic answer validation against source text.

---

## Project Files

```text
run.py                         Main pipeline script
questions_100.csv              Input questions file
NEXUS_submission.csv           Final submission output
NEXUS_submission_detailed.csv  Detailed output for review
README.md                      Project documentation
```

---

## Example Command

```bash
python run.py --questions questions_100.csv --output NEXUS_submission.csv
```

---

## Summary

This project is an evidence-assisted multiple-choice answering system. It combines web search, TF-IDF ranking, RAG prompting, two-model ensemble voting, and confidence scoring to produce more reliable answers than a single direct LLM call.
