# SecondOpinion

**Can you trust your AI's explanation of code?**

---

## The Problem

AI tools explain code helpfully — but different AI perspectives can silently contradict each other. A security-focused model might say a password is hashed; a general-purpose model might say it's stored in plain text. Without cross-checking, you'd never know which answer to trust.

SecondOpinion surfaces those contradictions automatically.

---

## How It Works

1. **Three parallel explainer agents** analyse the same code snippet simultaneously from three angles:
   - **Functionality** — what the code actually does (senior engineer perspective)
   - **Security** — input validation, credential handling, injection risks (security engineer perspective)
   - **Beginner-friendly** — plain-language explanation with a real-life analogy

2. **Judge agent** receives all three explanations and cross-checks them for factual contradictions, identifying both major and minor disagreements.

3. **Stability Score (0–100)** — the judge computes a score and a verdict:
   - **Trustworthy** (≥ 80): perspectives are broadly consistent
   - **Use with Caution** (50–79): some inconsistencies detected
   - **Unreliable** (< 50): significant disagreements — treat with care

---

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Demo Mode

**No API key required.** If `OPENAI_API_KEY` is not set in your environment, the app runs in demo mode and returns a pre-built sample analysis so you can explore the full UI immediately.

To use a real OpenAI key:

```bash
export OPENAI_API_KEY=sk-...
python app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| LLM | OpenAI API (`gpt-4o-mini` by default) |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` (3 parallel agents) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |

---

## Project Structure

```
second-opinion/
├── app.py               # Flask app — agent pipeline + API route
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Single-page frontend
└── bob-sessions/        # Bob task session screenshots
```

---

## Built with IBM Bob 2.0

This project was built entirely inside **IBM Bob**, using Agent mode across four focused tasks:

| Task | What was built |
|---|---|
| **1 — Scaffold** | Flask project structure, `app.py` skeleton, `index.html` layout, `requirements.txt` |
| **2 — Agent pipeline** | Three parallel specialist agents, judge agent, Stability Score logic, `/api/check` route |
| **3 — Demo mode** | Full demo response (no API key needed), demo badge UI, error states, loading spinner |
| **4 — Polish** | `README.md`, `LICENSE`, `.gitignore`, `bob-sessions/` folder, final bug-fix review |

Bob's Agent mode wrote, wired, and validated every file — no manual coding required.

---

## License

MIT — see [LICENSE](LICENSE).
