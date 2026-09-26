import os
import json
import concurrent.futures
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _chat(client: OpenAI, system: str, user: str) -> str:
    """Single blocking chat completion. Returns the assistant text."""
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# The three specialist agents
# ---------------------------------------------------------------------------

_FUNCTIONALITY_SYSTEM = (
    "You are a senior software engineer. "
    "When given a code snippet, explain exactly what it does: its inputs, "
    "outputs, and main logic steps. Be factual and precise. "
    "Write 4–6 clear sentences. Do NOT use bullet points or headers."
)

_SECURITY_SYSTEM = (
    "You are a security engineer. "
    "When given a code snippet, identify all security-relevant behaviors: "
    "input validation (or lack of it), how data is stored or transmitted, "
    "whether secrets or credentials appear, authentication/authorisation "
    "patterns, and injection risks. "
    "Write 4–6 clear sentences. Do NOT use bullet points or headers."
)

_BEGINNER_SYSTEM = (
    "You are a patient teacher who explains code to complete beginners. "
    "Use simple everyday language and include one real-life analogy. "
    "Avoid jargon. "
    "Write 4–6 clear sentences. Do NOT use bullet points or headers."
)


def _agent_functionality(client: OpenAI, code: str, language: str) -> str:
    return _chat(client, _FUNCTIONALITY_SYSTEM,
                 f"Language: {language}\n\n```\n{code}\n```")


def _agent_security(client: OpenAI, code: str, language: str) -> str:
    return _chat(client, _SECURITY_SYSTEM,
                 f"Language: {language}\n\n```\n{code}\n```")


def _agent_beginner(client: OpenAI, code: str, language: str) -> str:
    return _chat(client, _BEGINNER_SYSTEM,
                 f"Language: {language}\n\n```\n{code}\n```")


# ---------------------------------------------------------------------------
# Judge agent
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a rigorous fact-checker. "
    "You receive three explanations of the same code snippet, labelled "
    "Functionality, Security, and Beginner. "
    "Your job:\n"
    "1. Extract the key factual claims from each explanation.\n"
    "2. Cross-check those claims against each other and identify "
    "CONTRADICTIONS — places where two explanations make claims that "
    "directly disagree.\n"
    "3. Compute a Stability Score (integer 0–100): start at 100, subtract "
    "25 for each MAJOR contradiction (a direct factual disagreement), "
    "subtract 10 for each MINOR contradiction (vague or hedged disagreement), "
    "floor at 0.\n"
    "4. Set the verdict: 'Trustworthy' if score >= 80, "
    "'Use with Caution' if 50–79, 'Unreliable' if below 50.\n"
    "5. Write each contradiction in plain, non-technical English, e.g. "
    "'Explanation 1 says the password is stored safely, but Explanation 2 "
    "says it is stored as plain text.'\n\n"
    "Respond with ONLY valid JSON in this exact shape:\n"
    "{\n"
    '  "score": <integer 0-100>,\n'
    '  "verdict": "<Trustworthy|Use with Caution|Unreliable>",\n'
    '  "contradictions": ["<plain-English description>", ...]\n'
    "}\n"
    "If there are no contradictions, return an empty array for contradictions."
)


def _agent_judge(
    client: OpenAI,
    functionality: str,
    security: str,
    beginner: str,
) -> dict:
    user_msg = (
        f"Functionality explanation:\n{functionality}\n\n"
        f"Security explanation:\n{security}\n\n"
        f"Beginner explanation:\n{beginner}"
    )
    raw = _chat(client, _JUDGE_SYSTEM, user_msg)

    # Strip markdown code fences if the model wraps its JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Demo-mode sample response (used when OPENAI_API_KEY is not set)
# ---------------------------------------------------------------------------

_DEMO_RESPONSE = {
    "demo": True,
    "score": 62,
    "verdict": "Use with Caution",
    "contradictions": [
        (
            "The functionality explanation says user passwords are hashed with bcrypt "
            "before being stored, but the security explanation says passwords are saved "
            "as plain text directly in the database."
        ),
        (
            "The functionality explanation says the function returns True on a successful "
            "login, but the beginner explanation says it returns a session token string."
        ),
    ],
    "explanations": {
        "functionality": (
            "This Python function, login(username, password), accepts a username and a "
            "plain-text password as arguments and attempts to authenticate a user against "
            "a database. It first queries the users table by username, then uses bcrypt to "
            "compare the supplied password against the stored hash. If the hashes match it "
            "creates a session entry and returns True; otherwise it returns False. The "
            "function raises a DatabaseError if the connection to the database cannot be "
            "established."
        ),
        "security": (
            "The function performs no input sanitisation on the username field, leaving it "
            "open to SQL injection if a raw string query is used elsewhere in the codebase. "
            "Critically, passwords are stored as plain text in the database column — there "
            "is no evidence of hashing or salting, which means a data breach would expose "
            "every user's password immediately. The function does not implement any "
            "rate-limiting or account lock-out mechanism, making it trivially vulnerable to "
            "brute-force attacks. Returning a generic boolean also leaks information about "
            "whether the account exists."
        ),
        "beginner": (
            "Think of this function as a bouncer at a nightclub who checks your name against "
            "a guest list. You hand the bouncer your name (username) and a secret code word "
            "(password). The bouncer looks you up in the list and checks whether your code "
            "word matches what's written next to your name. If everything matches, the "
            "bouncer hands you a wristband (a session token) so you can get in; if not, you "
            "are turned away. The important thing to know is that your code word should "
            "never be written in plain sight on the list — it should be scrambled so only "
            "you know the original."
        ),
    },
}


@app.route("/api/check", methods=["POST"])
def check():
    data = request.get_json(force=True)
    code = (data.get("code") or "").strip()
    language = (data.get("language") or "unknown").strip()

    if not code:
        return jsonify({"error": "No code provided"}), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(_DEMO_RESPONSE)

    client = OpenAI(api_key=api_key)

    # ── Step 1: run the three specialist agents IN PARALLEL ──────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        fut_func = pool.submit(_agent_functionality, client, code, language)
        fut_sec  = pool.submit(_agent_security,      client, code, language)
        fut_beg  = pool.submit(_agent_beginner,      client, code, language)

        functionality = fut_func.result()
        security      = fut_sec.result()
        beginner      = fut_beg.result()

    # ── Step 2: judge agent cross-checks all three ───────────────────────────
    judgment = _agent_judge(client, functionality, security, beginner)

    # ── Step 3: return the combined result ───────────────────────────────────
    return jsonify({
        "score":         judgment["score"],
        "verdict":       judgment["verdict"],
        "contradictions": judgment.get("contradictions", []),
        "explanations": {
            "functionality": functionality,
            "security":      security,
            "beginner":      beginner,
        },
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
