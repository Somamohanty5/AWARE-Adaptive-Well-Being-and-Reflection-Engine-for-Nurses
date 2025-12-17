# AWARE: Adaptive Well-Being Reflection App for Healthcare Workers

AWARE is a small web app that helps healthcare workers pause for a moment, check in with their stress level, and get a short, adaptive reflection in return.

It is **not** a therapy tool and **does not** give diagnoses or crisis advice. Instead, it acts like a calm, lightweight reflection partner that adjusts tone and depth based on what the user writes.

> One sentence summary:  
> AWARE is an adaptive, LLM-powered check-in tool that supports nurses’ well-being with brief, context-aware reflections.

---

## Why AWARE?

Nurses and other healthcare workers face high rates of burnout and emotional exhaustion. Many wellness tools are:

- generic and one-size-fits-all,
- not sensitive to *how* people are coping, and  
- hard to fit into an already overloaded workflow.

AWARE explores a different design:

- **Short, realistic check-ins** (1–2 minutes)
- **Adaptive responses**, based on:
  - self-reported stress (1–5),
  - coping style,
  - user intent (e.g., asking for suggestions vs. just venting),
  - recent history
- **Clear, predictable modes**: Quick, Normal, Deep

This project was built for a course on Intelligent User Interfaces and evaluated in a small within-subject user study.

---

## What AWARE Does (Core Features)

From the user’s point of view:

- **Login & landing page**
  - Simple nurse login (demo credentials via environment variables)
  - Landing page explaining purpose and safety constraints

- **Check-in form**
  - Participant ID  
  - Stress rating (1–5)
  - Mode: **Quick**, **Normal**, or **Deep**
  - Free-text reflection

- **Adaptive reflection**
  - Uses Google’s Gemini LLM (`gemini-flash-latest`) to generate:
    - empathetic but **non-clinical** text,
    - different length and “depth” depending on mode,
    - tone sensitive to stress level and message content,
    - 1–2 small, non-clinical suggestions if the user clearly asks for help (e.g., “Any advice?”).

- **Coping-style tagging (rule-based)**
  - Classifies each message as:
    - *problem-focused*,  
    - *emotion-focused*,  
    - *avoidant*, or  
    - *mixed/unclear*  
  - This tag is stored for analysis and lightly informs the LLM prompt (not shown directly to the user).

- **Relational memory**
  - AWARE can notice differences in stress since the last check-in (higher / lower / similar).
  - The LLM prompt includes a short, textual summary of recent entries so responses can feel more continuous over time.

- **Recent check-ins panel (UI)**
  - On the main app page, users can see a compact list of their latest check-ins and reflections without leaving the screen.

- **Pattern summary**
  - A separate endpoint generates a short, human-readable summary of recent patterns (e.g., average stress, coping style trends).
  - If Gemini is unavailable, a simple rule-based summary is used instead.

- **JSON logging for research**
  - Every check-in is logged as a JSON entry per participant:
    - timestamp  
    - user_id (Participant ID)  
    - stress, mode, coping_mode  
    - text, reply  
    - whether relational memory was available  
    - reply length (for verifying Quick vs Normal vs Deep)

---

## Architecture (High-Level)

- **Backend:** FastAPI (Python)
- **Frontend:** Jinja2 templates + vanilla HTML/CSS/JS (served via FastAPI)
- **LLM:** Google Gemini (`gemini-flash-latest`) with a carefully constrained prompt
- **Storage:**
  - Per-participant JSON log at `data/<participant_id>/checkins.json`
  - In-memory history (per run) for fast relational memory
- **Authentication:** very simple demo login using cookies (`user_id`, `session_id`)

Directory sketch:

```text
aware_app/
  app.py              # FastAPI app and core logic
  templates/
    landing.html
    login.html
    index.html        # main AWARE UI
    about.html
    workflow.html
  static/
    css/, js/, images/
  data/
    <participant_id>/
      checkins.json
  requirements.txt
  README.md
```
# Getting Started
1. Prerequisites

Python 3.9+

A Google Gemini API key (for gemini-flash-latest)

pip and virtualenv (or conda)

2. Clone and set up environment
git clone <your-repo-url>.git
cd aware_app

(Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

3. Set environment variables

At minimum:

export GEMINI_API_KEY="your_api_key_here"

Optional: customize demo login credentials
export AWARE_DEMO_USER="nurse"
export AWARE_DEMO_PASS="password"


On Windows PowerShell:

$env:GEMINI_API_KEY="your_api_key_here"
$env:AWARE_DEMO_USER="nurse"
$env:AWARE_DEMO_PASS="password"

4. Run the app
uvicorn app:app --reload


Open your browser and go to:

http://127.0.0.1:8000


Log in using the demo credentials you set (default: nurse / password), then go to the main app page and start a check-in.

# Using AWARE (Walkthrough)

Log in as “nurse”
Use the login page (/login) with the configured username/password.

Go to the main app (/app)

Enter a Participant ID (e.g., P001).

Choose a stress rating from 1 (very low) to 5 (very high).

Pick a mode:

Quick: short, 2–3 sentence reflection, no follow-up questions.

Normal: 3–4 sentences and at most one gentle question.

Deep: 4–5 sentences, up to two reflective questions.

Type a short free-text reflection and submit.

Read the generated reflection

AWARE responds with a concise, empathetic message.

If you explicitly ask for help (“Any suggestions?”, “What should I do?”), it adds 1–2 small, non-clinical ideas (e.g., taking a break, reflecting on one thing that went okay).

Review recent check-ins

The side panel shows your latest entries and responses for that Participant ID.

Pattern summary (for study/evaluation)

A separate endpoint (/api/pattern_summary/{user_id}) can be used to pull a brief description of patterns over time (used in the user study).

# Safety & Scope

AWARE is intentionally limited:

Not a diagnostic tool
It does not detect or treat mental health conditions.

No crisis handling
It does not provide crisis instructions or emergency advice.

Reflection only
It is designed as a reflection aid and a research prototype for adaptive IUIs, not a clinical product.

These constraints are explicitly encoded in the LLM prompts.

# How This Was Evaluated (Research Context)

This app was evaluated in a small within-subject study with 13 participants over two days:

Pre-task survey (background, prior chatbot use, well-being proxies)

Interaction with AWARE in Normal / Deep modes

Pattern summary view for their Participant ID

Post-task survey with:

5-point Likert ratings for empathy, clarity, usefulness, personalization, trust, continuity

Open-ended feedback on tone and adaptivity

Log-level metrics:

stress, mode, coping style labels, reply length

used to verify that Quick / Normal / Deep behave differently in practice

The codebase you see here supports that study: all interactions are logged to JSON for later analysis.

# Limitations & Next Steps

Some known limitations:

Short, two-day study with a small sample (N=13)

Participants simulated nurse scenarios; not deployed in real clinical workflow

Tone and coping detection are still imperfect (simple rules + LLM)

Future directions we’d like to explore:

Larger, longitudinal deployments with real nurses

Direct comparisons against a generic LLM baseline

More personalization (tone preferences, coping style over time)

A mobile / in-clinic version with micro check-ins and notifications
