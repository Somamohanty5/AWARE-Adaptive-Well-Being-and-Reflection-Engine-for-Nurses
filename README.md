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
