import os
import csv
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field, validator

# ===== Google Gemini LLM client (env-based init) =====
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-flash-latest")
        print("Gemini initialized in AWARE backend (using env GEMINI_API_KEY).")
    except Exception as e:
        print("Error initializing Gemini:", e)
else:
    print("GEMINI_API_KEY not set in environment; Gemini disabled.")



# ================== CONFIG ==================

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
LOG_FILE = BASE_DIR / "aware_logs.csv"

# Mount static folder (for CSS/JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize log file with header if not existing
if not LOG_FILE.exists():
    with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "user_id",
            "stress",
            "mode",
            "coping_mode",
            "text",
            "reply",
            "used_memory",
            "reply_length"
        ])

# ================== IN-MEMORY STATE ==================

# Simple in-memory history per user_id (for relational memory during current run)
user_history: Dict[str, List[dict]] = {}


# ================== Pydantic MODELS ==================

class CheckinRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    stress: int  # 1–5
    mode: str    # "Quick" | "Normal" | "Deep"
    text: str

    @validator("stress")
    def stress_in_range(cls, v):
        if v < 1:
            return 1
        if v > 5:
            return 5
        return v

    @validator("mode")
    def normalize_mode(cls, v):
        allowed = {"Quick", "Normal", "Deep"}
        if v not in allowed:
            return "Normal"
        return v

    @validator("text")
    def non_empty_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty.")
        return v.strip()


class CheckinResponse(BaseModel):
    reply: str
    coping_mode: str
    used_memory: bool


class HistoryItem(BaseModel):
    timestamp: str
    stress: int
    mode: str
    coping_mode: str
    text: str
    reply: str
    used_memory: bool


class PatternSummaryResponse(BaseModel):
    summary: str


# ================== CORE LOGIC (RULE-BASED) ==================

def classify_coping_mode(text: str) -> str:
    """
    Very simple rule-based coping style classifier.
    This is our 'user modeling' for coping style.
    """
    t = text.lower()

    problem_words = ["fix", "plan", "schedule", "manage", "organize",
                     "solve", "solution", "next step", "to-do", "task"]
    emotion_words = ["feel", "sad", "anxious", "anxiety", "overwhelmed",
                     "angry", "upset", "tired", "exhausted", "burned out",
                     "burnt out", "frustrated", "stressed"]
    avoidant_words = ["ignore", "numb", "scroll", "avoid", "escape",
                      "don’t want to think", "dont want to think", "shut down",
                      "zone out", "tune out"]

    score_problem = sum(w in t for w in problem_words)
    score_emotion = sum(w in t for w in emotion_words)
    score_avoid = sum(w in t for w in avoidant_words)

    if score_problem >= max(score_emotion, score_avoid) and score_problem > 0:
        return "problem-focused"
    if score_emotion >= max(score_problem, score_avoid) and score_emotion > 0:
        return "emotion-focused"
    if score_avoid > 0:
        return "avoidant"
    return "mixed/unclear"


def generate_feedback(stress: int,
                      mode: str,
                      coping_mode: str,
                      text: str,
                      last_entry: Optional[dict]):
    """
    Adaptive feedback generator (rule-based):
    - Adapts by stress level (tone)
    - Adapts by coping style (content)
    - Uses relational memory when last_entry is present
    - Adapts depth by mode (Quick / Normal / Deep)
    """

    used_memory = False
    lines = []
        # Special intent: user explicitly asks for encouragement
    lower_text = text.lower()
    if "encourag" in lower_text or "motivat" in lower_text or "cheer me up" in lower_text:
        lines.append(
            "It makes sense to want a bit of encouragement when things feel heavy or draining."
        )
        lines.append(
            "You’ve already done something important just by pausing and noticing how you feel instead of pushing it away."
        )
        lines.append(
            "You don’t have to be perfect or handle everything alone — even small acts of care for yourself count, especially on hard days."
        )
        if mode == "Quick":
            lines.append(
                "Before you move on, can you name one thing you did today that took effort or care, even if nobody else noticed?"
            )
        else:
            lines.append(
                "If you’d like, take a moment to write down one thing you handled today that was hard, and one small kindness you can offer yourself about it."
            )
        reply = " ".join(lines)
        return reply, used_memory
    # Opening: acknowledge stress level (adaptive tone)
    if stress <= 2:
        openings = [
            "Thank you for checking in. It sounds like your stress is on the lighter side today.",
            "I appreciate you pausing for a moment. It sounds like things are relatively manageable right now."
        ]
    elif stress <= 4:
        openings = [
            "Thanks for taking a moment to check in. It sounds like things are a bit heavy right now.",
            "I’m glad you’re making space to reflect. It sounds like your day has been pretty demanding."
        ]
    else:
        openings = [
            "I’m really glad you reached out. It sounds like you’re under a lot of stress at the moment.",
            "It takes courage to pause when things feel this intense. I’m glad you’re giving yourself this moment."
        ]
    lines.append(random.choice(openings))

    # Coping-mode-specific reflection (user modeling → feedback)
    if coping_mode == "problem-focused":
        lines.append(
            "You seem to be thinking in terms of plans and actions, which can be helpful as long as you’re not putting all the pressure on yourself."
        )
        lines.append(
            "Even one small, realistic step can be enough for today—everything doesn’t have to be fixed at once."
        )
    elif coping_mode == "emotion-focused":
        lines.append(
            "You’re spending energy on how you feel, which is completely valid—especially in a demanding role like nursing."
        )
        lines.append(
            "Naming feelings can be an important part of coping and can make it easier to notice when you need support."
        )
    elif coping_mode == "avoidant":
        lines.append(
            "I’m hearing some signals of wanting to shut down or escape, which can happen when things feel like too much."
        )
        lines.append(
            "It might help to gently notice one small thing you can stay present with, without forcing yourself to solve everything right now."
        )
    else:
        lines.append(
            "Your reflection touches on several things at once, which is very normal when your day has many moving parts."
        )
        lines.append(
            "You don’t have to untangle everything perfectly—just noticing what stands out most right now is already useful."
        )

    # Relational memory: reference last check-in if available
    if last_entry is not None:
        used_memory = True
        prev_stress = last_entry.get("stress")
        if prev_stress is not None:
            if stress > prev_stress:
                lines.append("Compared to your last check-in, your stress seems a bit higher today.")
            elif stress < prev_stress:
                lines.append("Compared to your last check-in, your stress looks a little lower today.")
            else:
                lines.append("Your stress level looks similar to your last check-in.")

        lines.append(
            "Thanks for staying consistent with these check-ins—patterns over time can tell you when you might need extra support or rest."
        )

    # Depth based on mode (adaptivity by user preference)
    if mode == "Quick":
        lines.append(
            "Before you head back into your shift, is there one small, realistic thing you can do for yourself today (even a 2-minute pause)?"
        )
    elif mode == "Normal":
        lines.append(
            "If you’d like, notice: what’s the hardest part of your shift right now, and what’s one small support—or boundary—that would ease it even a little?"
        )
    else:  # Deep
        lines.append(
            "If you have a minute, you might reflect on three things:\n"
            "• What’s draining you most right now?\n"
            "• What has helped even a little in the past in similar moments?\n"
            "• Who or what could share a bit of this load with you, even in a small way?"
        )

    reply = " ".join(lines)
    return reply, used_memory


def log_interaction(user_id: str,
                    stress: int,
                    mode: str,
                    coping_mode: str,
                    text: str,
                    reply: str,
                    used_memory: bool):
    """Append a row to the CSV log (for evaluation + analysis)."""
    row = [
        datetime.now().isoformat(),
        user_id.strip(),         
        stress,
        mode,
        coping_mode,
        text,
        reply,
        int(used_memory),
        len(reply),
    ]

    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ================== HISTORY & PATTERN HELPERS ==================

def load_user_rows_from_csv(user_id: str) -> List[dict]:
    """
    Load all rows for a user_id from the CSV log.
    Matching is case-insensitive and ignores extra spaces.
    """
    rows = []
    target = (user_id or "").strip().lower()

    if LOG_FILE.exists():
        with LOG_FILE.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("user_id") or "").strip().lower()
                if rid == target:
                    rows.append(row)

    print(f"Found {len(rows)} rows for user '{target}'")
    return rows


def basic_pattern_summary(rows: List[dict]) -> str:
    """Simple rule-based pattern summary if LLM not available."""
    if not rows:
        return "No check-ins yet for this participant."

    stresses = [int(r["stress"]) for r in rows]
    avg_stress = sum(stresses) / len(stresses)

    # Most common coping mode
    from collections import Counter
    coping_counts = Counter(r["coping_mode"] for r in rows)
    common_coping, _ = coping_counts.most_common(1)[0]

    if avg_stress < 2.5:
        stress_desc = "generally on the lighter side"
    elif avg_stress < 4:
        stress_desc = "moderate to elevated"
    else:
        stress_desc = "consistently high"

    return (
        f"Across {len(rows)} recent check-ins, your stress has been {stress_desc} "
        f"(average around {avg_stress:.1f} on a 1–5 scale). "
        f"Your coping style has most often looked {common_coping}. "
        "This is not a diagnosis or medical advice—just a simple reflection of patterns in your entries."
    )


def summarize_history_for_llm(rows: List[dict], max_items: int = 5) -> str:
    """Create a short textual summary of recent check-ins to feed into Gemini."""
    if not rows:
        return "No previous check-ins."

    recent = rows[-max_items:]
    parts = []
    for r in recent:
        parts.append(
            f"- {r['timestamp']}: stress {r['stress']}, mode {r['mode']}, coping {r['coping_mode']}, text: {r['text']}"
        )
    return "\n".join(parts)


# ================== LLM REFINEMENT (GEMINI) ==================

def maybe_gemini_refine_reply(user_text: str,
                              stress: int,
                              mode: str,
                              coping_mode: str,
                              draft_reply: str,
                              history_summary: str,
                              last_entry: Optional[dict]) -> str:

    """
    - If the user text is very short/vague: always return a short clarification question
      (no LLM needed, deterministic behavior).
    - Otherwise, if Gemini is available, refine the draft reply.
    - If Gemini is unavailable or fails, fall back to the draft reply.
    """

    # 1) Short / vague messages → deterministic clarification, no LLM
    short_or_vague = len(user_text.split()) < 3 or user_text.lower().strip() in {
        "ok", "fine", "idk", "i don't know", "i dont know",
        "nothing", "na", "n/a", "good", "not good", "bad"
    }

    if short_or_vague:
        print("Short or vague input detected, returning clarification prompt (no LLM):", user_text)
        return (
            "It sounds like something is off, but I don’t have much to go on yet. "
            "Could you share a bit more about what’s felt hardest or most stressful today?"
        )

    # 2) If no Gemini, just use the rule-based draft
    if gemini_model is None:
        print("Gemini not available (gemini_model is None). Using draft reply.")
        return draft_reply
        # Build a tiny recent transcript for conversational flow
    if last_entry is not None:
        last_user = last_entry.get("text", "")
        last_system = last_entry.get("reply", "")
        conversation_context = (
            "Last AWARE turn:\n"
            f"USER: {last_user}\n"
            f"AWARE: {last_system}\n"
            f"USER (now): {user_text}\n"
        )
    else:
        conversation_context = f"USER (now): {user_text}\n"
    # 3) Use Gemini to lightly refine the draft reply
    try:
        print("Gemini: refining draft reply for input:", user_text)
        prompt = f"""
You are AWARE, a lightweight well-being reflection tool for nurses.
You are NOT a therapist and must NOT give medical or mental health advice, diagnoses, or crisis instructions.
You only offer gentle reflection prompts.

User context:
- Stress level (1–5): {stress}
- Reflection mode: {mode}
- Detected coping style: {coping_mode}

Recent conversation (most recent first):
{conversation_context}

Recent check-ins (summary as simple bullet log):
{history_summary}

You are given a draft system reply created by simple rules:

DRAFT REPLY:
\"\"\"{draft_reply}\"\"\"

Your task:
- Treat the new user message as a continuation of the dialogue above.
- Keep the *meaning* and core structure of the draft, but rewrite it to sound a bit more natural and conversational.
- Explicitly acknowledge any clarification the user just gave (for example, if they say stress is “due to the work”, reflect that back).
- Stay supportive, concise, and non-judgmental.
- Do NOT add clinical or diagnostic language.
- Avoid promising outcomes; focus on reflection.
- Aim for about 3–5 sentences maximum.
- If the draft already looks fine, you may keep it with light edits.
- Do NOT mention that you are rewriting a draft.

Return only the final reply text.

"""
        resp = gemini_model.generate_content(prompt)
        refined = (resp.text or "").strip()
        if not refined:
            print("Gemini returned empty text, falling back to draft.")
            return draft_reply
        return refined
    except Exception as e:
        print("Gemini error in maybe_gemini_refine_reply:", e)
        return draft_reply



# ================== ROUTES ==================

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     """
#     Render main UI (AWARE check-in interface).
#     """
#     return templates.TemplateResponse("index.html", {"request": request})
# Simple demo users (replace with real auth or backend)
USERS = {
    os.environ.get("AWARE_DEMO_USER", "nurse"): os.environ.get("AWARE_DEMO_PASS", "password")
}
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.cookies.get("user_id")
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {"request": request, "user_id": user})

# Login / logout routes
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    expected = USERS.get(username)
    if expected and expected == password:
        resp = RedirectResponse("/", status_code=status.HTTP_302_FOUND)
        resp.set_cookie(key="user_id", value=username, httponly=True)
        return resp
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password."},
        status_code=401
    )

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=status.HTTP_302_FOUND)
    resp.delete_cookie("user_id")
    return resp

@app.post("/api/checkin", response_model=CheckinResponse)
async def checkin(payload: CheckinRequest):
    """
    Core mixed-initiative, adaptive check-in endpoint.
    Combines rule-based logic + optional Gemini refinement.
    """
    try:
        user_id = payload.user_id
        stress = payload.stress
        mode = payload.mode
        text = payload.text
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})

    # In-memory history for relational memory during this server run
    history = user_history.get(user_id, [])
    last_entry = history[-1] if history else None

    # Also load CSV rows so LLM can see brief history context
    csv_rows = load_user_rows_from_csv(user_id)
    history_summary = summarize_history_for_llm(csv_rows)

    # User modeling: coping mode (rule-based)
    coping_mode = classify_coping_mode(text)

    # Adaptive feedback using stress, mode, coping style, and memory
    draft_reply, used_memory = generate_feedback(stress, mode, coping_mode, text, last_entry)

    # LLM refinement (optional, safe fallback)
    final_reply = maybe_gemini_refine_reply(
        user_text=text,
        stress=stress,
        mode=mode,
        coping_mode=coping_mode,
        draft_reply=draft_reply,
        history_summary=history_summary,
        last_entry=last_entry,
    )


    # Update in-memory history
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "stress": stress,
        "mode": mode,
        "coping_mode": coping_mode,
        "text": text,
        "reply": final_reply,
        "used_memory": used_memory,
    }
    history.append(entry)
    user_history[user_id] = history

    # Log to CSV for evaluation
    log_interaction(user_id, stress, mode, coping_mode, text, final_reply, used_memory)

    return CheckinResponse(
        reply=final_reply,
        coping_mode=coping_mode,
        used_memory=used_memory
    )


@app.get("/api/history/{user_id}", response_model=List[HistoryItem])
async def get_history(user_id: str):
    """
    Return last few check-ins for a given participant.
    Reads from the CSV log so history persists across restarts.
    """
    rows = load_user_rows_from_csv(user_id)
    recent = rows[-5:]

    print(f"Returning {len(recent)} history items for '{user_id}'")

    return [
        HistoryItem(
            timestamp=row["timestamp"],
            stress=int(row["stress"]),
            mode=row["mode"],
            coping_mode=row["coping_mode"],
            text=row["text"],
            reply=row["reply"],
            used_memory=bool(int(row["used_memory"])),
        )
        for row in recent
    ]


@app.get("/api/pattern_summary/{user_id}", response_model=PatternSummaryResponse)
async def pattern_summary(user_id: str):
    """
    Use Gemini (if available) to summarize recent patterns in the user's check-ins.
    Falls back to a simple rule-based summary if Gemini is not available.
    """
    rows = load_user_rows_from_csv(user_id)
    if not rows:
        return PatternSummaryResponse(summary="No check-ins yet for this participant.")

    if gemini_model is None:
        return PatternSummaryResponse(summary=basic_pattern_summary(rows))

    try:
        history_text = summarize_history_for_llm(rows, max_items=8)
        prompt = f"""
You are AWARE, a simple reflection tool for nurses. You are NOT a clinician and must NOT give diagnoses,
medical advice, or crisis instructions. You only summarize patterns in stress and reflections in a gentle,
supportive way.

Here are recent check-ins from one nurse (each has timestamp, stress 1–5, mode, coping mode, and text):

{history_text}

Your task:
- Write a short summary (3–5 sentences) about patterns you see:
  - How stress has been trending (e.g., often high, mixed, etc.).
  - Any repeating situations or emotions (if visible).
  - Any small strengths or coping efforts you notice.
- Use simple, encouraging language.
- DO NOT give mental health advice.
- DO NOT mention that these are "rows" or "logs".
- This is not a diagnosis; keep it clearly reflective.

Return only the summary paragraph.
"""
        resp = gemini_model.generate_content(prompt)
        summary = (resp.text or "").strip()
        if not summary:
            summary = basic_pattern_summary(rows)
        return PatternSummaryResponse(summary=summary)
    except Exception:
        return PatternSummaryResponse(summary=basic_pattern_summary(rows))
