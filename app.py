import os
import random
import json
import secrets
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException

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
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Return simpler, user-friendly messages for validation errors.
    """
    errors = []
    for err in exc.errors():
        loc = err.get("loc", [])
        # loc example: ("body", "stress")
        field = None
        if len(loc) >= 2 and loc[0] == "body":
            field = loc[1]
        elif loc:
            field = loc[-1]

        msg = err.get("msg", "Invalid value.")
        if field in ("user_id", "stress", "mode", "text"):
            # Friendlier field names
            pretty = {
                "user_id": "Participant ID",
                "stress": "Stress",
                "mode": "Reflection mode",
                "text": "Reflection text",
            }.get(field, str(field))
            errors.append(f"{pretty}: {msg}")
        else:
            errors.append(msg)

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": errors or ["Invalid input."],
        },
    )
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log to console for debugging
    print("Unexpected server error:", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Sorry, something went wrong on the server."},
    )


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"  # root for JSON logs

# Mount static folder (for CSS/JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

DATA_DIR.mkdir(exist_ok=True)


# ================== IN-MEMORY STATE ==================

# Simple in-memory history per user_id (for relational memory during current run)
user_history: Dict[str, List[dict]] = {}
# Map (nurse_login, participant_id) -> session_id for this server run
session_tokens: Dict[tuple[str, str], str] = {}


# ================== Pydantic MODELS ==================

class CheckinRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    stress: int  # 1–5
    mode: str    # "Quick" | "Normal" | "Deep"
    text: str

    @validator("stress")
    def stress_in_range(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("Stress must be between 1 and 5.")
        return v

    @validator("mode")
    def normalize_mode(cls, v):
        allowed = {"Quick", "Normal", "Deep"}
        if v not in allowed:
            raise ValueError("Mode must be one of: Quick, Normal, or Deep.")
        return v

    @validator("text")
    def non_empty_text(cls, v):
        if not v.strip():
            raise ValueError("Please write a few words about how you’re doing.")
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


# ================== JSON STORAGE HELPERS ==================

def _safe_user_id(user_id: str) -> str:
    """Normalize user_id for filesystem paths."""
    s = (user_id or "").strip()
    safe = "".join(c for c in s if c.isalnum() or c in ("-", "_")).lower()
    return safe or "anonymous"


def get_user_dir(user_id: str) -> Path:
    """Return the folder for this participant's JSON logs."""
    safe = _safe_user_id(user_id)
    d = DATA_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_user_log_path(user_id: str) -> Path:
    """Single JSON file per participant containing a list of check-ins."""
    return get_user_dir(user_id) / "checkins.json"


def load_user_rows_from_json(user_id: str) -> List[dict]:
    """
    Load all check-ins for a participant from JSON.
    Returns [] if none exist yet.
    """
    path = get_user_log_path(user_id)
    if not path.exists():
        print(f"No JSON log yet for user '{user_id}'")
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"Loaded {len(data)} check-ins from JSON for user '{user_id}'")
                return data
            else:
                print(f"JSON log for '{user_id}' was not a list; resetting.")
                return []
    except Exception as e:
        print(f"Error reading JSON log for user '{user_id}':", e)
        return []


def append_checkin_to_json(user_id: str, entry: dict) -> None:
    """
    Append a single check-in entry to the participant's JSON log.
    Structure: a list of dicts, one per check-in.
    """
    path = get_user_log_path(user_id)
    rows = load_user_rows_from_json(user_id)
    rows.append(entry)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Appended check-in to JSON for user '{user_id}' (total now {len(rows)}).")
    except Exception as e:
        print(f"Error writing JSON log for user '{user_id}':", e)


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
    lines: List[str] = []

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


# ================== PATTERN & SUMMARY HELPERS ==================

def basic_pattern_summary(rows: List[dict]) -> str:
    """Simple rule-based pattern summary if LLM not available."""
    if not rows:
        return "No check-ins yet for this participant."

    stresses = [int(r["stress"]) for r in rows]
    avg_stress = sum(stresses) / len(stresses)

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



def simple_fallback_reply(user_text: str, stress: int) -> str:
    """Lightweight, neutral fallback used only when Gemini is unavailable or errors.

    Intentionally avoids the heavier rule-based template so positive or mixed
    messages don't get overwritten by a "demanding day" style reply.
    """

    lower = user_text.strip().lower()
    help_patterns = [
        "how to feel better",
        "how can i feel better",
        "how do i feel better",
        "any suggestions",
        "any advice",
        "what should i do",
        "what can i do",
        "how can i handle this",
        "suggest some ideas",
        "suggest some idea",
        "help me feel better",
        "ways to feel better",
        "ideas to feel good",
        "how to handle this",
    ]
    asking_for_help = any(p in lower for p in help_patterns)

    if asking_for_help:
        # When the user explicitly asks for suggestions, keep the same tone
        # but add 2–3 small, non-clinical options.
        if stress <= 3:
            return (
                "Thanks for checking in. From what you've shared, it sounds like things may be at least "
                "somewhat manageable right now, even if there are still ups and downs. Since you're looking "
                "for ways to feel a bit better, you could consider taking a brief pause for a few slow breaths, "
                "jotting down one thing that went even a little bit well today, or planning a small, enjoyable "
                "activity for later like a short walk or listening to music you like."
            )
        else:
            return (
                "Thanks for taking a moment to check in. It sounds like there is a fair amount on your plate "
                "right now, and wanting ideas to feel better is completely understandable. You might consider "
                "taking a short pause to step away from your tasks, focusing on three slow, grounding breaths, "
                "or checking in briefly with a trusted colleague or friend about how the day went. Even planning "
                "one simple, low-effort recovery activity for later—like a quiet moment with a drink you enjoy or "
                "a few minutes of stretching—can be a small step toward easing some of the strain."
            )

    if stress <= 3:
        return (
            "Thanks for checking in. From what you've shared, it sounds like things may be at least "
            "somewhat manageable right now, even if there are still ups and downs. If you'd like, you "
            "can add a bit more about what feels most important today, and you can use this space just "
            "to pause and notice how you're doing."
        )

    # For higher stress, keep a gentle, validating tone without going deep.
    return (
        "Thanks for taking a moment to check in. It sounds like there is a fair amount on your plate "
        "right now. You don't have to solve everything in this moment — even briefly noticing how "
        "you're feeling can be a small step toward deciding what support you might need next."
        )
def maybe_suppress_history_for_prompt(user_text: str, stress: int, history_summary: str) -> str:
    lower = user_text.strip().lower()
    words = lower.split()
    positive_tokens = {"good", "nice", "well", "okay", "ok", "fine", "grateful", "thankful"}

    if stress <= 3 and len(words) <= 8 and any(w in positive_tokens for w in words):
        return "Recent check-ins are omitted here so you can focus on how things are going today."

    return history_summary

def is_advice_seeking(user_text: str) -> bool:
    advice_seeking_phrases = [
        "how to feel better",
        "how can i feel better",
        "how do i feel better",
        "any suggestions",
        "any advice",
        "how can i fix this",
        "what should i do",
        "what can i do",
        "how can i handle this",
        "suggest some ideas",
        "suggest some idea",
        "help me feel better",
        "ways to feel better",
        "ideas to feel good",
        "how to handle this",
    ]
    lower_text = user_text.strip().lower()
    return any(p in lower_text for p in advice_seeking_phrases)

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
    if is_advice_seeking(user_text) :
        answer_mode= "focus on providing 1-2 small, concrete , non-clinical suggestions, not just reflection"  
        max_sentences= "1-2"
    else:
        answer_mode= "focus on reflection and support without giving advice"
        max_sentences= "3-5"

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
- Avoid promising outcomes; {answer_mode}.
- Aim for about {max_sentences} sentences maximum.
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

# Simple demo users (replace with real auth or backend)
USERS = {
    os.environ.get("AWARE_DEMO_USER", "nurse"): os.environ.get("AWARE_DEMO_PASS", "password")
}
def get_cookie_user(request: Request) -> Optional[str]:
    """
    Return the logged-in username from cookies, or None if not logged in.
    Treat empty string or things like 'None', 'null', 'undefined' as not logged in.
    """
    user = request.cookies.get("user_id")
    print("DEBUG user_id cookie =", repr(user))  # you can remove this later

    if not user:
        return None

    # normalize weird values
    if isinstance(user, str) and user.strip().lower() in {"none", "null", "undefined", '""', "''"}:
        return None

    return user

@app.get("/landing", response_class=HTMLResponse)
async def landing(request: Request):
    user = get_cookie_user(request)
    return templates.TemplateResponse("landing.html", {"request": request, "user_id": user})

@app.get("/", response_class=HTMLResponse)
async def landing_root(request: Request):
    user = get_cookie_user(request)
    return templates.TemplateResponse("landing.html", {"request": request, "user_id": user})

@app.get("/app", response_class=HTMLResponse)
async def app_home(request: Request):
    user = get_cookie_user(request)
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {"request": request, "user_id": user})


# Login / logout routes
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    user = get_cookie_user(request)
    if user:
        return RedirectResponse("/app")
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    user = get_cookie_user(request)
    return templates.TemplateResponse("about.html", {"request": request, "user_id": user})

@app.get("/flow", response_class=HTMLResponse)
async def flow(request: Request):
    user = get_cookie_user(request)
    return templates.TemplateResponse("workflow.html", {"request": request, "user_id": user})


@app.post("/login")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    expected = USERS.get(username)
    if expected and expected == password:
        resp = RedirectResponse("/app", status_code=status.HTTP_302_FOUND)
        # app-level user account (e.g., 'nurse')
        resp.set_cookie(key="user_id", value=username, httponly=True)
        # session_id groups all check-ins in this browser login session
        session_id = secrets.token_hex(16)
        resp.set_cookie(key="session_id", value=session_id, httponly=True)
        return resp

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password."},
        status_code=401
    )


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=status.HTTP_302_FOUND)
    resp.delete_cookie("user_id", path="/")
    resp.delete_cookie("session_id", path="/")
    resp.set_cookie(key="user_id", value="", max_age=0, expires=0, path="/", httponly=True)
    resp.set_cookie(key="session_id", value="", max_age=0, expires=0, path="/", httponly=True)
    return resp




def llm_generate_reply_for_checkin(
    user_text: str,
    stress: int,
    mode: str,
    coping_mode: str,
    history_summary: str,
    last_entry: Optional[dict],
) -> str:
    """
    Gemini is now the PRIMARY conversation engine.

    It should:
    - Give empathetic, concise reflection replies.
    - Make Quick / Normal / Deep clearly different in length and number of questions:
        * Quick: 2–3 short sentences, no questions unless the message is extremely short/vague.
        * Normal: 3–4 sentences, at most 1 gentle question.
        * Deep: 4–5 sentences, 1–2 reflective questions.
    - Never give diagnoses or clinical advice.
    """

    # Fallback if LLM is not available: use a lightweight neutral reply
    if gemini_model is None:
        return simple_fallback_reply(user_text, stress)

    # Deterministic handling for very short greetings only; all other cases go to Gemini.
    lower_text = user_text.strip().lower()
    words = lower_text.split()
    greeting_words = {"hi", "hey", "hello"}

    if len(words) <= 2 and lower_text in greeting_words:
        return (
            "Hi, and thanks for checking in. When you’re ready, you can share a sentence or two "
            "about how your shift or day is going, and I’ll offer a brief reflection."
        )

    # Build brief conversation context for continuity
    if last_entry is not None:
        last_user = last_entry.get("text", "")
        last_system = last_entry.get("reply", "")
        conversation_context = (
            "Previous AWARE turn:\n"
            f"USER: {last_user}\n"
            f"AWARE: {last_system}\n\n"
        )
    else:
        conversation_context = ""
    if is_advice_seeking(lower_text) :
        answer_mode= "focus on providing 1-2 small, concrete , non-clinical suggestions, not just reflection"  
        max_sentences= "1-2"
    else:
        answer_mode= "focus on reflection and support without giving advice"
        max_sentences= "3-5"
    try:
        prompt = f"""
You are AWARE, a lightweight well-being reflection tool for nurses.
You are NOT a therapist and must NOT give medical or mental health advice,
diagnoses, or crisis instructions. You only offer gentle reflection prompts
and supportive observations.

Context:
- Stress level (1–5): {stress}
- Reflection mode: {mode}  (Quick / Normal / Deep)
- Detected coping style (for the clinician/researcher, not user-facing term): {coping_mode}

Recent check-ins (summary, newest last):
{history_summary}

{conversation_context}User's current message:
\"\"\"{user_text}\"\"\"

Your task:

1. First, use the message together with the stress rating to infer the overall emotional tone
   and how heavy the day feels in plain language. Internally, you can think about whether
   the user sounds:
   - mostly steady or okay,
   - mixed (tired or pressured but still coping),
   - or heavily strained and worn down.
   You must NOT diagnose anything.

2. Decide which of these broad categories the current check-in fits best:
   - Clearly or mostly positive / stable (for example, the user describes things as going
     fine, manageable, or good, and stress is low or moderate).
   - Mixed or mildly strained (some stress or frustration, but also some positives or
     coping, and the user does not sound overwhelmed).
   - High strain (for example, strong fatigue, feeling drained, or feeling ineffective,
     especially when stress is high).

   Additional rule for detecting MIXED tone:
   - Treat the message as MIXED if it contains both:
       1) at least one fatigue or strain indicator (for example: tired, exhausting,
          demanding, stressful, overwhelming), AND
       2) at least one coping or stability indicator (for example: "I’m coping",
          "I’m managing", "I’m alright", "I’m okay", "I’m holding up").
   - If both appear in the same message, classify it as MIXED even if the numeric stress
     rating alone might look low or high.

3. Respond accordingly:
   - If the message is clearly or mostly positive *right now* (for example "life is nice",
     "today has been pretty good", or "work is busy but mostly okay"):
       * Treat this check-in as positive even if some past entries sounded stressed.
       * Briefly acknowledge and reinforce what seems to be going okay.
       * Keep the reply light and concise (about 2–3 sentences).
       * Do NOT describe the day as heavy, demanding, or very stressful unless the
         user clearly said so in this message.
       * Do NOT introduce problems or worries that the user did not mention.
       * You may gently invite further reflection only as an option, not a requirement.
   - If the message has a mixed emotional tone (for example, the user mentions
     fatigue, pressure, or emotional strain *together with* statements like
     "I’m coping", "I’m managing", "I’m alright", or "it’s not too bad"):
       * Write 2–3 sentences.
       * Acknowledge that something feels tiring or heavy.
       * Affirm that the user is still managing and maintaining some control.
       * Keep the tone steady and supportive without escalating the stress.
       * Avoid offering deep-dive reflection unless the user clearly asks for it.
       * In effect, reflect back the tiring part, reinforce the coping part, and gently
         highlight that it’s okay to hold both at once.
   - If the message suggests high strain:
       * Validate that things sound demanding or draining.
       * Use a calm, compassionate tone without clinical labels.
       * Offer 3–5 sentences, including at most 1–2 gentle questions that invite the
         user to notice what is most draining and what small support or boundary might
         help a little.

4. Explicitly handle direct requests for help, advice, or suggestions. If the user asks
   things like "What should I do?", "Any advice?", "Any suggestions?", "How can I handle
   this?", or similar:
   - Still follow the appropriate category above (positive, mixed, or high strain).
   - In addition, include 2–3 small, concrete, non-clinical options they *might* consider.
     Examples of options you can use:
       * taking a short pause or breathing break,
       * jotting down one thing that went okay in the shift,
       * briefly checking in with a trusted colleague,
       * setting one small boundary for the rest of the day,
       * planning one simple recovery activity after work (like a walk, music, or quiet time).
   - Phrase these clearly as gentle possibilities (e.g., "You could consider…", "One option
     might be…") rather than strong instructions.
   - Do NOT suggest medication, diagnosis, treatment, or anything that sounds like clinical care.

5. Make Quick / Normal / Deep clearly different in how you answer:
   - For Quick:
       * Write 2–3 short sentences.
       * Do NOT ask follow-up questions unless the message is extremely short or vague.
   - For Normal:
       * Write 3–4 sentences total.
       * You may ask at most ONE gentle question.
   - For Deep:
       * Write 4–5 sentences total.
       * You may include 1–2 reflective questions if they feel natural.

6. Reduce repetition and help the user know what to say:
   - In your reply, paraphrase at least ONE concrete detail from the user’s latest message
     so they can recognize that you really heard them (for example, mention their shift,
     feeling "off", feeling drained, or something they said they were unsure about).
   - If you invite the user to share more, include 2–3 example options of what they could
     talk about (for example: "what felt most stressful", "what went even a little bit well",
     or "what you’re most worried about next"), phrased briefly and clearly.

7. Throughout, avoid clinical or diagnostic language, do not promise outcomes, and do not
   give crisis instructions. Stay focused on reflection, noticing patterns, and small,
   realistic steps.

8. You may gently connect to patterns from recent check-ins if it feels natural, but do
   not overload the answer with history.

    Return ONLY the reply text, with no labels, headings, or extra formatting.
    """

        resp = gemini_model.generate_content(prompt)
        reply = (resp.text or "").strip()
        if not reply:
            # If Gemini returns nothing, fall back to simple neutral reply
            return simple_fallback_reply(user_text, stress)
        return reply
    except Exception as e:
        print("Gemini error in llm_generate_reply_for_checkin:", e)
        # Fallback to simple neutral reply if LLM fails
        return simple_fallback_reply(user_text, stress)


# ================== HISTORY & PATTERN HELPERS ==================

def load_user_rows_from_csv(user_id: str) -> list:
    """
    Compatibility helper so newer code still works.

    Now we simply delegate to the JSON-based loader so that
    LLM context includes persisted history, not just this run.
    """
    return load_user_rows_from_json(user_id)

def log_interaction(
    user_id: str,
    stress: int,
    mode: str,
    coping_mode: str,
    text: str,
    reply: str,
    used_memory: bool,
    final_reply_length: Optional[int] = None
) -> None:
    """
    Append one check-in to the participant's JSON log.

    This is the JSON version of the old CSV logger.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "stress": stress,
        "mode": mode,
        "coping_mode": coping_mode,
        "text": text,
        "reply": reply,
        "used_memory": used_memory,
        "reply_length": final_reply_length
    }
    append_checkin_to_json(user_id, entry)


@app.post("/api/checkin", response_model=CheckinResponse)
async def checkin(payload: CheckinRequest):
    """
    Core mixed-initiative, adaptive check-in endpoint.
    NOW: the main reply is generated entirely by Gemini (LLM),
    with rule-based code used only for coping-mode tagging and fallback.
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

    # Load CSV/JSON rows so LLM can see brief history context
    csv_rows = load_user_rows_from_csv(user_id)
    history_summary = summarize_history_for_llm(csv_rows)
    history_summary = maybe_suppress_history_for_prompt(text, stress, history_summary)

    # User modeling: coping mode (still rule-based for tagging)
    coping_mode = classify_coping_mode(text)

    # For logging: we consider "memory available" if there was a last entry
    used_memory = last_entry is not None

    # LLM is the primary conversation engine
    final_reply = llm_generate_reply_for_checkin(
        user_text=text,
        stress=stress,
        mode=mode,
        coping_mode=coping_mode,
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

    # Log to CSV/JSON for evaluation
    log_interaction(user_id, stress, mode, coping_mode, text, final_reply, used_memory, final_reply_length=len(final_reply))

    return CheckinResponse(
        reply=final_reply,
        coping_mode=coping_mode,
        used_memory=used_memory,
    )


@app.get("/api/history/{user_id}", response_model=List[HistoryItem])
async def get_history(user_id: str):
    """
    Return last few check-ins for a given participant.
    Reads from the per-user JSON log so history persists across restarts.
    """
    rows = load_user_rows_from_json(user_id)
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
            used_memory=bool(row.get("used_memory", False)),
        )
        for row in recent
    ]


@app.get("/api/pattern_summary/{user_id}", response_model=PatternSummaryResponse)
async def pattern_summary(user_id: str):
    """
    Use Gemini (if available) to summarize recent patterns in the user's check-ins.
    Falls back to a simple rule-based summary if Gemini is not available.
    Data is read from the per-user JSON log.
    """
    rows = load_user_rows_from_json(user_id)
    if not rows:
        return PatternSummaryResponse(summary="No check-ins yet for this participant.")

    if gemini_model is None:
        return PatternSummaryResponse(summary=basic_pattern_summary(rows))

    try:
        history_text = summarize_history_for_llm(rows, max_items=8)
        latest = rows[-1]
        latest_stress = latest.get("stress", "?")
        latest_text = latest.get("text", "")

        prompt = f"""
You are AWARE, a simple reflection tool for nurses. You are NOT a clinician and must NOT give diagnoses,
medical advice, or crisis instructions. You only summarize patterns in stress and reflections in a gentle,
supportive way.

Here are recent check-ins from one nurse (each has timestamp, stress 1–5, mode, coping mode, and text):

{history_text}

Most recent check-in (latest):
- Stress: {latest_stress}
- Text: "{latest_text}"

Your task:
- First, pay close attention to the *current* mood in the most recent check-ins, especially the latest one.
  - If the most recent check-ins sound clearly or mostly positive or stable, the overall summary should
    explicitly acknowledge that things seem to be going at least somewhat okay *now*, even if older
    entries were more strained.
  - If the most recent check-ins sound more strained or exhausted, the summary should reflect that current
    strain while still noticing any strengths or coping efforts.
- Then write a short summary (3–5 sentences) about patterns you see over time:
  - How stress has been trending (e.g., often high, mixed, improving, etc.).
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
