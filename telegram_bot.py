"""
Telegram Bot Coach for Garmin Connect powered by Gemini.

Allows chatting with your AI Running & Performance Coach from your mobile phone,
with full access to your live Garmin Connect metrics and activity history.
"""

import io
import os
import sys
import json
import logging
import asyncio
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# Timezone configuration (Configurable via TIMEZONE env variable)
TZ_NAME = os.getenv("TIMEZONE", os.getenv("LOCAL_TZ", "America/Mexico_City"))
try:
    LOCAL_TZ = ZoneInfo(TZ_NAME)
except Exception:
    LOCAL_TZ = ZoneInfo("America/Mexico_City")

from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from garmin_mcp import _init_garmin_client
from garmin_mcp import (
    health_wellness,
    activity_management,
    training,
    devices,
    gear_management,
    weight_management,
    workouts,
    user_profile,
)

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("GarminCoachBot")

# Start healthcheck server immediately for Render Free Tier port detection
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Garmin Coach Bot is alive!")
    def log_message(self, format, *args):
        pass

def start_health_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Healthcheck server listening on port {port}")
    server.serve_forever()

# The healthcheck server is started conditionally in main()

# Initialize Garmin client
garmin_client = _init_garmin_client()

# Configure Garmin modules
for mod in [
    health_wellness,
    activity_management,
    training,
    devices,
    gear_management,
    weight_management,
    workouts,
    user_profile,
]:
    mod.configure(garmin_client)

# Gemini API Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = os.getenv("TELEGRAM_ALLOWED_USER_ID")  # Optional security filter

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set in environment or .env!")

if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN not set in environment or .env!")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Tools definition for Gemini Function Calling
def get_today_date() -> str:
    """Returns today's date in YYYY-MM-DD format based on Guadalajara, Jalisco local time."""
    return datetime.datetime.now(LOCAL_TZ).date().isoformat()

def get_daily_stats(date: str) -> str:
    """Get daily health and activity statistics (steps, calories, HR, stress, body battery) for a date (YYYY-MM-DD)."""
    return json.dumps(garmin_client.get_stats(date), default=str)

def get_sleep_data(date: str) -> str:
    """Get sleep duration, sleep stages (deep, light, REM, awake), sleep score and SpO2 for a date (YYYY-MM-DD)."""
    return json.dumps(garmin_client.get_sleep_data(date), default=str)

def get_training_readiness(date: str) -> str:
    """Get Garmin training readiness score and contributing recovery factors for a date (YYYY-MM-DD)."""
    return json.dumps(garmin_client.get_training_readiness(date), default=str)

def get_training_status(date: str) -> str:
    """Get training status (Productive, Maintaining, Recovery, etc.), acute load and load balance for a date (YYYY-MM-DD)."""
    return json.dumps(garmin_client.get_training_status(date), default=str)

def get_hrv_data(date: str) -> str:
    """Get heart rate variability (HRV) nocturnal metrics and balance for a date (YYYY-MM-DD)."""
    return json.dumps(garmin_client.get_hrv_data(date), default=str)

def get_recent_activities(limit: int = 10) -> str:
    """Get the most recent activities (swimming, running, cycling, gym, etc.) with metrics (distance, duration, HR, load)."""
    acts = garmin_client.get_activities(0, min(limit, 20))
    summary = []
    for a in acts:
        summary.append({
            "id": a.get("activityId"),
            "name": a.get("activityName"),
            "type": a.get("activityType", {}).get("typeKey"),
            "date": a.get("startTimeLocal"),
            "distance_km": round((a.get("distance", 0) or 0) / 1000.0, 2),
            "duration_min": round((a.get("duration", 0) or 0) / 60.0, 1),
            "avg_hr": a.get("averageHR"),
            "max_hr": a.get("maxHR"),
            "calories": a.get("calories"),
            "training_load": a.get("activityTrainingLoad"),
            "aerobic_training_effect": a.get("aerobicTrainingEffect"),
        })
    return json.dumps(summary, default=str)

def get_activity_details(activity_id: int) -> str:
    """Get comprehensive details and splits for a specific activity ID."""
    return json.dumps(garmin_client.get_activity(activity_id), default=str)

def get_body_battery(date: str) -> str:
    """Get body battery charged, drained and current values for a date (YYYY-MM-DD)."""
    return json.dumps(garmin_client.get_body_battery(date), default=str)

def get_fitness_scores() -> str:
    """Get VO2 max, endurance score, and hill score."""
    today = datetime.datetime.now(LOCAL_TZ).date().isoformat()
    res = {}
    try: res["max_metrics"] = garmin_client.get_max_metrics(today)
    except: pass
    try: res["endurance_score"] = garmin_client.get_endurance_score(today)
    except: pass
    try: res["hill_score"] = garmin_client.get_hill_score(today)
    except: pass
    try: res["race_predictions"] = garmin_client.get_race_predictions()
    except: pass
    return json.dumps(res, default=str)

def get_saved_workouts(limit: int = 15) -> str:
    """Get list of saved workouts in Garmin Connect."""
    try:
        workouts = garmin_client.get_workouts(0, limit)
        summary = []
        for w in (workouts or []):
            summary.append({
                "workout_id": w.get("workoutId"),
                "name": w.get("workoutName"),
                "sport": w.get("sportType", {}).get("sportTypeKey"),
                "estimated_duration_min": round((w.get("estimatedDurationInSecs", 0) or 0) / 60.0, 1),
            })
        return json.dumps(summary, default=str)
    except Exception as e:
        return f"Error retrieving workouts: {e}"

def schedule_existing_workout(workout_id: int, date: str) -> str:
    """Schedule an existing workout ID to a specific date (YYYY-MM-DD) on Garmin calendar/watch."""
    try:
        res = garmin_client._client.schedule_workout(workout_id, date)
        return json.dumps({"status": "success", "workout_id": workout_id, "scheduled_date": date, "response": res}, default=str)
    except Exception as e:
        return f"Error scheduling workout: {e}"

def create_running_interval_workout(name: str, warmup_min: int = 10, interval_min: int = 3, recovery_min: int = 2, repetitions: int = 5, cooldown_min: int = 5, schedule_date: str = "") -> str:
    """Create a structured running interval workout (Warmup -> Repetitions of Interval/Recovery -> Cooldown) and upload to Garmin Connect.
    Optionally schedule it on schedule_date (YYYY-MM-DD) to sync with Garmin watch.
    """
    try:
        payload = {
            "workoutName": name,
            "description": f"Entrenamiento estructurado por tu Coach: {repetitions}x{interval_min}min con {recovery_min}min rec",
            "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
            "workoutSegments": [{
                "segmentOrder": 1,
                "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
                "workoutSteps": [
                    {
                        "type": "ExecutableStepDTO",
                        "stepOrder": 1,
                        "stepType": {"stepTypeId": 1, "stepTypeKey": "warmup"},
                        "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                        "endConditionValue": float(warmup_min * 60),
                    },
                    {
                        "type": "RepeatGroupDTO",
                        "stepOrder": 2,
                        "stepType": {"stepTypeId": 6, "stepTypeKey": "repeat"},
                        "numberOfIterations": repetitions,
                        "smartRepeat": False,
                        "workoutSteps": [
                            {
                                "type": "ExecutableStepDTO",
                                "stepOrder": 1,
                                "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
                                "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                                "endConditionValue": float(interval_min * 60),
                            },
                            {
                                "type": "ExecutableStepDTO",
                                "stepOrder": 2,
                                "stepType": {"stepTypeId": 4, "stepTypeKey": "recovery"},
                                "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                                "endConditionValue": float(recovery_min * 60),
                            }
                        ]
                    },
                    {
                        "type": "ExecutableStepDTO",
                        "stepOrder": 3,
                        "stepType": {"stepTypeId": 2, "stepTypeKey": "cooldown"},
                        "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                        "endConditionValue": float(cooldown_min * 60),
                    }
                ]
            }]
        }
        res = garmin_client._client.upload_workout(json.dumps(payload))
        w_id = res.get("workoutId")
        result_info = {"status": "created", "workout_id": w_id, "workout_name": res.get("workoutName")}
        
        if schedule_date and w_id:
            sched_res = garmin_client._client.schedule_workout(w_id, schedule_date)
            result_info["scheduled_date"] = schedule_date
            result_info["scheduled_status"] = "success"
            
        return json.dumps(result_info, default=str)
    except Exception as e:
        return f"Error creating running interval workout: {e}"

def create_running_base_workout(name: str, duration_minutes: int, schedule_date: str = "", notes: str = "") -> str:
    """Create a continuous running workout (Base / Z2 / Long Run / Tempo) and upload to Garmin Connect.
    Optionally schedule it on schedule_date (YYYY-MM-DD) to sync directly to the watch.
    """
    try:
        payload = {
            "workoutName": name,
            "description": notes or f"Carrera aeróbica continua de {duration_minutes} min guiada por tu Coach",
            "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
            "workoutSegments": [{
                "segmentOrder": 1,
                "sportType": {"sportTypeId": 1, "sportTypeKey": "running"},
                "workoutSteps": [
                    {
                        "type": "ExecutableStepDTO",
                        "stepOrder": 1,
                        "stepType": {"stepTypeId": 3, "stepTypeKey": "interval"},
                        "endCondition": {"conditionTypeId": 2, "conditionTypeKey": "time"},
                        "endConditionValue": float(duration_minutes * 60),
                    }
                ]
            }]
        }
        res = garmin_client._client.upload_workout(json.dumps(payload))
        w_id = res.get("workoutId")
        result_info = {"status": "created", "workout_id": w_id, "workout_name": res.get("workoutName")}
        
        if schedule_date and w_id:
            sched_res = garmin_client._client.schedule_workout(w_id, schedule_date)
            result_info["scheduled_date"] = schedule_date
            result_info["scheduled_status"] = "success"
            
        return json.dumps(result_info, default=str)
    except Exception as e:
        return f"Error creating running base workout: {e}"

def get_heart_rate_zones() -> str:
    """Get exact configured Heart Rate (HR) zones, lactate threshold, and resting/max HR from the user's Garmin profile."""
    try:
        zones = garmin_client._client.get_heart_rate_zones()
        return json.dumps(zones, default=str)
    except Exception as e:
        return f"Error retrieving HR zones: {e}"

def get_weekly_training_summary(weeks_ago: int = 0) -> str:
    """Get accumulated weekly training volume (swimming km/sessions, running km/sessions, gym sessions, and total training load).
    weeks_ago: 0 for current week (starting Monday), 1 for previous week, etc.
    """
    try:
        today = datetime.datetime.now(LOCAL_TZ).date()
        # Find the Monday of the requested week
        target_monday = today - datetime.timedelta(days=today.weekday() + (weeks_ago * 7))
        target_sunday = target_monday + datetime.timedelta(days=6)
        
        acts = garmin_client.get_activities(0, 50)
        swimming = {"km": 0.0, "sessions": 0, "minutes": 0.0, "load": 0.0}
        running = {"km": 0.0, "sessions": 0, "minutes": 0.0, "load": 0.0}
        gym = {"sessions": 0, "minutes": 0.0, "load": 0.0}
        cycling = {"km": 0.0, "sessions": 0, "minutes": 0.0, "load": 0.0}
        total_load = 0.0

        for a in (acts or []):
            dt_str = (a.get("startTimeLocal") or "")[:10]
            if target_monday.isoformat() <= dt_str <= target_sunday.isoformat():
                tp = (a.get("activityType", {}).get("typeKey") or "").lower()
                dist = (a.get("distance", 0) or 0) / 1000.0
                dur = (a.get("duration", 0) or 0) / 60.0
                ld = a.get("activityTrainingLoad", 0) or 0
                total_load += ld

                if "swim" in tp or "pool" in tp:
                    swimming["km"] += dist
                    swimming["sessions"] += 1
                    swimming["minutes"] += dur
                    swimming["load"] += ld
                elif "run" in tp or "treadmill" in tp:
                    running["km"] += dist
                    running["sessions"] += 1
                    running["minutes"] += dur
                    running["load"] += ld
                elif "strength" in tp or "gym" in tp or "fitness" in tp:
                    gym["sessions"] += 1
                    gym["minutes"] += dur
                    gym["load"] += ld
                elif "cycl" in tp or "bike" in tp:
                    cycling["km"] += dist
                    cycling["sessions"] += 1
                    cycling["minutes"] += dur
                    cycling["load"] += ld

        return json.dumps({
            "week_start_monday": target_monday.isoformat(),
            "week_end_sunday": target_sunday.isoformat(),
            "swimming": {k: round(v, 2) for k, v in swimming.items()},
            "running": {k: round(v, 2) for k, v in running.items()},
            "gym": {k: round(v, 2) for k, v in gym.items()},
            "cycling": {k: round(v, 2) for k, v in cycling.items()},
            "total_weekly_training_load": round(total_load, 1),
        }, default=str)
    except Exception as e:
        return f"Error retrieving weekly summary: {e}"

COACH_TOOLS = [
    get_today_date,
    get_daily_stats,
    get_sleep_data,
    get_training_readiness,
    get_training_status,
    get_hrv_data,
    get_heart_rate_zones,
    get_weekly_training_summary,
    get_recent_activities,
    get_activity_details,
    get_body_battery,
    get_fitness_scores,
    get_saved_workouts,
    schedule_existing_workout,
    create_running_interval_workout,
    create_running_base_workout,
]

def build_system_instruction() -> str:
    athlete_name = os.getenv("ATHLETE_NAME", os.getenv("COACH_USER_NAME", "Athlete"))
    device_model = os.getenv("GARMIN_DEVICE_MODEL", "Garmin watch")
    sport_focus = os.getenv("COACH_SPORT_FOCUS", "Running, swimming, cycling, strength, and hybrid performance")
    custom_prompt = os.getenv("COACH_CUSTOM_PROMPT", "").strip()
    profile_file = os.getenv("COACH_PROFILE_FILE", "coach_profile.txt")

    profile_text = ""
    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_text = f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read athlete profile file {profile_file}: {e}")

    base = f"""You are the elite AI Performance & Endurance Sports Coach for {athlete_name} (using a {device_model}).
Primary athletic focus: {sport_focus}.

CORE PRINCIPLES (NEVER ASSUME & USE REAL DATA):
1. NEVER assume fixed values for VO2 Max, Heart Rate (HR), distances, paces, sleep duration, or completed sessions.
2. NEVER assume physical sensations, soreness, perceived energy, or time availability.
3. ALWAYS query your Garmin Connect tools first to retrieve objective metrics (sleep stages/score, nocturnal HRV, resting HR, exact HR zones, recent activities) and correlate them with whatever the athlete reports.
4. If you have any doubt regarding time availability, terrain, or muscular fatigue, ASK DIRECTLY in a concise manner.
5. Language: Respond in the athlete's preferred language (match the language of their message or voice note).

3-BLOCK RESPONSE PROTOCOL (OPTIMIZED FOR MOBILE):
When the athlete asks for daily status or workout guidance, structure your response as:
1. 📊 **Daily Diagnosis**: Ultra-concise summary of today's real biometrics (Total/deep sleep, nocturnal HRV, resting HR, Body Battery, Training Readiness).
2. 🎯 **Recommended Session**: Detailed workout with exact Heart Rate Zones (Z1 to Z5) and phase durations (Warmup, Intervals, Recovery, Cooldown).
3. 💬 **Check-in & Action**: Interactive question regarding muscle feel/time and confirmation to schedule it on the watch.

RECOVERY ALERTS (INJURY & OVERTRAINING PREVENTION):
- If you detect poor sleep (<6.5h), unbalanced/low HRV, or Body Battery < 50: Proactively recommend scaling down (active recovery or gentle Z1).

POST-WORKOUT VOICE NOTES:
- When receiving a voice note after training, immediately query `get_recent_activities(limit=1)` to compare what the athlete describes with the actual watch data (pace, avg/max HR, and Training Load).

GARMIN WATCH WORKOUT SCHEDULING:
- Use `create_running_interval_workout` or `create_running_base_workout` with `schedule_date` when the athlete asks to schedule the workout onto their {device_model}.

TELEGRAM FORMATTING RULES:
- NEVER use markdown hashtags (#, ##, ###).
- ALWAYS use bold text with double asterisks (**Title**) for headings and key metrics.
- Use clean bullet points (•).
"""
    if profile_text:
        base += f"\n\nATHLETE PROFILE, WEEKLY STRUCTURE & PREFERENCES:\n{profile_text}\n"
    elif custom_prompt:
        base += f"\n\nCUSTOM INSTRUCTIONS & GOALS:\n{custom_prompt}\n"

    return base

SYSTEM_INSTRUCTION = build_system_instruction()

def format_for_telegram(text: str) -> str:
    """Clean markdown headings and format nicely for Telegram Markdown."""
    import re
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        m = re.match(r"^\s*#{1,6}\s+(.*)$", line)
        if m:
            title = m.group(1).strip().replace("**", "").replace("*", "")
            cleaned_lines.append(f"*{title}*")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# Store chat sessions and target user
user_chats = {}
SAVED_CHAT_FILE = ".telegram_chat_id"

def save_chat_id(chat_id: int):
    try:
        with open(SAVED_CHAT_FILE, "w") as f:
            f.write(str(chat_id))
    except Exception:
        pass

def load_chat_id() -> int | None:
    try:
        if os.path.exists(SAVED_CHAT_FILE):
            with open(SAVED_CHAT_FILE, "r") as f:
                val = f.read().strip()
                if val:
                    return int(val)
    except Exception:
        pass
    return int(ALLOWED_USER_ID) if ALLOWED_USER_ID else None

# Multi-model Failover Cascade for 100% Uptime and No Rate Limits
FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",     # 500 RPD / 15 RPM
    "gemini-3.1-flash-lite",     # 500 RPD / 15 RPM
    "gemini-flash-lite-latest",  # 500 RPD / 15 RPM
    "gemini-3.5-flash",          # 20 RPD fallback
    "gemini-3.7-flash",          # 20 RPD fallback
]

user_active_model_idx = {}
user_chats = {}

def get_chat_for_model(user_id: int, model_name: str):
    key = f"{user_id}_{model_name}"
    if key not in user_chats:
        chat = gemini_client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=COACH_TOOLS,
                temperature=0.7,
            ),
        )
        user_chats[key] = chat
    return user_chats[key]

def execute_with_failover(user_id: int, message_or_parts: Any) -> Any:
    """Send message to Gemini with automatic failover to alternative models if rate limits (429) or 503 occur."""
    start_idx = user_active_model_idx.get(user_id, 0)
    last_err = None

    for offset in range(len(FALLBACK_MODELS)):
        idx = (start_idx + offset) % len(FALLBACK_MODELS)
        model_name = FALLBACK_MODELS[idx]
        try:
            chat = get_chat_for_model(user_id, model_name)
            response = chat.send_message(message_or_parts)
            user_active_model_idx[user_id] = idx  # Keep using successful model
            return response
        except Exception as e:
            err_str = str(e)
            logger.warning(f"Model {model_name} failed ({err_str[:90]}). Failing over to next model...")
            last_err = e
            continue

    logger.error(f"All fallback models failed! Last error: {last_err}")
    raise last_err


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    if ALLOWED_USER_ID and str(user.id) != str(ALLOWED_USER_ID):
        await update.message.reply_text("⛔ Unauthorized access to this coach bot.")
        return

    save_chat_id(update.effective_chat.id)
    device_model = os.getenv("GARMIN_DEVICE_MODEL", "your Garmin watch")

    welcome_text = (
        f"Hello {user.first_name}! 🏃‍♂️💪\n\n"
        "I am your **Personal Garmin AI Coach**, connected live to your Garmin Connect data.\n\n"
        "I can help you with:\n"
        "• 🫀 **Analyzing your Training Readiness, HRV & Sleep score.**\n"
        "• 🏃‍♂️ **Reviewing your recent activities and Training Load balance.**\n"
        f"• ⏱️ **Planning and scheduling workouts directly onto your {device_model}.**\n"
        "• 📊 **Weekly volume and sport breakdown with /week.**\n"
        "• ☀️ **Generating your daily morning briefing with /briefing.**\n"
        "• 🎙️ **You can also send me voice notes right after your training session!**\n\n"
        "How are you feeling today?"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate on-demand Morning Briefing."""
    user = update.effective_user
    if ALLOWED_USER_ID and str(user.id) != str(ALLOWED_USER_ID):
        return

    save_chat_id(update.effective_chat.id)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    prompt = (
        "Generate my **Morning Briefing** for today. "
        "Check last night's sleep (deep/REM/score), nocturnal HRV, resting HR, Body Battery, and Training Readiness. "
        "Provide a quick diagnostic, workout recommendation based on recovery state, and ask about physical sensations and energy levels."
    )
    try:
        response = await asyncio.to_thread(execute_with_failover, user.id, prompt)
        if response and response.text:
            formatted = format_for_telegram(response.text)
            try:
                await update.message.reply_text(formatted, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Error in briefing: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Temporary AI service issue. Please try again in a moment.")


async def weekly_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate on-demand weekly summary."""
    user = update.effective_user
    if ALLOWED_USER_ID and str(user.id) != str(ALLOWED_USER_ID):
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    prompt = (
        "Generate a breakdown and analysis of my training volume and accumulated load for this week using `get_weekly_training_summary`. "
        "Break down distance, sessions, and total Training Load by sport, and provide coaching feedback."
    )
    try:
        response = await asyncio.to_thread(execute_with_failover, user.id, prompt)
        if response and response.text:
            formatted = format_for_telegram(response.text)
            try:
                await update.message.reply_text(formatted, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Error in weekly: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Temporary AI service issue. Please try again in a moment.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages from the user."""
    user = update.effective_user
    if ALLOWED_USER_ID and str(user.id) != str(ALLOWED_USER_ID):
        return

    save_chat_id(update.effective_chat.id)
    user_text = update.message.text
    if not user_text:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        response = await asyncio.to_thread(execute_with_failover, user.id, user_text)
        if response and response.text:
            formatted = format_for_telegram(response.text)
            try:
                await update.message.reply_text(formatted, parse_mode="Markdown")
            except Exception as err:
                logger.warning(f"Markdown parse failed ({err}), falling back to plain text")
                await update.message.reply_text(response.text)
        else:
            await update.message.reply_text("⚠️ No response received. Please try again.")
    except Exception as e:
        logger.error(f"Error handling message after failover: {e}", exc_info=True)
        await update.message.reply_text("⚠️ AI service experienced temporary high demand. Please resend your message in a few seconds.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice notes from the user."""
    user = update.effective_user
    if ALLOWED_USER_ID and str(user.id) != str(ALLOWED_USER_ID):
        return

    save_chat_id(update.effective_chat.id)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        voice = update.message.voice
        voice_file = await context.bot.get_file(voice.file_id)
        voice_bytes = await voice_file.download_as_bytearray()

        audio_part = types.Part.from_bytes(
            data=bytes(voice_bytes),
            mime_type="audio/ogg",
        )

        response = execute_with_failover(
            user.id,
            [audio_part, "Listen to my voice note and respond as my performance sports coach with access to my Garmin metrics."]
        )
        if response and response.text:
            formatted = format_for_telegram(response.text)
            try:
                await update.message.reply_text(formatted, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Error handling voice note: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Error processing voice note: {e}")


from telegram import BotCommand

async def post_init(application):
    """Automatically register bot commands menu in Telegram UI."""
    commands = [
        BotCommand("start", "Start or restart the Coach"),
        BotCommand("briefing", "☀️ Morning diagnosis (sleep, HRV, readiness)"),
        BotCommand("week", "📊 Weekly volume & training load breakdown"),
        BotCommand("semana", "📊 Resumen semanal (Spanish alias)"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Registered Telegram menu commands successfully.")

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: Please set TELEGRAM_BOT_TOKEN in .env")
        sys.exit(1)
    if not GEMINI_API_KEY:
        print("ERROR: Please set GEMINI_API_KEY in .env")
        sys.exit(1)

    print("Starting Garmin Coach Telegram Bot...")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("briefing", briefing_command))
    app.add_handler(CommandHandler("week", weekly_command))
    app.add_handler(CommandHandler("semana", weekly_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    webhook_url = os.getenv("WEBHOOK_URL")
    port = int(os.getenv("PORT", 8080))

    if webhook_url:
        print(f"🤖 Starting Telegram Bot in WEBHOOK mode on port {port}...")
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=webhook_url.rstrip("/") + "/" + TELEGRAM_BOT_TOKEN,
            url_path=TELEGRAM_BOT_TOKEN
        )
    else:
        print("🤖 Starting Telegram Bot in POLLING mode...")
        # Start healthcheck server so Cloud Run / Render TCP port checks pass during polling
        threading.Thread(target=start_health_server, daemon=True).start()
        app.run_polling()


if __name__ == "__main__":
    main()
