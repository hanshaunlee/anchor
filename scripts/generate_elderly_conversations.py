#!/usr/bin/env python3
"""
Generate tens of thousands of elderly conversation events in the exact schema
the edge AI uploads (EventPacket → POST /ingest/events).

Outputs:
- sessions.json: sessions to insert first (household_id, device_id, started_at, ended_at, mode, consent_state)
- event_packets.ndjson: one event per line (session_id, device_id, ts, seq, event_type, payload_version, payload)
- summaries.json: session_id -> { summary_text, summary_json } for viable session summaries
- ingest_payloads.jsonl: optional, same events in request shape { "events": [ ... ] } per batch

Design:
- Conversations span every day of the past year (365 days); multiple sessions per day.
- Same relationships/calls repeat: fixed pool of persons, phones, orgs reused across sessions.
- Diversity: normal (weather, reminders, family, doctor, shopping) and scam-like (IRS, Medicare, tech support, gift cards).
- Each session's summary_text is a viable summary of the final_asr/intent events in that session.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

# Repo root
_repo_root = Path(__file__).resolve().parent.parent

# ---------- Entity pools (reused for repetition in relationships/calls) ----------
PERSONS = [
    "Sarah", "John", "Mom", "Dad", "Grandkid", "Michael", "Linda", "Robert", "Patricia",
    "David", "Jennifer", "Neighbor", "Doctor Smith", "Nurse Amy", "Grandkid Emma",
]
PHONES = [
    "+15551234567", "+15559876543", "+18005551234", "+15551112222", "555-1234",
    "800-633-4227", "+15553334444", "+15556667777", "555-9876",
]
ORGS = [
    "IRS", "Medicare", "Social Security", "Amazon", "Apple", "Bank of America",
    "Wells Fargo", "Tech Support", "CVS", "Walmart", "Doctor's office",
]
MERCHANTS = ["Amazon", "Walmart", "CVS", "Target", "Best Buy", "Home Depot", "Gift Card Co"]
TOPICS = [
    "Medicare", "Social Security", "tax refund", "gift cards", "account suspended",
    "verify identity", "urgent", "prize", "computer virus", "warranty",
]

# ---------- Conversation templates ----------
# Each: (list of (speaker, text, intent_name?), summary_text)
# Speaker: "elder" | "agent" | "unknown". Intent optional; if None we still emit final_asr only.


def _hash(s: str) -> str:
    return hashlib.sha256(s.strip().lower().encode()).hexdigest()[:16]


def _person(): return random.choice(PERSONS)
def _phone(): return random.choice(PHONES)
def _org(): return random.choice(ORGS)


# Normal, non-scam conversations (majority)
NORMAL_TEMPLATES = [
    (
        [
            ("elder", "What's the weather today?", "weather"),
            ("agent", "It's partly cloudy, high of 72.", None),
            ("elder", "Remind me to take my medicine at 5.", "reminder"),
        ],
        "User asked for the weather and set a medication reminder for 5 PM.",
    ),
    (
        [
            ("elder", "Call Sarah for me.", "call_log"),
            ("agent", "Calling Sarah now.", None),
        ],
        "User requested a call to Sarah.",
    ),
    (
        [
            ("elder", "When is my doctor appointment?", "appointment"),
            ("agent", "Your appointment with Doctor Smith is Thursday at 10.", None),
        ],
        "User asked about their doctor appointment; agent confirmed Thursday at 10.",
    ),
    (
        [
            ("elder", "Add milk and bread to the shopping list.", "shopping_list"),
            ("agent", "Added milk and bread.", None),
        ],
        "User added milk and bread to the shopping list.",
    ),
    (
        [
            ("elder", "Play some music.", "play_music"),
            ("agent", "Playing your favorites.", None),
        ],
        "User asked for music; agent started playback.",
    ),
    (
        [
            ("elder", "What time is it?", "time"),
            ("agent", "It's 3:45 PM.", None),
        ],
        "User asked the time; agent responded 3:45 PM.",
    ),
    (
        [
            ("elder", "Remind me to take my pill at 8.", "reminder"),
            ("agent", "Reminder set for 8 PM.", None),
        ],
        "User set a pill reminder for 8 PM.",
    ),
    (
        [
            ("elder", "Did John call?", "call_log"),
            ("agent", "No new calls from John today.", None),
        ],
        "User asked if John had called; agent said no new calls from John.",
    ),
    (
        [
            ("elder", "Set a reminder for my refill next Monday.", "reminder"),
            ("agent", "Reminder set for next Monday for your refill.", None),
        ],
        "User set a reminder for prescription refill next Monday.",
    ),
    (
        [
            ("elder", "What's on my calendar tomorrow?", "calendar"),
            ("agent", "You have a check-in with Nurse Amy at 2 PM.", None),
        ],
        "User asked about tomorrow's calendar; agent listed check-in with Nurse Amy.",
    ),
    (
        [
            ("elder", "Call my daughter.", "call_log"),
            ("agent", "Calling Jennifer.", None),
        ],
        "User asked to call their daughter Jennifer.",
    ),
    (
        [
            ("elder", "Is it going to rain later?", "weather"),
            ("agent", "There's a 30 percent chance of rain after 6.", None),
        ],
        "User asked about rain; agent gave a 30% chance after 6 PM.",
    ),
    (
        [
            ("elder", "Add eggs to the list.", "shopping_list"),
            ("agent", "Done. Eggs added.", None),
        ],
        "User added eggs to the shopping list.",
    ),
    (
            [
                ("elder", "When do I take my blood pressure pill?", "medication"),
                ("agent", "You take it at 8 AM and 8 PM.", None),
            ],
        "User asked when to take blood pressure medication; agent said 8 AM and 8 PM.",
    ),
    (
        [
            ("elder", "I'm going to sit in the garden.", "general"),
            ("agent", "Enjoy. I'll be here if you need anything.", None),
        ],
        "User mentioned going to the garden; agent acknowledged.",
    ),
    (
        [
            ("elder", "Remind me when Grandkid Emma calls.", "reminder"),
            ("agent", "I'll remind you when Emma calls.", None),
        ],
        "User asked for a reminder when Grandkid Emma calls.",
    ),
    (
        [
            ("elder", "What did the neighbor say?", "general"),
            ("agent", "I don't have a message from the neighbor. Would you like to call them?", None),
        ],
        "User asked about the neighbor; agent had no message and offered to call.",
    ),
    (
        [
            ("elder", "Turn off the lights in the living room.", "smart_home"),
            ("agent", "Living room lights are off.", None),
        ],
        "User asked to turn off living room lights; agent confirmed.",
    ),
    (
        [
            ("elder", "Do I have any packages coming?", "general"),
            ("agent", "You have one delivery from Amazon expected tomorrow.", None),
        ],
        "User asked about packages; agent said one Amazon delivery tomorrow.",
    ),
    (
        [
            ("elder", "Read my messages.", "messages"),
            ("agent", "You have one message from Linda: she'll visit Saturday.", None),
        ],
        "User asked for messages; agent read one from Linda about visiting Saturday.",
    ),
    (
        [
            ("elder", "Call the doctor's office.", "call_log"),
            ("agent", "Calling your doctor's office.", None),
        ],
        "User requested a call to the doctor's office.",
    ),
    (
        [
            ("elder", "What day is my Social Security check?", "general"),
            ("agent", "Usually the second Wednesday. I can check the exact date if you want.", None),
        ],
        "User asked when Social Security check comes; agent said second Wednesday.",
    ),
    (
        [
            ("elder", "Remind me to water the plants.", "reminder"),
            ("agent", "Reminder set for watering plants. When would you like it?", None),
            ("elder", "Tomorrow morning.", "reminder"),
        ],
        "User set a reminder to water the plants for tomorrow morning.",
    ),
    (
        [
            ("elder", "How do I refill my prescription?", "medication"),
            ("agent", "You can call CVS or use their app. I can call CVS for you.", None),
        ],
        "User asked about prescription refill; agent suggested calling CVS or using the app.",
    ),
    (
        [
            ("elder", "Turn up the volume.", "smart_home"),
            ("agent", "Volume increased.", None),
        ],
        "User asked to turn up the volume; agent complied.",
    ),
    (
        [
            ("elder", "Good morning.", "greeting"),
            ("agent", "Good morning. How can I help?", None),
        ],
        "Brief morning greeting and offer to help.",
    ),
    (
        [
            ("elder", "Thank you, that's all.", "goodbye"),
            ("agent", "You're welcome. Take care.", None),
        ],
        "User thanked the agent and ended the conversation.",
    ),
    (
        [
            ("elder", "What's for dinner?", "general"),
            ("agent", "I don't have your meal plan. Would you like to order something or check the fridge?", None),
        ],
        "User asked about dinner; agent asked if they want to order or check the fridge.",
    ),
    (
        [
            ("elder", "Call Robert when you get a chance.", "call_log"),
            ("agent", "I'll place the call to Robert now.", None),
        ],
        "User asked to call Robert; agent initiated the call.",
    ),
    (
        [
            ("elder", "Set a reminder for my bridge game Tuesday.", "reminder"),
            ("agent", "Reminder set for Tuesday for your bridge game.", None),
        ],
        "User set a reminder for bridge game on Tuesday.",
    ),
    (
        [
            ("elder", "Did Medicare send anything?", "general"),
            ("agent", "I don't see new mail from Medicare. Want me to remind you to check the mailbox?", None),
        ],
        "User asked about mail from Medicare; agent had no new mail and offered to remind to check mailbox.",
    ),
    (
        [
            ("elder", "Call Mom.", "call_log"),
            ("agent", "Calling Mom.", None),
        ],
        "User asked to call Mom.",
    ),
    (
        [
            ("elder", "Call Dad.", "call_log"),
            ("agent", "Calling Dad.", None),
        ],
        "User asked to call Dad.",
    ),
    (
        [
            ("elder", "I need to call the pharmacy.", "call_log"),
            ("agent", "Calling CVS pharmacy.", None),
        ],
        "User requested a call to the pharmacy; agent called CVS.",
    ),
    (
        [
            ("elder", "What's my next reminder?", "reminder"),
            ("agent", "Next reminder is at 5 PM for your medicine.", None),
        ],
        "User asked for next reminder; agent said 5 PM medicine reminder.",
    ),
    (
        [
            ("elder", "Cancel my 3 PM reminder.", "reminder"),
            ("agent", "Reminder at 3 PM has been cancelled.", None),
        ],
        "User cancelled the 3 PM reminder.",
    ),
    (
        [
            ("elder", "Is Patricia coming over this week?", "general"),
            ("agent", "I don't see Patricia on your calendar. Want me to remind you to call her?", None),
        ],
        "User asked if Patricia is visiting; agent had no calendar entry and offered to remind to call.",
    ),
    (
        [
            ("elder", "Remind me to take my vitamins with breakfast.", "reminder"),
            ("agent", "Reminder set for with breakfast.", None),
        ],
        "User set a reminder to take vitamins with breakfast.",
    ),
    (
        [
            ("elder", "What's the number for the doctor?", "call_log"),
            ("agent", "Doctor Smith's office is 555-1234. Should I call?", None),
        ],
        "User asked for doctor's number; agent gave 555-1234 and offered to call.",
    ),
    (
        [
            ("elder", "I'm feeling a bit dizzy.", "general"),
            ("agent", "I'm sorry to hear that. Do you want me to call someone or set a reminder to check in with the doctor?", None),
        ],
        "User reported feeling dizzy; agent offered to call someone or set a doctor check-in reminder.",
    ),
    (
        [
            ("elder", "Tell David I said hello when he calls.", "reminder"),
            ("agent", "I'll remind you to tell David hello when he calls.", None),
        ],
        "User asked to be reminded to tell David hello when he calls.",
    ),
    (
        [
            ("elder", "Did the Neighbor bring the paper?", "general"),
            ("agent", "I don't have that information. You could call the Neighbor to check.", None),
        ],
        "User asked if the Neighbor brought the paper; agent suggested calling to check.",
    ),
    (
        [
            ("elder", "Set an alarm for 7 AM.", "reminder"),
            ("agent", "Alarm set for 7 AM.", None),
        ],
        "User set an alarm for 7 AM.",
    ),
    (
        [
            ("elder", "What did Jennifer say in her message?", "messages"),
            ("agent", "Jennifer said she'll pick you up for the appointment on Friday.", None),
        ],
        "User asked about Jennifer's message; agent relayed pickup for Friday appointment.",
    ),
    (
        [
            ("elder", "I want to order more tissues from Amazon.", "general"),
            ("agent", "I can open the Amazon app or website. Do you want to reorder from your last order?", None),
        ],
        "User wanted to order tissues from Amazon; agent offered to open app or reorder.",
    ),
    (
        [
            ("elder", "When is my bridge club?", "calendar"),
            ("agent", "Your bridge club is every Thursday at 2 PM.", None),
        ],
        "User asked about bridge club; agent said Thursdays at 2 PM.",
    ),
    (
        [
            ("elder", "Call Grandkid.", "call_log"),
            ("agent", "Calling Grandkid.", None),
        ],
        "User asked to call Grandkid.",
    ),
    (
        [
            ("elder", "Is it supposed to snow?", "weather"),
            ("agent", "No snow in the forecast for the next few days.", None),
        ],
        "User asked about snow; agent said no snow in forecast.",
    ),
]

# Scam / high-risk themed (minority but present)
SCAM_TEMPLATES = [
    (
        [
            ("elder", "Someone from Medicare called saying my account is suspended.", "sensitive_request"),
            ("agent", "That could be a scam. Medicare doesn't call to suspend accounts. Don't give personal info.", None),
        ],
        "User reported a call from someone claiming to be Medicare about account suspension; agent warned of possible scam.",
    ),
    (
        [
            ("elder", "A man said he's from the IRS and I owe money.", "sensitive_request"),
            ("agent", "The IRS usually writes first. Don't pay over the phone. Want me to note this for your caregiver?", None),
        ],
        "User reported a call from someone claiming to be the IRS demanding payment; agent advised caution and offered to notify caregiver.",
    ),
    (
        [
            ("elder", "They wanted me to buy gift cards to fix the computer.", "transfer_money"),
            ("agent", "That's a common scam. No legitimate company asks for gift cards. Don't send anything.", None),
        ],
        "User reported being asked to buy gift cards to fix a computer; agent identified it as a common scam.",
    ),
    (
        [
            ("elder", "Someone called about my car warranty expiring.", "verify_identity"),
            ("agent", "Warranty calls are often scams. Did they ask for payment or your Social Security number?", None),
        ],
        "User reported a call about car warranty expiring; agent cautioned about warranty scams and asked if they asked for payment or SSN.",
    ),
    (
        [
            ("elder", "They said I won a prize and need to pay a fee.", "transfer_money"),
            ("agent", "Real prizes don't require you to pay. This sounds like a scam. Don't send money.", None),
        ],
        "User reported a prize offer requiring a fee; agent said real prizes don't require payment and advised not to send money.",
    ),
    (
        [
            ("elder", "Tech support said my computer has a virus.", "sensitive_request"),
            ("agent", "Legitimate tech support won't call you first. Don't let them remote in or pay. I can help you check.", None),
        ],
        "User reported unsolicited tech support call about a virus; agent warned not to allow remote access or pay.",
    ),
    (
        [
            ("elder", "They asked for my Social Security number to verify.", "share_ssn"),
            ("agent", "Never give your Social Security number over the phone to someone who called you. This is likely a scam.", None),
        ],
        "User reported being asked for SSN to verify; agent advised never to give SSN to unsolicited callers.",
    ),
    (
        [
            ("elder", "Someone from the bank said my account was locked.", "account_takeover_risk"),
            ("agent", "Call the number on your card or statement, not one they give you. Could be a phishing scam.", None),
        ],
        "User reported a call about locked bank account; agent advised calling the number on the card, not one provided by caller.",
    ),
    (
        [
            ("elder", "They said I had to act now or I'd be arrested.", "urgency_language"),
            ("agent", "That's a scare tactic. Government and banks don't threaten arrest by phone. Don't comply.", None),
        ],
        "User reported being told to act now or face arrest; agent identified it as a scare tactic and advised not to comply.",
    ),
    (
        [
            ("elder", "A woman said she's from Social Security and needed to confirm my identity.", "verify_identity"),
            ("agent", "Social Security may write or call, but they won't ask for sensitive details on an unsolicited call. Be careful.", None),
        ],
        "User reported a call from someone claiming to be Social Security asking to confirm identity; agent advised caution.",
    ),
    (
        [
            ("elder", "They wanted me to wire money to get my grandson out of jail.", "transfer_money"),
            ("agent", "That's a common grandparent scam. Call your grandson on a number you know before doing anything.", None),
        ],
        "User reported a request to wire money for grandson in jail; agent warned of grandparent scam and advised calling grandson on known number.",
    ),
    (
        [
            ("elder", "I got a text saying my account will be closed.", "account_takeover_risk"),
            ("agent", "Don't click links in texts. Log in to your account by typing the URL or using the app.", None),
        ],
        "User received a text about account closure; agent advised not to click links and to log in via official app or URL.",
    ),
    (
        [
            ("elder", "They said there's a problem with my Medicare card and I need to give my number.", "share_ssn"),
            ("agent", "Medicare doesn't call to ask for your number like that. Hang up and call Medicare using the number on your card.", None),
        ],
        "User reported a call about Medicare card problem asking for number; agent said Medicare doesn't call that way and to call the number on the card.",
    ),
]

# Mixed / neutral (could go either way)
MIXED_TEMPLATES = [
    (
        [
            ("elder", "I had a long call with someone about insurance.", "call_log"),
            ("agent", "Do you remember who it was or what they wanted?", None),
            ("elder", "I'm not sure. They had a lot of questions.", "general"),
        ],
        "User mentioned a long call about insurance and was unsure of the caller or details.",
    ),
    (
        [
            ("elder", "Someone's been calling about a bill.", "call_log"),
            ("agent", "Do you know which company? We can look up the real number and you can call back.", None),
        ],
        "User mentioned repeated calls about a bill; agent offered to look up the real company number.",
    ),
    (
        [
            ("elder", "I got a letter from the IRS. Should I be worried?", "general"),
            ("agent", "Real IRS letters have a notice number. Want me to remind you to have someone you trust look at it?", None),
        ],
        "User received a letter from the IRS and asked if they should worry; agent suggested verifying with a trusted person.",
    ),
]

ALL_TEMPLATES = NORMAL_TEMPLATES + SCAM_TEMPLATES + MIXED_TEMPLATES


def pick_template() -> tuple[list[tuple[str, str, str | None]], str]:
    # Weight normal heavily, then scam, then mixed
    r = random.random()
    if r < 0.72:
        t = random.choice(NORMAL_TEMPLATES)
    elif r < 0.92:
        t = random.choice(SCAM_TEMPLATES)
    else:
        t = random.choice(MIXED_TEMPLATES)
    return t[0], t[1]


def generate_session_and_events(
    session_id: str,
    device_id: str,
    day_offset: int,
    template_utterances: list[tuple[str, str, str | None]],
    summary_text: str,
) -> tuple[dict, list[dict], dict]:
    """Generate one session record, event packets (edge schema), and summary.
    day_offset: 0 = today, 1 = yesterday, ... 364 = one year ago.
    """
    base_ts = datetime.now(timezone.utc) - timedelta(days=day_offset)
    # Spread session within the day (6 AM - 10 PM)
    hour = random.randint(6, 22)
    minute = random.randint(0, 59)
    start_dt = base_ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
    duration_min = max(1, len(template_utterances) * 2 + random.randint(2, 10))
    end_dt = start_dt + timedelta(minutes=duration_min)

    session = {
        "id": session_id,
        "device_id": device_id,
        "started_at": start_dt.isoformat(),
        "ended_at": end_dt.isoformat(),
        "mode": random.choice(["offline", "online"]),
        "consent_state": {},
    }

    events: list[dict] = []
    step_sec = max(5, duration_min * 60 // max(1, len(template_utterances) * 2 + 2))
    t = start_dt
    seq = 0

    # wake
    events.append({
        "session_id": session_id,
        "device_id": device_id,
        "ts": t.isoformat(),
        "seq": seq,
        "event_type": "wake",
        "payload_version": 1,
        "payload": {},
    })
    seq += 1
    t = t + timedelta(seconds=random.randint(2, 8))

    for speaker, text, intent_name in template_utterances:
        # final_asr
        events.append({
            "session_id": session_id,
            "device_id": device_id,
            "ts": t.isoformat(),
            "seq": seq,
            "event_type": "final_asr",
            "payload_version": 1,
            "payload": {
                "text": text,
                "text_hash": _hash(text),
                "confidence": round(random.uniform(0.82, 0.98), 2),
                "speaker": {"role": speaker},
            },
        })
        seq += 1
        t = t + timedelta(seconds=random.randint(1, 4))

        # intent (optional)
        if intent_name:
            events.append({
                "session_id": session_id,
                "device_id": device_id,
                "ts": t.isoformat(),
                "seq": seq,
                "event_type": "intent",
                "payload_version": 1,
                "payload": {
                    "name": intent_name,
                    "slots": {},
                    "confidence": round(random.uniform(0.75, 0.95), 2),
                },
            })
            seq += 1
            t = t + timedelta(seconds=random.randint(1, 3))

    # optional device_state sometimes
    if random.random() < 0.2:
        events.append({
            "session_id": session_id,
            "device_id": device_id,
            "ts": t.isoformat(),
            "seq": seq,
            "event_type": "device_state",
            "payload_version": 1,
            "payload": {"online": True, "battery": round(random.uniform(0.5, 1.0), 2)},
        })
        seq += 1

    summary = {
        "session_id": session_id,
        "summary_text": summary_text,
        "summary_json": {},
    }
    return session, events, summary


def generate_elderly_data_in_memory(
    household_id: str,
    device_ids: list[str],
    num_sessions: int = 5000,
    days: int = 365,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate sessions, events, and summaries in memory for a given household and devices.
    Returns (sessions, events, summaries) ready for DB insert (no file I/O).
    Used by seed_supabase_data.py --with-elderly-conversations.
    """
    random.seed(seed)
    session_day_offsets: list[int] = []
    for d in range(days):
        session_day_offsets.append(d)
    for _ in range(num_sessions - days):
        session_day_offsets.append(random.randint(0, days - 1))
    random.shuffle(session_day_offsets)

    sessions_out: list[dict] = []
    all_events: list[dict] = []
    summaries_out: list[dict] = []
    for i, day_offset in enumerate(session_day_offsets):
        session_id = str(uuid4())
        device_id = random.choice(device_ids)
        template_utterances, summary_text = pick_template()
        session, events, summary = generate_session_and_events(
            session_id, device_id, day_offset, template_utterances, summary_text
        )
        session["household_id"] = household_id
        sessions_out.append(session)
        all_events.extend(events)
        summaries_out.append(summary)
    return sessions_out, all_events, summaries_out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate elderly conversation events (edge upload schema).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: repo root / data/elderly_conversations)")
    parser.add_argument("--household-id", type=str, default=None, help="household_id for session records (optional; can be filled on insert)")
    parser.add_argument("--num-sessions", type=int, default=35_000, help="Total number of conversations (sessions)")
    parser.add_argument("--days", type=int, default=365, help="Span of days (every day will have at least one session)")
    parser.add_argument("--devices", type=int, default=3, help="Number of device UUIDs to rotate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=500, help="Events per ingest batch in ingest_payloads.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else _repo_root / "data" / "elderly_conversations"
    out_dir.mkdir(parents=True, exist_ok=True)

    household_id = args.household_id or str(uuid4())
    device_ids = [str(uuid4()) for _ in range(args.devices)]

    # Spread sessions across days: guarantee at least one session per day, then fill the rest
    total_sessions = args.num_sessions
    days = args.days
    session_day_offsets: list[int] = []
    # One session per day so every day is covered
    for d in range(days):
        session_day_offsets.append(d)
    # Remaining sessions assigned randomly (allows repetition of same day)
    for _ in range(total_sessions - days):
        session_day_offsets.append(random.randint(0, days - 1))
    random.shuffle(session_day_offsets)

    sessions_out: list[dict] = []
    all_events: list[dict] = []
    summaries_out: list[dict] = []
    session_id_to_summary: dict[str, dict] = {}

    print("Generating", total_sessions, "sessions over", days, "days...")
    for i, day_offset in enumerate(session_day_offsets):
        session_id = str(uuid4())
        device_id = random.choice(device_ids)
        template_utterances, summary_text = pick_template()
        session, events, summary = generate_session_and_events(
            session_id, device_id, day_offset, template_utterances, summary_text
        )
        session["household_id"] = household_id
        sessions_out.append(session)
        all_events.extend(events)
        summaries_out.append(summary)
        session_id_to_summary[session_id] = summary
        if (i + 1) % 5000 == 0:
            print("  ", i + 1, "sessions,", len(all_events), "events")

    # Write sessions (for DB insert; id included so you can use it)
    sessions_path = out_dir / "sessions.json"
    with open(sessions_path, "w") as f:
        json.dump({
            "household_id": household_id,
            "device_ids": device_ids,
            "sessions": sessions_out,
        }, f, indent=2)
    print("Wrote", sessions_path, "(", len(sessions_out), "sessions )")

    # Write event packets (one per line) - exact edge upload schema
    events_ndjson_path = out_dir / "event_packets.ndjson"
    with open(events_ndjson_path, "w") as f:
        for e in all_events:
            f.write(json.dumps(e) + "\n")
    print("Wrote", events_ndjson_path, "(", len(all_events), "events )")

    # Summaries (session-scoped; viable for each session's events)
    summaries_path = out_dir / "summaries.json"
    with open(summaries_path, "w") as f:
        json.dump({
            "household_id": household_id,
            "summaries": summaries_out,
        }, f, indent=2)
    print("Wrote", summaries_path, "(", len(summaries_out), "summaries )")

    # Optional: ingest payloads (batches for POST /ingest/events)
    ingest_path = out_dir / "ingest_payloads.jsonl"
    with open(ingest_path, "w") as f:
        for i in range(0, len(all_events), args.batch_size):
            batch = all_events[i : i + args.batch_size]
            f.write(json.dumps({"events": batch}) + "\n")
    print("Wrote", ingest_path, "(", (len(all_events) + args.batch_size - 1) // args.batch_size, "batches )")

    # Schema reference (what edge uploads)
    schema_ref = {
        "description": "Event packet schema (edge → POST /ingest/events)",
        "per_event": {
            "session_id": "uuid",
            "device_id": "uuid",
            "ts": "ISO8601 timestamptz",
            "seq": "int (monotonic per session)",
            "event_type": "wake | partial_asr | final_asr | intent | tool_call | tool_result | tts | error | device_state | watchlist_hit | transaction_detected | payee_added | bank_alert_received",
            "payload_version": "int (1)",
            "payload": "object (see event_packet_spec.md)",
        },
        "final_asr_payload": {"text": "optional", "text_hash": "optional", "lang": "optional", "confidence": "0-1", "speaker": {"role": "elder|agent|unknown"}},
        "intent_payload": {"name": "string", "slots": "object", "confidence": "0-1"},
    }
    schema_path = out_dir / "schema_reference.json"
    with open(schema_path, "w") as f:
        json.dump(schema_ref, f, indent=2)
    print("Wrote", schema_path)

    # Day coverage check
    days_with_sessions = len(set(session_day_offsets))
    print("Days with at least one session:", days_with_sessions, "/", days)
    if days_with_sessions < days:
        print("  (Some days have no session due to random spread; increase --num-sessions to guarantee every day.)")
    else:
        print("  (Every day has at least one conversation.)")


if __name__ == "__main__":
    main()
