"""
WhatsApp School Calendar Agent — Built with CrewAI + Gemini
------------------------------------------------------------
Agents:
  1. Calendar Fetcher  — reads & parses the iCal calendar
  2. Summarizer        — filters upcoming items, builds bullet-point summary
  3. WhatsApp Sender   — delivers the message via Twilio

Run:  python crew.py
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from dotenv import load_dotenv
import datetime, os, requests
from icalendar import Calendar
from twilio.rest import Client

load_dotenv()

# ── Gemini LLM (free tier — get key at aistudio.google.com) ──────────────────
# NEW CODE
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash", # Use 2.0 or 1.5-flash-latest
    api_key=os.getenv("GEMINI_API_KEY"),
)

LOOKAHEAD_DAYS     = int(os.getenv("LOOKAHEAD_DAYS", 1))
CALENDAR_URL       = os.getenv("CALENDAR_URL", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM        = os.getenv("TWILIO_WHATSAPP_FROM")
TWILIO_TO          = os.getenv("YOUR_WHATSAPP_NUMBER")


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool("Fetch Calendar Events")
def fetch_calendar_events(source: str = "") -> str:
    """
    Fetches and parses a school calendar from an iCal URL or local .ics file.
    Returns upcoming events (within LOOKAHEAD_DAYS) as a formatted string.
    Pass the URL or file path as source, or leave blank to use env vars.
    """
    try:
        src = source.strip() or CALENDAR_URL 
        if not src:
            return "ERROR: No calendar source provided. Set CALENDAR_URL or CALENDAR_FILE in .env"

        if src.startswith("http"):
            resp = requests.get(src, timeout=15)
            resp.raise_for_status()
            cal = Calendar.from_ical(resp.content)
        else:
            with open(src, "rb") as f:
                cal = Calendar.from_ical(f.read())

        today  = datetime.date.today()
        cutoff = today + datetime.timedelta(days=LOOKAHEAD_DAYS)
        events = []

        for component in cal.walk():
            if component.name != "VEVENT":
                continue
            dtstart = component.get("DTSTART")
            if not dtstart:
                continue
            event_date = dtstart.dt
            if isinstance(event_date, datetime.datetime):
                event_date = event_date.date()
            if today <= event_date <= cutoff:
                events.append({
                    "title":       str(component.get("SUMMARY", "Untitled")),
                    "date":        event_date.strftime("%A, %B %d"),
                    "description": str(component.get("DESCRIPTION", "")).strip(),
                })

        if not events:
            return f"NO_EVENTS: Nothing due in the next {LOOKAHEAD_DAYS} days."

        events.sort(key=lambda e: e["date"])
        lines = [
            f"- [{e['date']}] {e['title']}"
            + (f" | {e['description']}" if e['description'] and e['description'] != "None" else "")
            for e in events
        ]
        return "\n".join(lines)

    except Exception as ex:
        return f"ERROR fetching calendar: {ex}"


@tool("Send WhatsApp Message")
def send_whatsapp_message(message: str) -> str:
    """
    Sends a WhatsApp message to the student's phone number via Twilio.
    Pass the final formatted message text as message.
    Returns the Twilio message SID on success.
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            from_=TWILIO_FROM,
            to=TWILIO_TO,
            body=message
        )
        return f"SUCCESS: Message sent. SID={msg.sid}"
    except Exception as ex:
        return f"ERROR sending WhatsApp: {ex}"


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

calendar_agent = Agent(
    role="School Calendar Fetcher",
    goal=(
        "Retrieve all upcoming assignments, tests, and events from the school "
        "calendar for the next 7 days and return them as a clean structured list."
    ),
    backstory=(
        "You are a meticulous academic assistant that reads school calendars and "
        "extracts every important due item. You never miss a deadline."
    ),
    tools=[fetch_calendar_events],
    llm=gemini_llm,
    verbose=True,
)

summarizer_agent = Agent(
    role="Academic Summarizer",
    goal=(
        "Transform raw calendar events into a concise, friendly, bullet-point "
        "WhatsApp message that a student will actually read and find useful."
    ),
    backstory=(
        "You are a friendly academic coach who writes punchy, clear summaries. "
        "You group items by date, use emojis wisely, and always end with motivation."
    ),
    llm=gemini_llm,
    verbose=True,
)

whatsapp_agent = Agent(
    role="WhatsApp Delivery Agent",
    goal="Send the final formatted summary message to the student's WhatsApp number.",
    backstory=(
        "You are a reliable messaging agent responsible for delivering academic "
        "reminders to students via WhatsApp without fail."
    ),
    tools=[send_whatsapp_message],
    llm=gemini_llm,
    verbose=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════════════════════

task_fetch = Task(
    description=(
        f"Use the 'Fetch Calendar Events' tool to retrieve all school events "
        f"and assignments due in the next {LOOKAHEAD_DAYS} days. "
        "Return the full raw list exactly as returned by the tool."
    ),
    expected_output=(
        "A newline-separated list of upcoming events with their titles and due dates. "
        "Example:\n- [Monday, April 7] Math Test | Chapter 5 & 6\n- [Wednesday, April 9] Essay Draft"
    ),
    agent=calendar_agent,
)

task_summarize = Task(
    description=(
        "Take the raw event list from the previous task and write a WhatsApp message "
        "for a student. Follow these rules:\n"
        "- Start with a friendly greeting\n"
        "- Use bullet points for each item\n"
        "- Group items under date headers if multiple things are due the same day\n"
        "- Each bullet: subject/task + due date + any key detail\n"
        "- Use emojis: tests, assignments, events, urgent if due tomorrow\n"
        "- End with a short motivational line\n"
        "- Keep it under 300 words\n"
        "- If no events, write a short positive message about having a clear week."
    ),
    expected_output=(
        "A complete, ready-to-send WhatsApp message string with bullet points, "
        "date groupings, emojis, and a closing motivational line."
    ),
    agent=summarizer_agent,
    context=[task_fetch],
)

task_send = Task(
    description=(
        "Take the final formatted WhatsApp message from the summarizer and send it "
        "using the 'Send WhatsApp Message' tool. Report the result."
    ),
    expected_output=(
        "Confirmation that the WhatsApp message was sent successfully, including the Twilio SID."
    ),
    agent=whatsapp_agent,
    context=[task_summarize],
)


# ══════════════════════════════════════════════════════════════════════════════
# CREW
# ══════════════════════════════════════════════════════════════════════════════

crew = Crew(
    agents=[calendar_agent, summarizer_agent, whatsapp_agent],
    tasks=[task_fetch, task_summarize, task_send],
    process=Process.sequential,
    verbose=True,
)


if __name__ == "__main__":
    print("\n Starting School Calendar WhatsApp Agent...\n")
    result = crew.kickoff()
    print("\n Crew finished!")
    print(result)