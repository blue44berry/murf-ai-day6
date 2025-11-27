import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("fraud-agent")
load_dotenv(".env.local")

# -----------------------------
#  Fraud "Database" Structures
# -----------------------------

FRAUD_DB_PATH = Path(__file__).parent.parent / "fraud_cases.json"


@dataclass
class FraudCase:
    userName: str
    securityIdentifier: str
    cardEnding: str
    amount: str
    merchant: str
    location: str
    timestamp: str
    securityQuestion: str
    securityAnswer: str
    status: str = "pending_review"
    outcomeNote: str = ""

    @staticmethod
    def from_dict(d: dict) -> "FraudCase":
        return FraudCase(
            userName=d["userName"],
            securityIdentifier=d["securityIdentifier"],
            cardEnding=d["cardEnding"],
            amount=d["amount"],
            merchant=d["merchant"],
            location=d["location"],
            timestamp=d["timestamp"],
            securityQuestion=d["securityQuestion"],
            securityAnswer=d["securityAnswer"],
            status=d.get("status", "pending_review"),
            outcomeNote=d.get("outcomeNote", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_fraud_db() -> list[FraudCase]:
    if not FRAUD_DB_PATH.exists():
        logger.warning(f"Fraud DB not found at {FRAUD_DB_PATH}, creating empty file.")
        FRAUD_DB_PATH.write_text("[]", encoding="utf-8")
        return []

    with FRAUD_DB_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cases: list[FraudCase] = []
    for entry in raw:
        try:
            cases.append(FraudCase.from_dict(entry))
        except KeyError as e:
            logger.error(f"Skipping malformed fraud entry: {entry} (missing {e})")
    return cases


def save_fraud_db(cases: list[FraudCase]) -> None:
    data = [c.to_dict() for c in cases]
    with FRAUD_DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Fraud DB updated.")


def find_case_by_username(username: str) -> Optional[FraudCase]:
    username_norm = username.strip().lower()
    for case in load_fraud_db():
        if case.userName.strip().lower() == username_norm:
            return case
    return None


def update_case_in_db(updated_case: FraudCase) -> None:
    cases = load_fraud_db()
    replaced = False
    for i, c in enumerate(cases):
        if c.userName.strip().lower() == updated_case.userName.strip().lower():
            cases[i] = updated_case
            replaced = True
            break

    if not replaced:
        cases.append(updated_case)

    save_fraud_db(cases)


# -----------------------------
#  Session State
# -----------------------------

@dataclass
class SessionState:
    current_case: Optional[FraudCase] = None
    is_verified: bool = False
    # track whether outcome already written to DB
    case_closed: bool = False


RunCtx = RunContext[SessionState]


# -----------------------------
#  Fraud Agent Definition
# -----------------------------

class FraudAlertAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a calm, professional fraud detection representative for a fictional bank called "SecureTrust Bank".

Your job:
- Greet the customer by name once you know it.
- Explain clearly that you're calling about a suspicious card transaction.
- Use **only non-sensitive information** to verify identity.
- Never ask for full card number, PIN, OTP, passwords, CVV, or any sensitive credentials.

Call Flow (very important):
1. Introduce yourself as SecureTrust Bank's fraud department.
2. Ask for the customer's first name.
3. Use the tool `load_fraud_case_by_name` **exactly once** after you know their name.
4. If no case is found, politely explain that there is no active fraud alert and end the call.
5. If a case is found:
   - Briefly state that you'll ask one verification question for security.
   - Ask the `securityQuestion` from the case (do not reveal the answer).
6. When the user answers, call `verify_security_answer` with their answer.
   - If verification fails: call `mark_verification_failed_and_close` and politely end the call.
   - If verification passes: calmly read the suspicious transaction details using `get_case_details`.
7. Ask the user if they made this transaction or not.
   - If they say YES / they confirm: call `mark_transaction_safe_and_close`.
   - If they say NO / they deny: call `mark_transaction_fraud_and_close`.
8. At the end, briefly confirm what action was taken (e.g., "We’ve marked this as fraudulent and blocked the card" – this is mock behavior).
9. Keep your answers short, clear, human, and reassuring.
10.You must ONLY ask the verification question provided in the loaded fraud case.
   - Do not invent any other security questions.
   - Never ask about mother’s maiden name, passwords, PINs, card numbers, address, or any personal data not already inside the fraud case JSON.


Behavior rules:
- Always rely on the tools for fraud data (do NOT invent transaction details).
- Do not change the bank name; always say "SecureTrust Bank".
- If you are unsure, ask a simple clarifying question instead of guessing.
""",
        )

    # -------- TOOLS -------- #

    @function_tool()
    async def load_fraud_case_by_name(self, ctx: RunCtx, user_name: str) -> str:
     """
     Load the fraud case for the given customer name from the database.

     Call this after you know the user's name.
     If there is no case found, you should tell the user politely that no active fraud alert exists.
     """
     case = find_case_by_username(user_name)
     if not case:
        ctx.userdata.current_case = None
        return (
            f"No active fraud case found for the name '{user_name}'. "
            "You should gently tell the user that there is no active suspicious transaction on file."
        )

     ctx.userdata.current_case = case
     ctx.userdata.is_verified = False
     ctx.userdata.case_closed = False

    # 🔹 Force the agent to ask ONLY the securityQuestion from the JSON
     await ctx.session.say(
         f"For verification, please answer this question: {case.securityQuestion}"
    )

     return (
        f"Loaded fraud case for user '{case.userName}'. "
        "You have already asked the security question out loud. "
        "Now you must wait for the user's answer and then call 'verify_security_answer' "
        "with what they said. Do NOT invent any other security questions."
    )


    @function_tool()
    async def get_case_details(self, ctx: RunCtx) -> str:
        """
        Return a natural-language description of the suspicious transaction for the current case.

        Use this after the customer has been verified successfully.
        """
        case = ctx.userdata.current_case
        if not case:
            return "There is no active fraud case loaded for this session."

        return (
            f"The suspicious transaction is for {case.amount} at merchant '{case.merchant}', "
            f"located in {case.location}, on {case.timestamp}, "
            f"using the card ending with {case.cardEnding}."
        )

    @function_tool()
    async def verify_security_answer(self, ctx: RunCtx, user_answer: str) -> str:
        """
        Verify the user's answer to the security question for the loaded fraud case.

        Compare the provided answer (case-insensitive) with the stored `securityAnswer`.
        """
        case = ctx.userdata.current_case
        if not case:
            return "No fraud case is currently loaded. You should tell the user something went wrong and end the call."

        correct = case.securityAnswer.strip().lower()
        given = user_answer.strip().lower()

        if given == correct:
            ctx.userdata.is_verified = True
            return (
                "Verification successful. You can now calmly explain the suspicious transaction and ask if they made it."
            )
        else:
            ctx.userdata.is_verified = False
            return (
                "Verification failed. Let the user know you cannot continue for security reasons, "
                "and then you should call 'mark_verification_failed_and_close' to update the case."
            )

    @function_tool()
    async def mark_transaction_safe_and_close(self, ctx: RunCtx) -> str:
        """
        Mark the loaded fraud case as confirmed safe (customer made the transaction).

        Use this ONLY when:
        - The customer has been verified, AND
        - They clearly confirm that they made the transaction.
        """
        case = ctx.userdata.current_case
        if not case:
            return "No fraud case is loaded. Nothing to update."

        case.status = "confirmed_safe"
        case.outcomeNote = "Customer confirmed the suspicious transaction as legitimate."
        update_case_in_db(case)
        ctx.userdata.case_closed = True
        return (
            "The fraud case has been updated to 'confirmed_safe'. "
            "Reassure the user that everything is okay and close the call politely."
        )

    @function_tool()
    async def mark_transaction_fraud_and_close(self, ctx: RunCtx) -> str:
        """
        Mark the loaded fraud case as confirmed fraudulent (customer did NOT make the transaction).

        Use this ONLY when:
        - The customer has been verified, AND
        - They clearly deny making the transaction.
        """
        case = ctx.userdata.current_case
        if not case:
            return "No fraud case is loaded. Nothing to update."

        case.status = "confirmed_fraud"
        case.outcomeNote = (
            "Customer denied the suspicious transaction. "
            "Mock action: card blocked and dispute process initiated."
        )
        update_case_in_db(case)
        ctx.userdata.case_closed = True
        return (
            "The fraud case has been updated to 'confirmed_fraud'. "
            "Tell the user that you have blocked the card and started a dispute process (mock), "
            "and remind them this is a demo."
        )

    @function_tool()
    async def mark_verification_failed_and_close(self, ctx: RunCtx) -> str:
        """
        Mark the loaded fraud case as verification failed.

        Use this when the security check does not pass.
        """
        case = ctx.userdata.current_case
        if not case:
            # nothing to update, but still instruct the agent
            return (
                "There is no fraud case loaded. You should politely say that, "
                "for security reasons, you cannot proceed and end the call."
            )

        case.status = "verification_failed"
        case.outcomeNote = "Verification failed. Caller could not correctly answer the security question."
        update_case_in_db(case)
        ctx.userdata.case_closed = True
        return (
            "The fraud case has been updated to 'verification_failed'. "
            "Politely explain that you cannot proceed without successful verification and end the call."
        )


# -----------------------------
#  Worker / Session Wiring
# -----------------------------

def prewarm(proc: JobProcess):
    # Load VAD once and share across sessions
    proc.userdata["vad"] = silero.VAD.load()


async def fraud_agent_entry(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session_state = SessionState()

    session = AgentSession[SessionState](
        userdata=session_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=FraudAlertAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


async def entrypoint(ctx: JobContext):
    await fraud_agent_entry(ctx)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
