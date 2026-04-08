"""
Email Triage Environment — Pydantic Models.

Defines the typed Action, Observation, State, and supporting models
for the Email Triage OpenEnv environment.

The Action and Observation extend the OpenEnv base types so that
the HTTP server can correctly serialize/deserialize them.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ── Supporting Models ──────────────────────────────────────────────

class EmailItem(BaseModel):
    """A single email in the inbox."""
    email_id: str = Field(..., description="Unique identifier for this email")
    sender: str = Field(..., description="Sender email address")
    sender_name: str = Field("", description="Display name of the sender")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    timestamp: str = Field(..., description="ISO-8601 timestamp when email was received")
    has_attachment: bool = Field(False, description="Whether the email has attachments")
    reply_to: Optional[str] = Field(None, description="Email ID this is a reply to")
    thread_length: int = Field(1, description="Number of messages in this thread")
    is_read: bool = Field(False, description="Whether the email has been read")


class EmailGroundTruth(BaseModel):
    """Hidden ground truth labels for scoring."""
    email_id: str
    category: str  # billing, technical, sales, hr, spam, phishing
    priority: str  # critical, high, medium, low
    department: str  # billing_dept, engineering, sales_team, hr_dept, security, general
    requires_response: bool
    key_response_points: list[str] = Field(default_factory=list)
    is_spam: bool = False
    is_phishing: bool = False


# ── Core OpenEnv Models ───────────────────────────────────────────

class TriageAction(Action):
    """Action the agent takes on a single email.
    
    Extends the OpenEnv base Action so the HTTP server can 
    properly deserialize incoming JSON payloads.
    """
    email_id: str = Field(..., description="ID of the email to act on")
    action_type: Literal[
        "categorize",
        "prioritize",
        "route",
        "draft_response",
        "mark_spam",
        "skip"
    ] = Field(..., description="Type of action to perform")
    category: Optional[str] = Field(
        None,
        description="Category label: billing, technical, sales, hr, general"
    )
    priority: Optional[Literal["critical", "high", "medium", "low"]] = Field(
        None,
        description="Priority level for this email"
    )
    route_to: Optional[str] = Field(
        None,
        description="Department: billing_dept, engineering, sales_team, hr_dept, security, general"
    )
    draft_response: Optional[str] = Field(
        None,
        description="Draft reply text for emails that need a response"
    )


class TriageObservation(Observation):
    """What the agent observes after each step.
    
    Extends the OpenEnv base Observation which provides `done`, `reward`, `metadata`.
    Custom fields are serialized into the observation dict by the HTTP server.
    """
    inbox: list[EmailItem] = Field(
        default_factory=list,
        description="Current unprocessed emails in the inbox"
    )
    processed_count: int = Field(0, description="Number of emails processed so far")
    total_count: int = Field(0, description="Total emails in this episode")
    current_score: float = Field(0.0, description="Running average score (0.0-1.0)")
    step_reward: float = Field(0.0, description="Reward from the last action")
    feedback: str = Field("", description="Natural language feedback on the last action")
    task_id: str = Field("", description="Current task identifier")
    task_description: str = Field("", description="Human-readable task objective")


class TriageState(State):
    """Internal episode state metadata.
    
    Extends the OpenEnv base State which provides `episode_id`, `step_count`.
    """
    task_id: str = Field("", description="Current task identifier")
    max_steps: int = Field(50, description="Maximum allowed steps")
    emails_remaining: int = Field(0, description="Emails not yet processed")
    emails_processed: int = Field(0, description="Emails processed so far")
    total_emails: int = Field(0, description="Total emails in inbox")
    score: float = Field(0.0, description="Current cumulative score (0.0-1.0)")
    rewards_history: list[float] = Field(default_factory=list, description="Per-step rewards")
