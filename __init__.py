"""
Email Triage Environment — OpenEnv environment for AI agent training.

Simulates real-world email inbox management where agents must
categorize, prioritize, route, and draft responses to emails.

Usage:
    from email_triage_env import EmailTriageEnv, TriageAction

    async with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="easy_categorize")
        result = await env.step(TriageAction(
            email_id="email-0001",
            action_type="categorize",
            category="billing",
        ))
        print(f"Reward: {result.reward}, Done: {result.done}")
"""

from models import TriageAction, TriageObservation, TriageState
from client import EmailTriageEnv

__all__ = ["EmailTriageEnv", "TriageAction", "TriageObservation", "TriageState"]
