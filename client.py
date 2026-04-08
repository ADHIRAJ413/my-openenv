"""
Email Triage Environment Client.

Provides a typed client for connecting to an Email Triage Environment server.
Extends EnvClient with proper serialization for TriageAction → TriageObservation.

Example:
    >>> import asyncio
    >>> from client import EmailTriageEnv
    >>> from models import TriageAction, TriageObservation
    >>>
    >>> async def main():
    ...     async with EmailTriageEnv(base_url="http://localhost:8000") as env:
    ...         result = await env.reset(task_id="easy_categorize")
    ...         print(result.observation.feedback)
    ...
    ...         result = await env.step(TriageAction(
    ...             email_id="email-0001",
    ...             action_type="categorize",
    ...             category="billing"
    ...         ))
    ...         print(f"Reward: {result.reward}, Done: {result.done}")
    >>>
    >>> asyncio.run(main())
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import TriageAction, TriageObservation, TriageState, EmailItem


class EmailTriageEnv(EnvClient[TriageAction, TriageObservation, TriageState]):
    """
    Typed client for the Email Triage Environment.

    Inherits from EnvClient with:
    - TriageAction as the action type
    - TriageObservation as the observation type
    - TriageState as the state type

    Provides:
    - reset(**kwargs) → StepResult[TriageObservation]
    - step(action) → StepResult[TriageObservation]
    - state() → TriageState
    - from_docker_image(image) → connected client
    - from_env(repo_id) → connected client from HF Space
    - close() → cleanup
    """

    def _step_payload(self, action: TriageAction) -> Dict[str, Any]:
        """Serialize a TriageAction to JSON payload for the server."""
        if hasattr(action, "model_dump"):
            return action.model_dump()
        if isinstance(action, dict):
            return action
        return vars(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TriageObservation]:
        """Parse server response into StepResult[TriageObservation]."""
        obs_data = payload.get("observation", {})

        # Rebuild EmailItem list from raw dicts
        inbox_raw = obs_data.get("inbox", [])
        inbox = [EmailItem(**e) if isinstance(e, dict) else e for e in inbox_raw]

        observation = TriageObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            inbox=inbox,
            processed_count=obs_data.get("processed_count", 0),
            total_count=obs_data.get("total_count", 0),
            current_score=obs_data.get("current_score", 0.0),
            step_reward=obs_data.get("step_reward", 0.0),
            feedback=obs_data.get("feedback", ""),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TriageState:
        """Parse state response into TriageState."""
        return TriageState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            max_steps=payload.get("max_steps", 50),
            emails_remaining=payload.get("emails_remaining", 0),
            emails_processed=payload.get("emails_processed", 0),
            total_emails=payload.get("total_emails", 0),
            score=payload.get("score", 0.0),
            rewards_history=payload.get("rewards_history", []),
        )
