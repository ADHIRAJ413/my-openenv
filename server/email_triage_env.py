"""
Email Triage Environment Implementation.

Implements the OpenEnv Environment interface (step/reset/state)
using the base Environment class from openenv.core.env_server.interfaces.
"""

import sys
import os
from typing import Any, Optional
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment

from models import (
    TriageAction,
    TriageObservation,
    TriageState,
    EmailItem,
    EmailGroundTruth,
)
from server.email_generator import EmailGenerator
from server.tasks import get_task, list_tasks, grade_step, grade_episode, TaskDefinition


class EmailTriageEnvironment(Environment[TriageAction, TriageObservation, TriageState]):
    """
    Email Triage Environment — a real-world task simulation where an AI agent
    must process an inbox of emails by categorizing, prioritizing, routing,
    and drafting responses.

    Implements the standard OpenEnv API:
    - reset(task_id=...) → initial TriageObservation
    - step(action: TriageAction) → TriageObservation with reward
    - state → TriageState  (episode metadata)
    """

    def __init__(self):
        """Initialize the email triage environment."""
        super().__init__()
        self._current_task: Optional[TaskDefinition] = None
        self._emails: list[EmailItem] = []
        self._ground_truths: dict[str, EmailGroundTruth] = {}
        self._processed_ids: set[str] = set()
        self._action_history: list[tuple[TriageAction, EmailGroundTruth]] = []
        self._triage_state = TriageState()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        """
        Reset the environment with a specific task.

        Args:
            seed: Random seed for email generation (default: 42)
            episode_id: Optional episode ID
            **kwargs: Pass task_id="easy_categorize"|"medium_triage"|"hard_full_triage"

        Returns:
            Initial TriageObservation with the full inbox
        """
        task_id = kwargs.get("task_id", "easy_categorize")
        actual_seed = seed if seed is not None else 42

        # Load task
        self._current_task = get_task(task_id)

        # Generate email dataset
        generator = EmailGenerator(seed=actual_seed)
        if self._current_task.difficulty == "easy":
            emails, truths = generator.generate_easy_set()
        elif self._current_task.difficulty == "medium":
            emails, truths = generator.generate_medium_set()
        else:
            emails, truths = generator.generate_hard_set()

        self._emails = emails
        self._ground_truths = {gt.email_id: gt for gt in truths}
        self._processed_ids = set()
        self._action_history = []

        # Initialize state
        eid = episode_id or str(uuid4())
        self._triage_state = TriageState(
            episode_id=eid,
            step_count=0,
            task_id=task_id,
            max_steps=self._current_task.max_steps,
            emails_remaining=len(emails),
            emails_processed=0,
            total_emails=len(emails),
            score=0.0,
            rewards_history=[],
        )

        return TriageObservation(
            done=False,
            reward=0.0,
            inbox=emails,
            processed_count=0,
            total_count=len(emails),
            current_score=0.0,
            step_reward=0.0,
            feedback=f"Inbox loaded with {len(emails)} emails. Task: {self._current_task.name}",
            task_id=task_id,
            task_description=self._current_task.description,
        )

    def step(
        self,
        action: TriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        """
        Execute one triage action on an email.

        Args:
            action: TriageAction specifying email_id, action_type, and details
            timeout_s: Optional timeout (unused)

        Returns:
            TriageObservation with updated inbox, reward, feedback
        """
        if self._current_task is None:
            return TriageObservation(
                done=True,
                reward=0.0,
                feedback="No task loaded. Call reset() first.",
            )

        self._triage_state.step_count += 1

        # Check if episode is already done
        if self._triage_state.step_count > self._triage_state.max_steps:
            return self._finalize_episode("Maximum steps reached.")

        if self._triage_state.emails_remaining <= 0:
            return self._finalize_episode("All emails processed.")

        # Validate email_id
        if action.email_id not in self._ground_truths:
            return TriageObservation(
                done=False,
                reward=0.0,
                inbox=[e for e in self._emails if e.email_id not in self._processed_ids],
                processed_count=self._triage_state.emails_processed,
                total_count=self._triage_state.total_emails,
                current_score=self._triage_state.score,
                step_reward=0.0,
                feedback=f"Invalid email_id: '{action.email_id}'. Use an ID from the inbox.",
                task_id=self._triage_state.task_id,
                task_description=self._current_task.description,
            )

        if action.email_id in self._processed_ids:
            return TriageObservation(
                done=False,
                reward=0.0,
                inbox=[e for e in self._emails if e.email_id not in self._processed_ids],
                processed_count=self._triage_state.emails_processed,
                total_count=self._triage_state.total_emails,
                current_score=self._triage_state.score,
                step_reward=0.0,
                feedback=f"Email '{action.email_id}' was already processed.",
                task_id=self._triage_state.task_id,
                task_description=self._current_task.description,
            )

        # Grade the action
        truth = self._ground_truths[action.email_id]
        reward, feedback = grade_step(action, truth, self._current_task)

        # Update state
        self._processed_ids.add(action.email_id)
        self._action_history.append((action, truth))
        self._triage_state.emails_processed += 1
        self._triage_state.emails_remaining -= 1
        self._triage_state.rewards_history.append(reward)
        self._triage_state.score = (
            sum(self._triage_state.rewards_history) / len(self._triage_state.rewards_history)
        )

        # Check if episode is done
        done = (
            self._triage_state.emails_remaining <= 0
            or self._triage_state.step_count >= self._triage_state.max_steps
        )

        if done:
            final_score, summary = grade_episode(
                self._action_history,
                self._current_task,
                self._triage_state.step_count,
            )
            self._triage_state.score = final_score
            feedback += f" | EPISODE COMPLETE: {summary}"
            reward = final_score  # overwrite with final episode score

        remaining_emails = [e for e in self._emails if e.email_id not in self._processed_ids]

        return TriageObservation(
            done=done,
            reward=reward,
            inbox=remaining_emails,
            processed_count=self._triage_state.emails_processed,
            total_count=self._triage_state.total_emails,
            current_score=self._triage_state.score,
            step_reward=reward,
            feedback=feedback,
            task_id=self._triage_state.task_id,
            task_description=self._current_task.description,
        )

    def _finalize_episode(self, reason: str) -> TriageObservation:
        """Finalize the episode and return final observation."""
        if self._action_history and self._current_task:
            final_score, summary = grade_episode(
                self._action_history,
                self._current_task,
                self._triage_state.step_count,
            )
            self._triage_state.score = final_score
        else:
            summary = "No actions taken."
            final_score = 0.0

        return TriageObservation(
            done=True,
            reward=final_score,
            inbox=[],
            processed_count=self._triage_state.emails_processed,
            total_count=self._triage_state.total_emails,
            current_score=final_score,
            step_reward=0.0,
            feedback=f"{reason} | {summary}",
            task_id=self._triage_state.task_id,
            task_description=self._current_task.description if self._current_task else "",
        )

    @property
    def state(self) -> TriageState:
        """Get the current environment state."""
        return self._triage_state
