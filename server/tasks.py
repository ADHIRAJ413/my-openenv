"""
Task Definitions & Graders for the Email Triage environment.

Defines 3 tasks (easy, medium, hard) with programmatic graders
that score agent performance from 0.0 to 1.0.
"""

from dataclasses import dataclass, field
from typing import Optional
from models import TriageAction, EmailGroundTruth


@dataclass
class TaskDefinition:
    """Defines a single task with its parameters and grading criteria."""
    task_id: str
    name: str
    description: str
    difficulty: str
    email_count: int
    max_steps: int
    required_actions: list[str]
    category_weight: float = 0.0
    priority_weight: float = 0.0
    routing_weight: float = 0.0
    response_weight: float = 0.0
    spam_detection_weight: float = 0.0
    efficiency_bonus: float = 0.0


# ── Task Definitions ──────────────────────────────────────────────

TASKS: dict[str, TaskDefinition] = {
    "easy_categorize": TaskDefinition(
        task_id="easy_categorize",
        name="Basic Email Categorization",
        description=(
            "You are an email triage assistant. Your job is to categorize each email "
            "in the inbox into one of these categories: billing, technical, sales, hr, or spam. "
            "For each email, use the 'categorize' action and specify the correct category. "
            "Spam emails should be identified using the 'mark_spam' action instead."
        ),
        difficulty="easy",
        email_count=10,
        max_steps=15,
        required_actions=["categorize", "mark_spam"],
        category_weight=1.0,
    ),

    "medium_triage": TaskDefinition(
        task_id="medium_triage",
        name="Priority Triage & Routing",
        description=(
            "You are an email triage assistant handling a busy inbox. For each email you must:\n"
            "1. Categorize it (billing, technical, sales, hr, or spam)\n"
            "2. Set its priority level (critical, high, medium, or low)\n"
            "3. Route it to the correct department (billing_dept, engineering, sales_team, hr_dept, security, general)\n"
            "Spam and phishing emails should be marked with 'mark_spam' and routed to security.\n"
            "Be efficient — you have limited steps."
        ),
        difficulty="medium",
        email_count=20,
        max_steps=30,
        required_actions=["categorize", "prioritize", "route", "mark_spam"],
        category_weight=0.40,
        priority_weight=0.30,
        routing_weight=0.30,
    ),

    "hard_full_triage": TaskDefinition(
        task_id="hard_full_triage",
        name="Full Inbox Management",
        description=(
            "You are a senior email triage assistant managing a critical inbox. For each email:\n"
            "1. Categorize it (billing, technical, sales, hr, or spam)\n"
            "2. Set priority (critical, high, medium, low)\n"
            "3. Route to the correct department\n"
            "4. Draft a professional response for emails that require one\n"
            "5. Detect and flag spam/phishing attempts (mark_spam action)\n\n"
            "IMPORTANT: Some emails are phishing attempts disguised as legitimate messages. "
            "Look for suspicious sender addresses, urgent language, and requests for sensitive info.\n\n"
            "You have limited steps, so be efficient. Critical emails should be handled first."
        ),
        difficulty="hard",
        email_count=30,
        max_steps=40,
        required_actions=["categorize", "prioritize", "route", "draft_response", "mark_spam"],
        category_weight=0.25,
        priority_weight=0.20,
        routing_weight=0.20,
        response_weight=0.20,
        spam_detection_weight=0.10,
        efficiency_bonus=0.05,
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    """Get a task definition by ID."""
    if task_id not in TASKS:
        available = ", ".join(TASKS.keys())
        raise ValueError(f"Unknown task_id '{task_id}'. Available tasks: {available}")
    return TASKS[task_id]


def list_tasks() -> list[dict]:
    """List all available tasks with metadata."""
    return [
        {
            "task_id": t.task_id,
            "name": t.name,
            "difficulty": t.difficulty,
            "email_count": t.email_count,
            "description": t.description,
        }
        for t in TASKS.values()
    ]


# ── Grading Functions ─────────────────────────────────────────────

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _grade_category(action: TriageAction, truth: EmailGroundTruth) -> float:
    """Grade category accuracy. Returns 0.0 or 1.0."""
    if action.action_type == "mark_spam":
        return 1.0 if truth.is_spam else 0.0

    if action.category is None:
        return 0.0

    agent_cat = action.category.lower().strip()
    true_cat = truth.category.lower().strip()

    if agent_cat == true_cat:
        return 1.0

    # Partial credit for close matches
    close_pairs = [
        ("billing", "sales"),  # Sometimes overlap
        ("technical", "general"),
    ]
    for a, b in close_pairs:
        if (agent_cat == a and true_cat == b) or (agent_cat == b and true_cat == a):
            return 0.3

    return 0.0


def _grade_priority(action: TriageAction, truth: EmailGroundTruth) -> float:
    """Grade priority accuracy. Returns 0.0-1.0 with partial credit."""
    if action.priority is None:
        return 0.0

    agent_p = action.priority.lower().strip()
    true_p = truth.priority.lower().strip()

    if agent_p == true_p:
        return 1.0

    # Partial credit for being within 1 level
    agent_idx = PRIORITY_ORDER.get(agent_p, -1)
    true_idx = PRIORITY_ORDER.get(true_p, -1)

    if agent_idx < 0 or true_idx < 0:
        return 0.0

    diff = abs(agent_idx - true_idx)
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.15
    return 0.0


def _grade_routing(action: TriageAction, truth: EmailGroundTruth) -> float:
    """Grade routing accuracy. Returns 0.0 or 1.0."""
    if action.route_to is None:
        return 0.0

    agent_dept = action.route_to.lower().strip()
    true_dept = truth.department.lower().strip()

    return 1.0 if agent_dept == true_dept else 0.0


def _grade_response(action: TriageAction, truth: EmailGroundTruth) -> float:
    """Grade draft response quality using keyword matching and heuristics."""
    if not truth.requires_response:
        # No response needed — reward for not drafting one
        if action.draft_response is None or action.draft_response.strip() == "":
            return 1.0
        return 0.5  # Mild penalty for unnecessary response

    if action.draft_response is None or action.draft_response.strip() == "":
        return 0.0  # Didn't respond to an email that needed it

    response = action.draft_response.lower()

    score = 0.0

    # Length check: reasonable response (50-500 chars)
    rlen = len(response)
    if rlen < 20:
        score += 0.05
    elif rlen < 50:
        score += 0.15
    elif rlen <= 500:
        score += 0.3
    else:
        score += 0.2  # Slightly penalize very long responses

    # Professionalism markers
    professional_markers = [
        "thank", "please", "appreciate", "regards", "sincerely",
        "dear", "hi ", "hello", "best", "team",
    ]
    marker_count = sum(1 for m in professional_markers if m in response)
    score += min(0.2, marker_count * 0.05)

    # Key response points coverage
    if truth.key_response_points:
        points_covered = 0
        for point in truth.key_response_points:
            # Check if any keyword from the point appears in the response
            point_words = point.lower().split()
            if any(word in response for word in point_words if len(word) > 3):
                points_covered += 1
        coverage = points_covered / len(truth.key_response_points)
        score += 0.5 * coverage

    return min(1.0, score)


def _grade_spam_detection(action: TriageAction, truth: EmailGroundTruth) -> float:
    """Grade spam/phishing detection accuracy."""
    is_marked_spam = action.action_type == "mark_spam"

    if truth.is_spam or truth.is_phishing:
        if is_marked_spam:
            return 1.0  # Correctly detected spam/phishing
        return 0.0  # Missed spam/phishing
    else:
        if is_marked_spam:
            return 0.0  # False positive — marked legitimate as spam
        return 1.0  # Correctly left legitimate alone


def grade_step(
    action: TriageAction,
    truth: EmailGroundTruth,
    task: TaskDefinition,
) -> tuple[float, str]:
    """
    Grade a single step action against ground truth.

    Returns:
        Tuple of (reward: float 0.0-1.0, feedback: str)
    """
    scores = {}
    feedback_parts = []

    # Category grading
    if task.category_weight > 0:
        cat_score = _grade_category(action, truth)
        scores["category"] = cat_score * task.category_weight
        if cat_score == 1.0:
            feedback_parts.append("✓ Category correct")
        elif cat_score > 0:
            feedback_parts.append(f"~ Category partially correct ({cat_score:.0%})")
        else:
            feedback_parts.append(f"✗ Category wrong (expected: {truth.category})")

    # Priority grading
    if task.priority_weight > 0:
        pri_score = _grade_priority(action, truth)
        scores["priority"] = pri_score * task.priority_weight
        if pri_score == 1.0:
            feedback_parts.append("✓ Priority correct")
        elif pri_score > 0:
            feedback_parts.append(f"~ Priority close ({pri_score:.0%})")
        else:
            feedback_parts.append(f"✗ Priority wrong (expected: {truth.priority})")

    # Routing grading
    if task.routing_weight > 0:
        route_score = _grade_routing(action, truth)
        scores["routing"] = route_score * task.routing_weight
        if route_score == 1.0:
            feedback_parts.append("✓ Routing correct")
        else:
            feedback_parts.append(f"✗ Routing wrong (expected: {truth.department})")

    # Response grading
    if task.response_weight > 0:
        resp_score = _grade_response(action, truth)
        scores["response"] = resp_score * task.response_weight
        if resp_score >= 0.8:
            feedback_parts.append("✓ Good response")
        elif resp_score >= 0.4:
            feedback_parts.append("~ Response needs improvement")
        elif truth.requires_response:
            feedback_parts.append("✗ Response missing or poor")
        else:
            feedback_parts.append("✓ No response needed (correct)")

    # Spam detection grading
    if task.spam_detection_weight > 0:
        spam_score = _grade_spam_detection(action, truth)
        scores["spam_detection"] = spam_score * task.spam_detection_weight
        if truth.is_spam or truth.is_phishing:
            if spam_score == 1.0:
                label = "phishing" if truth.is_phishing else "spam"
                feedback_parts.append(f"✓ Correctly detected {label}")
            else:
                feedback_parts.append("✗ Missed spam/phishing!")

    # Skip penalty
    if action.action_type == "skip":
        scores["skip_penalty"] = -0.1
        feedback_parts.append("⚠ Email skipped (-0.1 penalty)")

    total = sum(scores.values())
    total = max(0.0, min(1.0, total))

    feedback = " | ".join(feedback_parts)
    return total, feedback


def grade_episode(
    actions: list[tuple[TriageAction, EmailGroundTruth]],
    task: TaskDefinition,
    total_steps: int,
) -> tuple[float, str]:
    """
    Grade an entire episode.

    Args:
        actions: List of (action, ground_truth) pairs for processed emails
        task: The task definition
        total_steps: Total steps taken in the episode

    Returns:
        Tuple of (final_score: float 0.0-1.0, summary: str)
    """
    if not actions:
        return 0.0, "No emails were processed."

    step_scores = []
    for action, truth in actions:
        score, _ = grade_step(action, truth, task)
        step_scores.append(score)

    # Base score: mean of all step scores
    base_score = sum(step_scores) / len(step_scores)

    # Efficiency bonus: reward for processing all emails in fewer steps
    efficiency = 0.0
    if task.efficiency_bonus > 0:
        emails_processed = len(actions)
        if emails_processed >= task.email_count:
            # Processed all emails
            step_ratio = total_steps / task.email_count
            if step_ratio <= 1.1:  # Very efficient
                efficiency = task.efficiency_bonus
            elif step_ratio <= 1.5:
                efficiency = task.efficiency_bonus * 0.5
        # Penalty for not processing all emails
        coverage = emails_processed / task.email_count
        if coverage < 1.0:
            efficiency -= (1.0 - coverage) * 0.1

    final_score = max(0.0, min(1.0, base_score + efficiency))

    summary_parts = [
        f"Emails processed: {len(actions)}/{task.email_count}",
        f"Average step score: {base_score:.3f}",
        f"Total steps used: {total_steps}/{task.max_steps}",
    ]
    if task.efficiency_bonus > 0:
        summary_parts.append(f"Efficiency bonus: {efficiency:+.3f}")
    summary_parts.append(f"Final score: {final_score:.3f}")

    return final_score, " | ".join(summary_parts)
