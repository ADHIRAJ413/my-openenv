#!/usr/bin/env python3
"""
Baseline Inference Script for the Email Triage OpenEnv Environment.

Uses the OpenAI API client to run a model against all 3 tasks and
produce reproducible baseline scores.

Usage:
    export OPENAI_API_KEY="sk-..."
    python baseline/run_baseline.py

    # Custom model or endpoint:
    OPENAI_API_KEY="sk-..." python baseline/run_baseline.py --model gpt-4o-mini

    # Against a specific server:
    python baseline/run_baseline.py --base-url http://localhost:8000

Environment Variables:
    OPENAI_API_KEY: Required. Your OpenAI API key.
    OPENAI_BASE_URL: Optional. Custom API base URL (for compatible providers).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    from email_triage_env import EmailTriageEnv
except ImportError:
    # Fallback: try importing from parent
    from client import EmailTriageEnv


# ── System Prompts ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage assistant. You process emails in an inbox by:
1. Reading each email carefully
2. Taking the appropriate action using the available tools

AVAILABLE TOOLS:
- list_inbox: View unprocessed emails
- process_email: Act on an email (categorize, prioritize, route, draft response, mark spam)
- get_task_info: See current task requirements
- get_score: Check your running score

CATEGORIES: billing, technical, sales, hr, general
PRIORITIES: critical, high, medium, low
DEPARTMENTS: billing_dept, engineering, sales_team, hr_dept, security, general

SPAM/PHISHING DETECTION:
- Look for: suspicious sender domains, urgency manipulation, requests for credentials/money
- Phishing emails often mimic legitimate senders (e.g., "paypa1.com" instead of "paypal.com")
- Use 'mark_spam' action for spam and phishing

RESPONSE DRAFTING (when required):
- Be professional and concise
- Address the sender's key points
- Use appropriate tone (urgent for critical, conversational for routine)

Always process every email. Start by listing the inbox, then process each one systematically."""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_inbox",
            "description": "List all unprocessed emails in the inbox.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_email",
            "description": "Process an email by taking a triage action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "The ID of the email to process (e.g. 'email-0001')",
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["categorize", "prioritize", "route", "draft_response", "mark_spam", "skip"],
                        "description": "Type of action to perform",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category: billing, technical, sales, hr, general (for categorize action)",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                        "description": "Priority level (for prioritize action)",
                    },
                    "route_to": {
                        "type": "string",
                        "description": "Department: billing_dept, engineering, sales_team, hr_dept, security, general (for route action)",
                    },
                    "draft_response": {
                        "type": "string",
                        "description": "Draft reply text (for draft_response action)",
                    },
                },
                "required": ["email_id", "action_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_info",
            "description": "Get information about the current task.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_score",
            "description": "Get current running score and statistics.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Baseline Runner ───────────────────────────────────────────────

class BaselineRunner:
    """Runs baseline inference against the Email Triage environment."""

    def __init__(
        self,
        env_base_url: str,
        model: str = "gpt-4o-mini",
        openai_base_url: str | None = None,
        max_turns: int = 60,
        verbose: bool = True,
    ):
        self.env_base_url = env_base_url
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        client_kwargs = {"api_key": api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url

        self.openai_client = OpenAI(**client_kwargs)

    def _log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    def _call_env_tool(self, env, tool_name: str, arguments: dict) -> dict:
        """Call an environment MCP tool and return the result."""
        try:
            result = env.call_tool(tool_name, **arguments)
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {"result": result}
            return result if isinstance(result, dict) else {"result": str(result)}
        except Exception as e:
            return {"error": str(e)}

    def run_task(self, task_id: str) -> dict:
        """
        Run a single task and return the results.

        Returns:
            Dict with task_id, score, steps, duration, and action history.
        """
        self._log(f"\n{'='*60}")
        self._log(f"Running task: {task_id}")
        self._log(f"{'='*60}")

        start_time = time.time()

        with EmailTriageEnv(base_url=self.env_base_url).sync() as env:
            # Reset environment with task
            env.reset(task_id=task_id)
            self._log(f"Environment reset with task: {task_id}")

            # Initialize conversation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"You are now working on the '{task_id}' task. "
                        "Start by getting the task info, then list the inbox, "
                        "and process every email systematically. "
                        "Be thorough and process ALL emails."
                    ),
                },
            ]

            action_history = []
            done = False
            turn = 0
            final_score = 0.0

            while not done and turn < self.max_turns:
                turn += 1
                self._log(f"\n--- Turn {turn} ---")

                # Call OpenAI
                try:
                    response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        tool_choice="auto",
                        temperature=0.1,
                    )
                except Exception as e:
                    self._log(f"OpenAI API error: {e}")
                    break

                choice = response.choices[0]
                message = choice.message

                # Add assistant message to conversation
                messages.append(message.model_dump())

                # Check if model wants to call tools
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        fn_name = tool_call.function.name
                        try:
                            fn_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}

                        self._log(f"Tool call: {fn_name}({json.dumps(fn_args, indent=None)[:100]}...)")

                        # Execute tool against environment
                        result = self._call_env_tool(env, fn_name, fn_args)

                        # Track process_email actions
                        if fn_name == "process_email":
                            action_history.append({
                                "email_id": fn_args.get("email_id", ""),
                                "action_type": fn_args.get("action_type", ""),
                                "result": result,
                            })
                            if result.get("done", False):
                                done = True
                                final_score = result.get("current_score", 0.0)
                                self._log(f"Episode done! Score: {final_score:.4f}")

                        # Log result summary
                        if fn_name == "process_email":
                            self._log(
                                f"  → reward={result.get('reward', 0):.3f} "
                                f"feedback={result.get('feedback', '')[:80]}"
                            )
                        elif fn_name == "list_inbox":
                            self._log(f"  → {result.get('count', '?')} emails in inbox")

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, default=str)[:4000],
                        })
                else:
                    # No tool calls — model is done or giving text response
                    if message.content:
                        self._log(f"Model response: {message.content[:200]}")

                    # Check if we should prompt for more
                    if not done:
                        # Get score to check status
                        score_result = self._call_env_tool(env, "get_score", {})
                        remaining = score_result.get("emails_remaining", 0)

                        if remaining == 0:
                            done = True
                            final_score = score_result.get("current_score", 0.0)
                        else:
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"There are still {remaining} emails to process. "
                                    "Please continue processing all remaining emails."
                                ),
                            })

                # Safety: check if environment says we're done
                if not done:
                    try:
                        score_check = self._call_env_tool(env, "get_score", {})
                        if score_check.get("emails_remaining", 1) == 0:
                            done = True
                            final_score = score_check.get("current_score", 0.0)
                    except Exception:
                        pass

        duration = time.time() - start_time

        result = {
            "task_id": task_id,
            "score": final_score,
            "steps": turn,
            "actions": len(action_history),
            "duration_seconds": round(duration, 2),
            "model": self.model,
        }

        self._log(f"\nTask {task_id} complete:")
        self._log(f"  Score: {final_score:.4f}")
        self._log(f"  Steps: {turn}")
        self._log(f"  Actions: {len(action_history)}")
        self._log(f"  Duration: {duration:.1f}s")

        return result

    def run_all_tasks(self) -> dict:
        """Run all 3 tasks and produce a summary report."""
        print(f"\n{'#'*60}")
        print(f"# Email Triage Baseline — Model: {self.model}")
        print(f"# Timestamp: {datetime.now().isoformat()}")
        print(f"# Environment: {self.env_base_url}")
        print(f"{'#'*60}")

        tasks = ["easy_categorize", "medium_triage", "hard_full_triage"]
        results = []

        for task_id in tasks:
            try:
                result = self.run_task(task_id)
                results.append(result)
            except Exception as e:
                print(f"\nERROR running task {task_id}: {e}")
                results.append({
                    "task_id": task_id,
                    "score": 0.0,
                    "steps": 0,
                    "actions": 0,
                    "duration_seconds": 0.0,
                    "model": self.model,
                    "error": str(e),
                })

        # Print summary
        print(f"\n{'='*60}")
        print("BASELINE RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Task':<25} {'Score':>8} {'Steps':>8} {'Actions':>8} {'Time(s)':>8}")
        print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        total_score = 0.0
        for r in results:
            print(
                f"{r['task_id']:<25} "
                f"{r['score']:>8.4f} "
                f"{r['steps']:>8} "
                f"{r['actions']:>8} "
                f"{r['duration_seconds']:>8.1f}"
            )
            total_score += r["score"]

        avg_score = total_score / len(results) if results else 0.0
        print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'AVERAGE':<25} {avg_score:>8.4f}")
        print()

        # Save results to file
        output = {
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "environment_url": self.env_base_url,
            "tasks": results,
            "average_score": avg_score,
        }

        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir,
            f"baseline_{self.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results saved to: {results_file}")
        return output


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline inference on the Email Triage environment"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Environment server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Custom OpenAI API base URL (for compatible providers)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=60,
        help="Maximum turns per task (default: 60)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run a specific task only (e.g. easy_categorize)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    runner = BaselineRunner(
        env_base_url=args.base_url,
        model=args.model,
        openai_base_url=args.openai_base_url,
        max_turns=args.max_turns,
        verbose=not args.quiet,
    )

    if args.task:
        result = runner.run_task(args.task)
        print(f"\nFinal score for {args.task}: {result['score']:.4f}")
    else:
        runner.run_all_tasks()


if __name__ == "__main__":
    main()
