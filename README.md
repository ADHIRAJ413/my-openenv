# 📧 Email Triage Environment — OpenEnv

> **A real-world OpenEnv environment where AI agents learn to manage an email inbox through categorization, prioritization, routing, and response drafting.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 Overview

The **Email Triage Environment** simulates a real-world email inbox management task — something knowledge workers do every day. An AI agent must process an inbox of synthetic but realistic emails by:

- **Categorizing** emails (billing, technical, sales, HR, spam)
- **Prioritizing** them (critical → low)
- **Routing** to the correct department
- **Drafting responses** for emails that need them
- **Detecting spam/phishing** attempts

This is not a toy or game environment. Email triage is a genuine business task with clear success criteria, graduated difficulty, and meaningful partial progress signals.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Agent (Client)                │
│  ┌────────────────────────────────────┐ │
│  │  EmailTriageEnv (MCPToolClient)    │ │
│  │  - list_tools()                    │ │
│  │  - call_tool("list_inbox")         │ │
│  │  - call_tool("process_email",...)  │ │
│  │  - reset(task_id="...")            │ │
│  └──────────────┬─────────────────────┘ │
└─────────────────┼───────────────────────┘
                  │ WebSocket
┌─────────────────▼───────────────────────┐
│     Docker Container (Server)           │
│  ┌────────────────────────────────────┐ │
│  │  FastAPI + EmailTriageEnvironment  │ │
│  │  - MCP tools for inbox mgmt       │ │
│  │  - Per-step reward computation     │ │
│  │  - Deterministic graders           │ │
│  │  - Synthetic email generation      │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

---

## 📋 Tasks

### Task 1: Basic Email Categorization (Easy)
| Property | Value |
|---|---|
| **ID** | `easy_categorize` |
| **Emails** | 10 |
| **Max Steps** | 15 |
| **Objective** | Categorize each email into the correct category |
| **Grading** | Category accuracy only |
| **Expected Baseline** | ~0.85 |

### Task 2: Priority Triage & Routing (Medium)
| Property | Value |
|---|---|
| **ID** | `medium_triage` |
| **Emails** | 20 |
| **Max Steps** | 30 |
| **Objective** | Categorize + prioritize + route each email |
| **Grading** | 40% category, 30% priority, 30% routing |
| **Expected Baseline** | ~0.65 |

### Task 3: Full Inbox Management (Hard)
| Property | Value |
|---|---|
| **ID** | `hard_full_triage` |
| **Emails** | 30 |
| **Max Steps** | 40 |
| **Objective** | Full triage: categorize, prioritize, route, draft responses, detect phishing |
| **Grading** | 25% category, 20% priority, 20% routing, 20% response, 10% spam detection, 5% efficiency |
| **Expected Baseline** | ~0.45 |

---

## 🔧 Action Space

```python
class TriageAction(BaseModel):
    email_id: str              # Which email to act on
    action_type: Literal[      # What to do
        "categorize",          # Assign a category
        "prioritize",          # Set priority level
        "route",               # Route to department
        "draft_response",      # Write a reply
        "mark_spam",           # Flag as spam/phishing
        "skip",                # Skip (small penalty)
    ]
    category: Optional[str]    # billing, technical, sales, hr, general
    priority: Optional[str]    # critical, high, medium, low
    route_to: Optional[str]    # billing_dept, engineering, sales_team, hr_dept, security, general
    draft_response: Optional[str]  # Reply text
```

The environment exposes these as MCP tools:
- `list_inbox()` — View unprocessed emails
- `process_email(email_id, action_type, ...)` — Act on an email
- `get_task_info()` — Current task description
- `list_available_tasks()` — All tasks
- `get_score()` — Running score

## 👁️ Observation Space

```python
class TriageObservation(BaseModel):
    inbox: list[EmailItem]       # Unprocessed emails
    processed_count: int         # Emails processed so far
    total_count: int             # Total in episode
    current_score: float         # Running score (0.0-1.0)
    step_reward: float           # Last action's reward
    feedback: str                # Natural language feedback
    done: bool                   # Episode over?
    task_id: str                 # Current task
    task_description: str        # Task objective
```

Each `EmailItem` contains:
```python
class EmailItem(BaseModel):
    email_id: str
    sender: str
    sender_name: str
    subject: str
    body: str
    timestamp: str
    has_attachment: bool
    reply_to: Optional[str]
    thread_length: int
    is_read: bool
```

---

## 🏆 Reward Function

The reward function provides **dense per-step signals** (not just binary end-of-episode):

| Component | Weight (Hard) | Scoring |
|---|---|---|
| **Category** | 25% | Exact match: 1.0, close match: 0.3, wrong: 0.0 |
| **Priority** | 20% | Exact: 1.0, within-1: 0.5, within-2: 0.15, wrong: 0.0 |
| **Routing** | 20% | Exact department match: 1.0, wrong: 0.0 |
| **Response** | 20% | Length + professionalism + key points coverage |
| **Spam Detection** | 10% | Correct detection: 1.0, miss: 0.0, false positive: 0.0 |
| **Efficiency** | 5% | Bonus for processing all emails with fewer steps |

**Penalties:**
- Skipping an email: −0.1
- Missing critical/phishing emails: implicitly 0.0 for detection
- Not processing all emails: efficiency malus

---

## 🚀 Setup & Usage

### Prerequisites
- Python ≥ 3.10
- Docker (for containerized execution)

### Install

```bash
# Clone the repository
git clone <repo-url>
cd email_triage_env

# Install the environment
pip install -e .

# Install with baseline script dependencies
pip install -e ".[baseline]"

# Install with dev dependencies
pip install -e ".[all]"
```

### Run Locally (No Docker)

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Or directly:
python -m server.app
```

### Run with Docker

```bash
# Build the image
docker build -t email-triage-env -f server/Dockerfile .

# Run the container
docker run -p 8000:8000 email-triage-env

# Verify health
curl http://localhost:8000/health
```

### Use the Client

```python
from email_triage_env import EmailTriageEnv

# Connect to running server
with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
    # Reset with a task
    env.reset(task_id="easy_categorize")

    # List inbox
    inbox = env.call_tool("list_inbox")
    print(f"Got {inbox['count']} emails")

    # Process an email
    result = env.call_tool(
        "process_email",
        email_id="email-0001",
        action_type="categorize",
        category="billing"
    )
    print(f"Reward: {result['reward']}, Feedback: {result['feedback']}")
```

### Run Baseline

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run all tasks
python baseline/run_baseline.py

# Run with custom model
python baseline/run_baseline.py --model gpt-4o

# Run a single task
python baseline/run_baseline.py --task easy_categorize
```

---

## 📊 Baseline Scores

Scores produced by `baseline/run_baseline.py` with default settings (GPT-4o-mini, seed=42):

| Task | Score | Steps | Description |
|---|---|---|---|
| `easy_categorize` | ~0.85 | ~12 | Basic categorization of 10 clear emails |
| `medium_triage` | ~0.65 | ~25 | Category + priority + routing for 20 emails |
| `hard_full_triage` | ~0.45 | ~38 | Full management of 30 emails incl. phishing |
| **Average** | **~0.65** | | |

> Note: Scores are approximate and may vary slightly between API calls due to model behavior, but the environment and grading are fully deterministic for the same seed.

---

## 🐳 Hugging Face Spaces Deployment

```bash
# Using OpenEnv CLI
pip install openenv-core[cli]
openenv push --repo-id your-username/email-triage-env

# Or manual upload to HF Spaces
# 1. Create a new Space (Docker type)
# 2. Upload all files
# 3. Space will auto-build from server/Dockerfile
```

---

## 📁 Project Structure

```
email_triage_env/
├── __init__.py              # Package exports
├── models.py                # Pydantic Action/Observation/State models
├── client.py                # MCPToolClient subclass
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package configuration
├── README.md                # This file
├── .dockerignore            # Docker build exclusions
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI app (create_app factory)
│   ├── email_triage_env.py  # Core Environment implementation
│   ├── tasks.py             # Task definitions & graders
│   ├── email_generator.py   # Synthetic email generation
│   ├── requirements.txt     # Server dependencies
│   └── Dockerfile           # Container image definition
├── baseline/
│   ├── run_baseline.py      # OpenAI API baseline script
│   └── results/             # Saved baseline scores
└── outputs/
    ├── logs/
    └── evals/
```

---

## 🧪 Validation

```bash
# Validate OpenEnv compliance
openenv validate

# Run tests
python -m pytest tests/ -v

# Test email generation reproducibility
python -c "
from server.email_generator import EmailGenerator
g = EmailGenerator(seed=42)
emails, truths = g.generate_easy_set()
print(f'Generated {len(emails)} emails')
for e, t in zip(emails[:3], truths[:3]):
    print(f'  {e.email_id}: {e.subject[:50]}... → {t.category}/{t.priority}')
"
```

---

## 📜 License

MIT License
