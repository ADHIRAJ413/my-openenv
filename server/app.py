"""
FastAPI application for the Email Triage Environment.

Creates an HTTP server that exposes the EmailTriageEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

from models import TriageAction, TriageObservation
from server.email_triage_env import EmailTriageEnvironment

# Create the app with typed Action and Observation classes
app = create_app(
    EmailTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="email_triage_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
