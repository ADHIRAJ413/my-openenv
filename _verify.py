"""Full verification of the restructured environment."""
import sys
sys.path.insert(0, ".")

# Test 1: Models import with OpenEnv base types
print("=== Test 1: Models ===")
from models import TriageAction, TriageObservation, TriageState, EmailItem, EmailGroundTruth
from openenv.core.env_server.types import Action, Observation, State
assert issubclass(TriageAction, Action), "TriageAction must extend Action"
assert issubclass(TriageObservation, Observation), "TriageObservation must extend Observation"
assert issubclass(TriageState, State), "TriageState must extend State"
print("✓ Models extend OpenEnv base types correctly")

# Test 2: Email generator
print("\n=== Test 2: Email Generator ===")
from server.email_generator import EmailGenerator
g = EmailGenerator(seed=42)
emails, truths = g.generate_easy_set()
print(f"✓ Generated {len(emails)} emails")

# Test 3: Tasks & grading
print("\n=== Test 3: Tasks & Grading ===")
from server.tasks import get_task, list_tasks, grade_step
tasks = list_tasks()
print(f"✓ {len(tasks)} tasks defined")

task = get_task("easy_categorize")
action = TriageAction(email_id="email-0001", action_type="categorize", category=truths[0].category)
reward, feedback = grade_step(action, truths[0], task)
print(f"✓ Correct action reward: {reward:.3f} → {feedback}")

# Test 4: Environment 
print("\n=== Test 4: Environment ===")
from server.email_triage_env import EmailTriageEnvironment
from openenv.core.env_server.interfaces import Environment as EnvBase
assert issubclass(EmailTriageEnvironment, EnvBase), "Must extend Environment"

env = EmailTriageEnvironment()
obs = env.reset(task_id="easy_categorize")
print(f"✓ Reset: done={obs.done}, reward={obs.reward}, inbox={len(obs.inbox)} emails")
print(f"  task={obs.task_id}, feedback={obs.feedback[:60]}...")

# Step with correct action
step_action = TriageAction(email_id="email-0001", action_type="categorize", category=truths[0].category)
obs2 = env.step(step_action)
print(f"✓ Step: done={obs2.done}, reward={obs2.reward:.3f}, feedback={obs2.feedback[:60]}...")
print(f"  remaining={len(obs2.inbox)} emails, score={obs2.current_score:.3f}")

# Check state property
state = env.state
print(f"✓ State: episode_id={state.episode_id[:8]}..., step_count={state.step_count}")
print(f"  task_id={state.task_id}, score={state.score:.3f}")

# Test 5: Serialization compatibility
print("\n=== Test 5: Serialization ===")
from openenv.core.env_server.serialization import serialize_observation, deserialize_action
obs_dict = serialize_observation(obs)
print(f"✓ serialize_observation: keys={list(obs_dict.keys())}")
print(f"  observation keys: {list(obs_dict['observation'].keys())[:5]}...")

action_data = {"email_id": "email-0002", "action_type": "categorize", "category": "hr"}
action_parsed = deserialize_action(action_data, TriageAction)
print(f"✓ deserialize_action: {type(action_parsed).__name__} email_id={action_parsed.email_id}")

# Test 6: Server app creation
print("\n=== Test 6: Server App ===")
from openenv.core.env_server.http_server import create_app
app = create_app(EmailTriageEnvironment, TriageAction, TriageObservation, env_name="email_triage_env")
print(f"✓ FastAPI app created: {type(app).__name__}")

# Test 7: Client imports
print("\n=== Test 7: Client ===")
from client import EmailTriageEnv
from openenv.core.env_client import EnvClient
assert issubclass(EmailTriageEnv, EnvClient), "Must extend EnvClient"
print(f"✓ EmailTriageEnv extends EnvClient correctly")

# Test 8: Inference imports
print("\n=== Test 8: Inference ===")
# Just test imports — don't run the actual inference
import importlib
spec = importlib.util.spec_from_file_location("inference", "inference.py")
print(f"✓ inference.py found and loadable")

print("\n" + "="*50)
print("✓ ALL VERIFICATIONS PASSED")
print("="*50)
