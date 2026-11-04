import argparse

from genai_engineering.core.providers import create_provider
from genai_engineering.task3_agent.agent import ToolAgent

parser = argparse.ArgumentParser()
parser.add_argument("objective")
args = parser.parse_args()
result = ToolAgent(create_provider()).run(args.objective)
for step in result.trace:
    print(
        f"ACTION: {step.action} → {step.tool}({step.tool_input}) "
        f"→ {step.observation} → NEXT: {step.next_action}"
    )
print("FINAL:", result.answer)
