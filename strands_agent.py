"""The same gate, inside a real Strands agent.

agent.py calls the gate directly, which is the clearest way to show what it
does. This is the drop-in version: a stock Strands agent with a retriever tool,
and the gate attached as an InterventionHandler that the agent's own code knows
nothing about.

    python strands_agent.py "our nightly payout batch keeps failing"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from strands import Agent, tool
from strands.models import BedrockModel

from agent import retrieve
from patterns import ContextGate
from patterns.strands_wiring import ContextGateIntervention
from strands_decider.infer import load_engine

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"
BEDROCK_MODEL = "us.amazon.nova-lite-v1:0"
BEDROCK_REGION = "us-east-1"

NOTES = [
    json.loads(l)
    for l in Path("data/memory.jsonl").read_text().splitlines()
    if l.strip()
]


@tool
def search_memory(query: str) -> str:
    """Search internal support notes. Returns JSON {"chunks": [{"id", "text"}]}."""
    hits = retrieve(query, NOTES, 6)
    return json.dumps({"chunks": [{"id": h["id"], "text": h["text"]} for h in hits]})


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    question = " ".join(sys.argv[1:]) or "our nightly payout batch keeps failing"

    gate = ContextGate(load_engine(CHECKPOINT, device=auto_device()))
    agent = Agent(
        model=BedrockModel(model_id=BEDROCK_MODEL, region_name=BEDROCK_REGION),
        tools=[search_memory],
        system_prompt=(
            "You are a support engineer. Search memory, then answer using only "
            "the notes you get back. If they do not answer the question, say so "
            "plainly rather than guessing. Be concise."
        ),
        interventions=[ContextGateIntervention(gate, retriever_tool="search_memory")],
    )

    print(f"Q: {question}\n")
    result = agent(question)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
