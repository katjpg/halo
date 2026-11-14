from typing import get_args

from halo.llm.agents.optimizer import OPTIMIZER
from halo.llm.agents.orchestrator import ORCHESTRATOR
from halo.llm.agents.researcher import ANALYST, RESEARCHER
from halo.llm.agents.strategist import STRATEGIST
from halo.types import AgentRole

AGENTS = {
    agent.role: agent
    for agent in (ORCHESTRATOR, RESEARCHER, ANALYST, STRATEGIST, OPTIMIZER)
}
assert set(AGENTS) == set(get_args(AgentRole))
