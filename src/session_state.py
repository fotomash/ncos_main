"""
Session State Module for NCOS v11.5 Phoenix-Mesh

This module defines the SessionState class which manages the current state
of the system session, including token budget, agent states, and memory.
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import resource
from uuid import uuid4

from pydantic import BaseModel, Field

from .core import TokenBudget, SystemConfig


class AgentState(BaseModel):
    """State of an individual agent in the session."""
    agent_id: str
    initialized: bool = False
    healthy: bool = True
    last_action_time: Optional[float] = None
    execution_count: int = 0
    error_count: int = 0
    token_usage: int = 0


class SessionState(BaseModel):
    """
    Represents the current state of the NCOS session.

    This includes token budget, agent states, and metadata.
    """
    session_id: str = Field(default_factory=lambda: f"session_{uuid4().hex[:8]}")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    last_snapshot_time: Optional[float] = None

    token_budget: TokenBudget
    config: SystemConfig

    # Agent states
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)

    # Execution state
    current_action_id: Optional[str] = None
    execution_phase: str = "initializing"
    execution_queue: List[str] = Field(default_factory=list)

    # Memory state
    memory_namespaces: Set[str] = Field(default_factory=set)
    memory_limit_mb: int = 1024
    current_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # Metrics
    total_tokens_used: int = 0
    total_actions_executed: int = 0

    class Config:
        arbitrary_types_allowed = True

    def update_agent_state(self, agent_id: str, **kwargs) -> None:
        """Update the state of an agent."""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentState(agent_id=agent_id)

        for key, value in kwargs.items():
            if hasattr(self.agent_states[agent_id], key):
                setattr(self.agent_states[agent_id], key, value)

        self.last_updated = datetime.now()

    def update_token_usage(self, tokens_used: int) -> None:
        """Update token usage metrics."""
        self.total_tokens_used += tokens_used
        self.token_budget.used += tokens_used
        self.last_updated = datetime.now()

    def increment_action_count(self) -> None:
        """Increment the action execution count."""
        self.total_actions_executed += 1
        self.last_updated = datetime.now()

    def update_memory_usage(self) -> None:
        """Update memory usage statistics."""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        self.current_memory_mb = usage
        if usage > self.peak_memory_mb:
            self.peak_memory_mb = usage
        self.last_updated = datetime.now()

    def memory_exceeded(self) -> bool:
        """Check if memory usage exceeds the configured limit."""
        return self.current_memory_mb >= self.memory_limit_mb

    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current session state."""
        return self.dict(exclude={"config"})

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore session state from a snapshot."""
        for key, value in snapshot.items():
            if hasattr(self, key) and key != "config":
                setattr(self, key, value)

        self.last_updated = datetime.now()
