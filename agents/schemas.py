# agent_server/schemas.py
from pydantic import BaseModel, Field
from typing import List

# --- Schemas for agent generation and discovery ---

class AgentGenerationRequest(BaseModel):
    """The simplified request payload for all agents."""
    prompt: str = Field(..., description="The main instruction, topic, or product description for the agent.")

class AgentGenerationResponse(BaseModel):
    """The simplified response from an agent, containing only the generated text."""
    generated_text: str

class AgentInfo(BaseModel):
    """Describes a single available AI agent."""
    id: str = Field(..., description="A unique identifier for the agent, used in the API path.")
    name: str = Field(..., description="A human-friendly name for the agent.")
    description: str = Field(..., description="A short summary of what the agent does.")
    endpoint: str = Field(..., description="The full API endpoint path to use this agent.")