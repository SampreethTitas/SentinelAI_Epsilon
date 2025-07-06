# agent_server/routes.py
from fastapi import APIRouter, HTTPException
from schemas import AgentGenerationRequest, AgentGenerationResponse, AgentInfo
import agents
from typing import List

# Router for agent discovery
discovery_router = APIRouter()
# Router for agent generation
generation_router = APIRouter(prefix="/generate")


@discovery_router.get("/agents", response_model=List[AgentInfo])
async def list_available_agents():
    """Returns a list of all available marketing AI agents."""
    return agents.AGENT_REGISTRY


# --- Simplified Generation Routes for Marketing Agents ---

@generation_router.post("/email", response_model=AgentGenerationResponse)
async def generate_email_agent(request: AgentGenerationRequest):
    """Generates a marketing email."""
    try:
        generated_text = await agents.generate_email(request.prompt)
        return AgentGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI agent failed to generate content: {e}")

@generation_router.post("/blog", response_model=AgentGenerationResponse)
async def generate_blog_agent(request: AgentGenerationRequest):
    """Generates a marketing blog post."""
    try:
        generated_text = await agents.generate_blog_post(request.prompt)
        return AgentGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI agent failed to generate content: {e}")

@generation_router.post("/social-post", response_model=AgentGenerationResponse)
async def generate_social_post_agent(request: AgentGenerationRequest):
    """Generates a social media post."""
    try:
        generated_text = await agents.generate_social_post(request.prompt)
        return AgentGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI agent failed to generate content: {e}")

@generation_router.post("/ad-copy", response_model=AgentGenerationResponse)
async def generate_ad_copy_agent(request: AgentGenerationRequest):
    """Generates advertising copy."""
    try:
        generated_text = await agents.generate_ad_copy(request.prompt)
        return AgentGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI agent failed to generate content: {e}")