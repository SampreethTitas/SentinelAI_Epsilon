from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import asyncio
import httpx
import logging
import json
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

from email_service import AdvancedEmailService, EmailAnalytics
from config import Config, FeatureFlags

# Configure logging
logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Email log storage (in-memory for demo purposes)
email_logs = []

class EmailRequest(BaseModel):
    prompt: str
    recipients: List[EmailStr]
    subject_hint: Optional[str] = None
    use_guard_prompt: bool = True
    use_adshield: bool = True

class GuardPromptRequest(BaseModel):
    prompt: str
    context: Optional[str] = None

class AdShieldRequest(BaseModel):
    text: str
    check_type: str = "comprehensive"

class DataValutRequest(BaseModel):
    source_db_connection: str
    filter_prompt: str

class EmailLog(BaseModel):
    id: str
    timestamp: str
    prompt: str
    recipients: List[str]
    subject: str
    content: str
    guard_result: Optional[Dict] = None
    adshield_result: Optional[Dict] = None
    status: str
    metadata: Dict[str, Any] = {}

# Backend service URLs
BACKEND_URLS = Config.BACKEND_URLS

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🤖 AI Agent Backend Server Starting Up...")
    logger.info("🔗 Integrating services: GuardPrompt, AdShield, DataValut, Email Agent")
    yield
    # Shutdown
    logger.info("🤖 AI Agent Backend Server Shutting Down...")

app = FastAPI(
    title="SentinelAI Epsilon - AI Agent Backend",
    description="Unified AI Agent backend integrating GuardPrompt, AdShield, DataValut, and Email services",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Email Generation Service
class EmailService:
    @staticmethod
    def generate_email_content(prompt: str, subject_hint: Optional[str] = None) -> Dict[str, str]:
        """Generate email content using the advanced email service"""
        return AdvancedEmailService.generate_complete_email(prompt, subject_hint)

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "🤖 SentinelAI Epsilon AI Agent Backend",
        "version": "1.0.0",
        "services": ["guard_prompt", "adshield", "datavalut", "email_agent"],
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "ai_agent": "operational",
        "services": {}
    }
    
    # Check backend services
    async with httpx.AsyncClient() as client:
        for service, url in BACKEND_URLS.items():
            try:
                response = await client.get(f"{url}/", timeout=5.0)
                health_status["services"][service] = "operational" if response.status_code == 200 else "degraded"
            except Exception:
                health_status["services"][service] = "unavailable"
    
    return health_status

# GuardPrompt Integration
@app.post("/guard-prompt/analyze")
async def analyze_guard_prompt(request: GuardPromptRequest):
    """Analyze text using GuardPrompt service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URLS['main']}/guard-prompt/analyze",
                json={"prompt": request.prompt, "context": request.context},
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="GuardPrompt service error")
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="GuardPrompt service timeout")
    except Exception as e:
        logger.error(f"GuardPrompt error: {e}")
        raise HTTPException(status_code=500, detail=f"GuardPrompt service error: {str(e)}")

# AdShield Integration
@app.post("/adshield/analyze")
async def analyze_adshield(request: AdShieldRequest):
    """Analyze text using AdShield service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URLS['main']}/adshield/analyze",
                json={"text": request.text, "check_type": request.check_type},
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="AdShield service error")
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="AdShield service timeout")
    except Exception as e:
        logger.error(f"AdShield error: {e}")
        raise HTTPException(status_code=500, detail=f"AdShield service error: {str(e)}")

# DataValut Integration
@app.post("/datavalut/create-db")
async def create_filtered_database(request: DataValutRequest):
    """Create filtered database using DataValut service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URLS['datavalut']}/datavalut/create-filtered-db",
                json={
                    "source_db_connection": request.source_db_connection,
                    "filter_prompt": request.filter_prompt
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="DataValut service error")
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="DataValut service timeout")
    except Exception as e:
        logger.error(f"DataValut error: {e}")
        raise HTTPException(status_code=500, detail=f"DataValut service error: {str(e)}")

@app.get("/datavalut/job/{job_id}")
async def get_datavalut_job(job_id: str):
    """Get DataValut job status"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_URLS['datavalut']}/datavalut/job/{job_id}",
                timeout=10.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="DataValut service error")
                
    except Exception as e:
        logger.error(f"DataValut job status error: {e}")
        raise HTTPException(status_code=500, detail=f"DataValut service error: {str(e)}")

# Email Agent
@app.post("/email/send")
async def send_email(request: EmailRequest, background_tasks: BackgroundTasks):
    """Send email using AI agent with optional GuardPrompt and AdShield analysis"""
    
    email_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Initialize email log
    email_log = EmailLog(
        id=email_id,
        timestamp=timestamp,
        prompt=request.prompt,
        recipients=request.recipients,
        subject="",
        content="",
        status="processing"
    )
    
    try:
        # Generate email content
        email_content = EmailService.generate_email_content(request.prompt, request.subject_hint)
        email_log.subject = email_content["subject"]
        email_log.content = email_content["content"]
        email_log.metadata.update({
            "email_type": email_content.get("email_type", "general"),
            "keywords": email_content.get("keywords", []),
            "generation_metadata": email_content.get("metadata", {})
        })
        
        # Optional GuardPrompt analysis
        if request.use_guard_prompt:
            try:
                async with httpx.AsyncClient() as client:
                    guard_response = await client.post(
                        f"{BACKEND_URLS['main']}/guard-prompt/analyze",
                        json={"prompt": request.prompt},
                        timeout=30.0
                    )
                    if guard_response.status_code == 200:
                        email_log.guard_result = guard_response.json()
                        logger.info(f"GuardPrompt analysis completed for email {email_id}")
            except Exception as e:
                logger.warning(f"GuardPrompt analysis failed for email {email_id}: {e}")
                email_log.metadata["guard_error"] = str(e)
        
        # Optional AdShield analysis
        if request.use_adshield:
            try:
                async with httpx.AsyncClient() as client:
                    adshield_response = await client.post(
                        f"{BACKEND_URLS['main']}/adshield/analyze",
                        json={"text": email_content["content"]},
                        timeout=30.0
                    )
                    if adshield_response.status_code == 200:
                        email_log.adshield_result = adshield_response.json()
                        logger.info(f"AdShield analysis completed for email {email_id}")
            except Exception as e:
                logger.warning(f"AdShield analysis failed for email {email_id}: {e}")
                email_log.metadata["adshield_error"] = str(e)
        
        # "Send" email (log it)
        email_log.status = "sent"
        email_logs.append(email_log)
        
        # Log email details
        logger.info(f"📧 EMAIL SENT - ID: {email_id}")
        logger.info(f"📧 TO: {', '.join(request.recipients)}")
        logger.info(f"📧 SUBJECT: {email_content['subject']}")
        logger.info(f"📧 CONTENT: {email_content['content'][:100]}...")
        
        return {
            "success": True,
            "message": "Email sent successfully",
            "email_id": email_id,
            "recipients": request.recipients,
            "subject": email_content["subject"],
            "timestamp": timestamp,
            "guard_analysis": email_log.guard_result is not None,
            "adshield_analysis": email_log.adshield_result is not None
        }
        
    except Exception as e:
        email_log.status = "failed"
        email_log.metadata["error"] = str(e)
        email_logs.append(email_log)
        
        logger.error(f"Email sending failed for {email_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.get("/email/logs")
async def get_email_logs(limit: int = 50):
    """Get email logs"""
    return {
        "logs": email_logs[-limit:],
        "total": len(email_logs)
    }

@app.get("/email/logs/{email_id}")
async def get_email_log(email_id: str):
    """Get specific email log by ID"""
    for log in email_logs:
        if log.id == email_id:
            return log
    
    raise HTTPException(status_code=404, detail="Email log not found")

@app.delete("/email/logs")
async def clear_email_logs():
    """Clear all email logs"""
    global email_logs
    count = len(email_logs)
    email_logs = []
    return {"message": f"Cleared {count} email logs"}

@app.get("/email/analytics")
async def get_email_analytics():
    """Get email analytics and performance metrics"""
    if not FeatureFlags.ENABLE_ANALYTICS:
        raise HTTPException(status_code=404, detail="Analytics feature is disabled")
    
    analytics = EmailAnalytics.analyze_email_performance(email_logs)
    return analytics

@app.post("/email/preview")
async def preview_email(prompt: str, subject_hint: Optional[str] = None):
    """Preview email content without sending"""
    email_content = EmailService.generate_email_content(prompt, subject_hint)
    return {
        "preview": True,
        "subject": email_content["subject"],
        "content": email_content["content"],
        "email_type": email_content.get("email_type"),
        "keywords": email_content.get("keywords"),
        "metadata": email_content.get("metadata")
    }

# Comprehensive AI Agent Endpoint
@app.post("/ai-agent/process")
async def process_with_ai_agent(
    prompt: str,
    recipients: Optional[List[EmailStr]] = None,
    analyze_guard: bool = True,
    analyze_adshield: bool = True,
    send_email: bool = False
):
    """Comprehensive AI agent that can analyze text and optionally send emails"""
    
    results = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "analyses": {}
    }
    
    # GuardPrompt analysis
    if analyze_guard:
        try:
            guard_request = GuardPromptRequest(prompt=prompt)
            guard_result = await analyze_guard_prompt(guard_request)
            results["analyses"]["guard_prompt"] = guard_result
        except Exception as e:
            results["analyses"]["guard_prompt"] = {"error": str(e)}
    
    # AdShield analysis
    if analyze_adshield:
        try:
            adshield_request = AdShieldRequest(text=prompt)
            adshield_result = await analyze_adshield(adshield_request)
            results["analyses"]["adshield"] = adshield_result
        except Exception as e:
            results["analyses"]["adshield"] = {"error": str(e)}
    
    # Send email if requested
    if send_email and recipients:
        try:
            email_request = EmailRequest(
                prompt=prompt,
                recipients=recipients,
                use_guard_prompt=analyze_guard,
                use_adshield=analyze_adshield
            )
            email_result = await send_email(email_request, BackgroundTasks())
            results["email"] = email_result
        except Exception as e:
            results["email"] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
