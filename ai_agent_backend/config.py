"""
Configuration for AI Agent Backend
"""

import os
from typing import Dict

class Config:
    # Server Configuration
    HOST = os.getenv("AI_AGENT_HOST", "0.0.0.0")
    PORT = int(os.getenv("AI_AGENT_PORT", 8003))
    
    # Backend Service URLs
    BACKEND_URLS = {
        "main": os.getenv("MAIN_BACKEND_URL", "http://127.0.0.1:8000"),
        "datavalut": os.getenv("DATAVALUT_BACKEND_URL", "http://127.0.0.1:8002"),
        "test_agents": os.getenv("TEST_AGENTS_BACKEND_URL", "http://127.0.0.1:8001")
    }
    
    # CORS Configuration
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    
    # Timeouts (in seconds)
    TIMEOUTS = {
        "guard_prompt": 30,
        "adshield": 30,
        "datavalut": 60,
        "health_check": 5
    }
    
    # Email Configuration
    EMAIL_CONFIG = {
        "max_recipients": 100,
        "max_content_length": 10000,
        "log_retention_days": 30,
        "enable_real_sending": False  # Set to True for actual email sending
    }
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Feature Flags
class FeatureFlags:
    ENABLE_GUARD_PROMPT = True
    ENABLE_ADSHIELD = True
    ENABLE_DATAVALUT = True
    ENABLE_EMAIL_AGENT = True
    ENABLE_ANALYTICS = True
    ENABLE_RATE_LIMITING = False  # For future implementation

# API Rate Limits (for future implementation)
class RateLimits:
    EMAIL_PER_MINUTE = 60
    ANALYSIS_PER_MINUTE = 120
    DATABASE_OPS_PER_HOUR = 10
