"""
AdShield: Marketing Content Security Analyzer
Comprehensive security analysis for marketing content including emails, blogs, social media captions
Built on the GuardPrompt foundation with marketing-specific enhancements
"""

import re
import logging
import hashlib
import time
import json
import io
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import numpy as np
from collections import defaultdict
import threading
from functools import lru_cache
import docx
from docx import Document
import mammoth
# import streamlit as st
from datetime import datetime
# import plotly.graph_objects as go
# import plotly.express as px
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat severity levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    """Types of marketing violations and security threats"""
    ABUSE_CONTENT = "abuse_content"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    ROLE_PLAY_ATTACK = "role_play_attack"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    MALICIOUS_CODE = "malicious_code"
    SOCIAL_ENGINEERING = "social_engineering"
    DATA_EXTRACTION = "data_extraction"
    DECEPTIVE_MARKETING = "deceptive_marketing"
    SPAM_CONTENT = "spam_content"
    PRIVACY_VIOLATION = "privacy_violation"
    BRAND_IMPERSONATION = "brand_impersonation"
    MISLEADING_CLAIMS = "misleading_claims"
    GDPR_VIOLATION = "gdpr_violation"
    DECEPTIVE_CLAIM = "deceptive_claim"
    UNETHICAL_BIAS = "unethical_bias"
    BRAND_VIOLATION = "brand_violation"
    PII_DETECTION = "pii_detection"
    # Marketing-specific additions
    FALSE_ADVERTISING = "false_advertising"
    CLICKBAIT = "clickbait"
    PRICE_MANIPULATION = "price_manipulation"
    FAKE_TESTIMONIALS = "fake_testimonials"
    COMPLIANCE_VIOLATION = "compliance_violation"

class ContentType(Enum):
    """Types of marketing content"""
    EMAIL = "email"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    AD_COPY = "ad_copy"
    LANDING_PAGE = "landing_page"
    NEWSLETTER = "newsletter"
    PRESS_RELEASE = "press_release"
    PRODUCT_DESCRIPTION = "product_description"
    GENERIC = "generic"

@dataclass
class DetectionResult:
    """Result of content analysis"""
    is_malicious: bool
    threat_level: ThreatLevel
    confidence: float
    attack_types: List[AttackType]
    flagged_patterns: List[str]
    processing_time: float
    recommendation: str
    pii_detected: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_type: ContentType = ContentType.GENERIC
    compliance_score: float = 0.0
    marketing_score: float = 0.0
    suggestions: List[str] = field(default_factory=list)

class AdShieldCore:
    """
    Core marketing content security analyzer
    Enhanced version of GuardPrompt specifically for marketing content
    """
    
    def __init__(self, 
                 model_name: str = "unitary/toxic-bert",
                 confidence_threshold: float = 0.75,
                 enable_caching: bool = True,
                 max_cache_size: int = 1000):
        """
        Initialize AdShield with comprehensive marketing content analysis
        
        Args:
            model_name: Primary toxicity detection model
            confidence_threshold: Minimum confidence for threat detection
            enable_caching: Enable result caching for performance
            max_cache_size: Maximum cache entries
        """
        self.confidence_threshold = confidence_threshold
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self.max_cache_size = max_cache_size
        
        # Rate limiting
        self.rate_limiter = defaultdict(list)
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_minute = 100
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize models
        self._initialize_models(model_name)
        
        # Enhanced patterns for marketing content
        self.malicious_patterns = self._load_marketing_patterns()
        
        # PII patterns
        self.pii_patterns = self._load_pii_patterns()
        
        # Marketing-specific patterns
        self.marketing_patterns = self._load_marketing_specific_patterns()
        
        logger.info("AdShield initialized successfully")
    
    def _initialize_models(self, model_name: str):
        """Initialize Hugging Face models"""
        try:
            # Primary toxicity classifier
            self.toxicity_classifier = pipeline(
                "text-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Offensive content detector
            self.offensive_classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-offensive",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Injection detection using MNLI
            self.injection_classifier = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Hate speech detector
            self.hate_speech_detector = pipeline(
                "text-classification",
                model="unitary/unbiased-toxic-roberta",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Bias detection using emotion classifier
            self.bias_detector = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            # Fallback to basic analysis if models fail
            self.toxicity_classifier = None
            self.offensive_classifier = None
            self.injection_classifier = None
            self.sentiment_analyzer = None
            self.hate_speech_detector = None
            self.bias_detector = None
    
    def _load_marketing_patterns(self) -> List[Dict]:
        """Load comprehensive marketing-specific malicious patterns"""
        return [
            # Original GuardPrompt patterns
            {
                "pattern": r"(?i)(ignore|forget|disregard).*(previous|above|earlier).*(instruction|prompt|rule)",
                "type": AttackType.PROMPT_INJECTION,
                "severity": ThreatLevel.HIGH,
                "description": "Instruction override attempt"
            },
            {
                "pattern": r"(?i)(jailbreak|bypass|hack|exploit).*(system|ai|model|chatbot)",
                "type": AttackType.JAILBREAK,
                "severity": ThreatLevel.CRITICAL,
                "description": "System bypass attempt"
            },
            {
                "pattern": r"(?i)(act as|pretend to be|role.?play).*(hacker|criminal|malicious|evil)",
                "type": AttackType.ROLE_PLAY_ATTACK,
                "severity": ThreatLevel.HIGH,
                "description": "Malicious role-play request"
            },
            {
                "pattern": r"(?i)(show|reveal|display|print).*(system prompt|initial prompt|instructions)",
                "type": AttackType.SYSTEM_PROMPT_LEAK,
                "severity": ThreatLevel.MEDIUM,
                "description": "System prompt extraction attempt"
            },
            {
                "pattern": r"(?i)<script|javascript:|data:text/html|eval\(|exec\(",
                "type": AttackType.MALICIOUS_CODE,
                "severity": ThreatLevel.CRITICAL,
                "description": "Code injection attempt"
            },
            {
                "pattern": r"(?i)(password|credit card|social security|bank account|api key|token)",
                "type": AttackType.DATA_EXTRACTION,
                "severity": ThreatLevel.HIGH,
                "description": "Sensitive data extraction attempt"
            },
            {
                "pattern": r"(?i)(bomb|kill|murder|suicide|terrorist|violence)",
                "type": AttackType.ABUSE_CONTENT,
                "severity": ThreatLevel.CRITICAL,
                "description": "Violent or harmful content"
            },
            
            # Marketing-specific enhanced patterns
            {
                "pattern": r"(?i)(fake|false|misleading|deceptive).*(review|testimonial|rating|endorsement)",
                "type": AttackType.FAKE_TESTIMONIALS,
                "severity": ThreatLevel.HIGH,
                "description": "Fake review or testimonial generation"
            },
            {
                "pattern": r"(?i)(spam|bulk|mass).*(email|message|sms|text|dm)",
                "type": AttackType.SPAM_CONTENT,
                "severity": ThreatLevel.MEDIUM,
                "description": "Spam content generation"
            },
            {
                "pattern": r"(?i)(collect|harvest|scrape|steal).*(email|phone|contact|personal|data)",
                "type": AttackType.PRIVACY_VIOLATION,
                "severity": ThreatLevel.HIGH,
                "description": "Personal data collection attempt"
            },
            {
                "pattern": r"(?i)(impersonate|pretend to be|pose as).*(brand|company|business|competitor)",
                "type": AttackType.BRAND_IMPERSONATION,
                "severity": ThreatLevel.HIGH,
                "description": "Brand impersonation attempt"
            },
            {
                "pattern": r"(?i)(guaranteed|100%|instant|miracle|secret|exclusive).*(weight loss|income|profit|cure)",
                "type": AttackType.FALSE_ADVERTISING,
                "severity": ThreatLevel.HIGH,
                "description": "Potentially false advertising claims"
            },
            {
                "pattern": r"(?i)(phishing|scam|fraud|cheat).*(email|website|landing|page)",
                "type": AttackType.SOCIAL_ENGINEERING,
                "severity": ThreatLevel.CRITICAL,
                "description": "Phishing or scam content"
            },
            
            # GDPR and Privacy Violations
            {
                "pattern": r"(?i)(store|collect|save|keep).*(data|information|info).*(without|no).*(consent|asking|permission)",
                "type": AttackType.GDPR_VIOLATION,
                "severity": ThreatLevel.HIGH,
                "description": "Data collection without consent"
            },
            {
                "pattern": r"(?i)(bypass|skip|ignore|avoid).*(gdpr|privacy|consent|data protection|user rights)",
                "type": AttackType.GDPR_VIOLATION,
                "severity": ThreatLevel.HIGH,
                "description": "GDPR compliance bypass attempt"
            },
            {
                "pattern": r"(?i)(no|without).*(opt.?out|unsubscribe|consent|permission).*(required|needed|necessary)",
                "type": AttackType.GDPR_VIOLATION,
                "severity": ThreatLevel.HIGH,
                "description": "GDPR violation - no opt-out mechanism"
            },
            
            # Deceptive Marketing Claims
            {
                "pattern": r"(?i)(only|last|final).*(1|one|few).*(left|remaining|seat|spot|item|chance)",
                "type": AttackType.DECEPTIVE_CLAIM,
                "severity": ThreatLevel.MEDIUM,
                "description": "False scarcity claim"
            },
            {
                "pattern": r"(?i)(limited time|expires|hurry|urgent|act now|don't miss).*(offer|deal|discount|sale)",
                "type": AttackType.DECEPTIVE_CLAIM,
                "severity": ThreatLevel.MEDIUM,
                "description": "False urgency claim"
            },
            {
                "pattern": r"(?i)(free|no cost|no charge).*(trial|sample|gift).*(credit card|payment|billing).*(required|needed)",
                "type": AttackType.DECEPTIVE_CLAIM,
                "severity": ThreatLevel.MEDIUM,
                "description": "Misleading free offer"
            },
            
            # Clickbait patterns
            {
                "pattern": r"(?i)(you won't believe|shocking|amazing|incredible|this will blow your mind)",
                "type": AttackType.CLICKBAIT,
                "severity": ThreatLevel.LOW,
                "description": "Clickbait headline detected"
            },
            {
                "pattern": r"(?i)(number \d+|reason \d+).*(will|that).*(shock|surprise|amaze|blow your mind)",
                "type": AttackType.CLICKBAIT,
                "severity": ThreatLevel.LOW,
                "description": "Numbered clickbait format"
            },
            
            # Price manipulation
            {
                "pattern": r"(?i)(was|originally|normally).*(£|\$|€|₹)?\d+.*now.*(£|\$|€|₹)?\d+",
                "type": AttackType.PRICE_MANIPULATION,
                "severity": ThreatLevel.MEDIUM,
                "description": "Price comparison without verification"
            },
            
            # Unethical bias patterns
            {
                "pattern": r"(?i)(only|exclusively).*(for|available to).*(men|women|male|female|boys|girls)",
                "type": AttackType.UNETHICAL_BIAS,
                "severity": ThreatLevel.HIGH,
                "description": "Gender-based discrimination"
            },
            {
                "pattern": r"(?i)(only|exclusively).*(for|available to).*(white|black|asian|hispanic|latino|indian)",
                "type": AttackType.UNETHICAL_BIAS,
                "severity": ThreatLevel.CRITICAL,
                "description": "Race-based discrimination"
            },
            
            # Brand violations
            {
                "pattern": r"(?i)(click|act|buy).*(now|immediately|today).*(or|otherwise).*(banned|blocked|suspended|lose|miss)",
                "type": AttackType.BRAND_VIOLATION,
                "severity": ThreatLevel.HIGH,
                "description": "Coercive call-to-action"
            },
            
            # Compliance violations
            {
                "pattern": r"(?i)(no|without).*(terms|conditions|privacy|policy|disclaimer)",
                "type": AttackType.COMPLIANCE_VIOLATION,
                "severity": ThreatLevel.MEDIUM,
                "description": "Missing required legal disclaimers"
            }
        ]
    
    def _load_marketing_specific_patterns(self) -> Dict[str, List[Dict]]:
        """Load marketing content type specific patterns"""
        return {
            "email": [
                {
                    "pattern": r"(?i)(unsubscribe|opt.?out).*link.*broken",
                    "type": AttackType.COMPLIANCE_VIOLATION,
                    "severity": ThreatLevel.HIGH,
                    "description": "Broken unsubscribe mechanism"
                },
                {
                    "pattern": r"(?i)(re:|fwd:|urgent:|important:).*(!{3,})",
                    "type": AttackType.SPAM_CONTENT,
                    "severity": ThreatLevel.MEDIUM,
                    "description": "Spam-like email formatting"
                }
            ],
            "social_media": [
                {
                    "pattern": r"(?i)(#ad|#sponsored|#paid).*missing",
                    "type": AttackType.COMPLIANCE_VIOLATION,
                    "severity": ThreatLevel.HIGH,
                    "description": "Missing sponsored content disclosure"
                },
                {
                    "pattern": r"(?i)(follow|like|share).*(for|to).*(win|get|receive)",
                    "type": AttackType.DECEPTIVE_MARKETING,
                    "severity": ThreatLevel.LOW,
                    "description": "Engagement bait tactics"
                }
            ],
            "blog_post": [
                {
                    "pattern": r"(?i)(affiliate|partner).*(link|commission).*disclosure.*missing",
                    "type": AttackType.COMPLIANCE_VIOLATION,
                    "severity": ThreatLevel.HIGH,
                    "description": "Missing affiliate disclosure"
                }
            ]
        }
    
    def _load_pii_patterns(self) -> Dict[str, List[str]]:
        """Load PII detection patterns"""
        return {
            "email": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ],
            "phone": [
                r"\b\d{3}-\d{3}-\d{4}\b",
                r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",
                r"\b\d{10}\b",
                r"\+\d{1,3}\s*\d{3,4}\s*\d{3,4}\s*\d{3,4}\b"
            ],
            "ssn": [
                r"\b\d{3}-\d{2}-\d{4}\b",
                r"\b\d{9}\b"
            ],
            "credit_card": [
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                r"\b\d{13,19}\b"
            ],
            "ip_address": [
                r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
            ],
            "address": [
                r"\b\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl)\b",
                r"\b\d{5}(?:-\d{4})?\b"
            ]
        }
    
    def _detect_content_type(self, content: str) -> ContentType:
        """Detect the type of marketing content"""
        content_lower = content.lower()
        
        # Email indicators
        if any(keyword in content_lower for keyword in ["subject:", "dear", "unsubscribe", "from:", "to:"]):
            return ContentType.EMAIL
        
        # Social media indicators
        if any(keyword in content_lower for keyword in ["#", "@", "follow", "like", "share", "retweet"]):
            return ContentType.SOCIAL_MEDIA
        
        # Blog indicators
        if any(keyword in content_lower for keyword in ["blog", "article", "read more", "continue reading"]):
            return ContentType.BLOG_POST
        
        # Ad copy indicators
        if any(keyword in content_lower for keyword in ["buy now", "click here", "order today", "limited time"]):
            return ContentType.AD_COPY
        
        # Newsletter indicators
        if any(keyword in content_lower for keyword in ["newsletter", "weekly", "monthly", "digest"]):
            return ContentType.NEWSLETTER
        
        # Landing page indicators
        if any(keyword in content_lower for keyword in ["sign up", "get started", "free trial", "download"]):
            return ContentType.LANDING_PAGE
        
        return ContentType.GENERIC
    
    def _get_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_rate_limit(self, identifier: str = "default") -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        # Clean old entries
        self.rate_limiter[identifier] = [
            t for t in self.rate_limiter[identifier] 
            if current_time - t < self.rate_limit_window
        ]
        
        # Check limit
        if len(self.rate_limiter[identifier]) >= self.max_requests_per_minute:
            return False
        
        # Add current request
        self.rate_limiter[identifier].append(current_time)
        return True
    
    def _detect_pii(self, content: str) -> Dict[str, List[str]]:
        """Detect PII in content"""
        detected_pii = {}
        
        for pii_type, patterns in self.pii_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, content, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                detected_pii[pii_type] = matches
        
        return detected_pii
    
    def _pattern_analysis(self, content: str, content_type: ContentType) -> Tuple[List[AttackType], List[str], float]:
        """Analyze content against malicious patterns"""
        detected_types = []
        flagged_patterns = []
        max_severity_score = 0.0
        
        # General patterns
        for pattern_info in self.malicious_patterns:
            if re.search(pattern_info["pattern"], content):
                detected_types.append(pattern_info["type"])
                flagged_patterns.append(pattern_info["description"])
                
                # Convert severity to numeric score
                severity_scores = {
                    ThreatLevel.LOW: 0.3,
                    ThreatLevel.MEDIUM: 0.5,
                    ThreatLevel.HIGH: 0.8,
                    ThreatLevel.CRITICAL: 1.0
                }
                max_severity_score = max(max_severity_score, 
                                       severity_scores.get(pattern_info["severity"], 0.0))
        
        # Content-type specific patterns
        content_type_key = content_type.value
        if content_type_key in self.marketing_patterns:
            for pattern_info in self.marketing_patterns[content_type_key]:
                if re.search(pattern_info["pattern"], content):
                    detected_types.append(pattern_info["type"])
                    flagged_patterns.append(pattern_info["description"])
                    
                    severity_scores = {
                        ThreatLevel.LOW: 0.3,
                        ThreatLevel.MEDIUM: 0.5,
                        ThreatLevel.HIGH: 0.8,
                        ThreatLevel.CRITICAL: 1.0
                    }
                    max_severity_score = max(max_severity_score, 
                                           severity_scores.get(pattern_info["severity"], 0.0))
        
        return detected_types, flagged_patterns, max_severity_score
    
    def _ml_analysis(self, content: str) -> Tuple[float, List[AttackType]]:
        """Enhanced ML analysis for marketing content"""
        try:
            combined_score = 0.0
            attack_types = []
            
            # Only proceed if models are available
            if not self.toxicity_classifier:
                return 0.0, []
            
            # Primary toxicity analysis
            toxicity_results = self.toxicity_classifier(content)
            toxicity_score = 0.0
            if toxicity_results and len(toxicity_results) > 0:
                toxicity_score = max([r.get('score', 0.0) for r in toxicity_results[0] 
                                    if r.get('label') in ['TOXIC', 'HARMFUL', 'NEGATIVE']], default=0.0)
            
            # Offensive content detection
            offensive_score = 0.0
            if self.offensive_classifier:
                offensive_results = self.offensive_classifier(content)
                if offensive_results and len(offensive_results) > 0:
                    offensive_score = max([r.get('score', 0.0) for r in offensive_results[0] 
                                         if r.get('label') in ['OFFENSIVE', 'HATEFUL', 'ABUSIVE']], default=0.0)
            
            # Injection detection using MNLI
            injection_score = 0.0
            if self.injection_classifier:
                injection_hypothesis = "This text contains prompt injection or attempts to override instructions"
                injection_results = self.injection_classifier(content, injection_hypothesis)
                if injection_results:
                    injection_score = max([r.get('score', 0.0) for r in injection_results 
                                         if r.get('label') == 'CONTRADICTION'], default=0.0)
            
            # Hate speech detection
            hate_score = 0.0
            if self.hate_speech_detector:
                hate_results = self.hate_speech_detector(content)
                if hate_results and len(hate_results) > 0:
                    hate_score = max([r.get('score', 0.0) for r in hate_results[0] 
                                    if r.get('label') in ['TOXIC', 'SEVERE_TOXIC', 'HATEFUL']], default=0.0)
            
            # Sentiment analysis
            negative_score = 0.0
            if self.sentiment_analyzer:
                sentiment_results = self.sentiment_analyzer(content)
                if sentiment_results and len(sentiment_results) > 0:
                    negative_score = max([r.get('score', 0.0) for r in sentiment_results[0] 
                                        if r.get('label') == 'NEGATIVE'], default=0.0)
            
            # Bias detection using emotion analysis
            anger_score = 0.0
            if self.bias_detector:
                bias_results = self.bias_detector(content)
                if bias_results and len(bias_results) > 0:
                    anger_score = max([r.get('score', 0.0) for r in bias_results[0] 
                                     if r.get('label') == 'anger'], default=0.0)
            
            # Combined scoring
            combined_score = (
                toxicity_score * 0.25 +
                offensive_score * 0.25 +
                injection_score * 0.25 +
                hate_score * 0.15 +
                negative_score * 0.05 +
                anger_score * 0.05
            )
            
            # Determine attack types based on scores
            if injection_score > 0.7:
                attack_types.append(AttackType.PROMPT_INJECTION)
            if toxicity_score > 0.8:
                attack_types.append(AttackType.ABUSE_CONTENT)
            if offensive_score > 0.7:
                attack_types.append(AttackType.DECEPTIVE_MARKETING)
            if hate_score > 0.8:
                attack_types.append(AttackType.ABUSE_CONTENT)
            if negative_score > 0.9:
                attack_types.append(AttackType.SOCIAL_ENGINEERING)
            if anger_score > 0.8:
                attack_types.append(AttackType.UNETHICAL_BIAS)
            
            return combined_score, attack_types
            
        except Exception as e:
            logger.error(f"ML analysis failed: {e}")
            return 0.0, []
    
    def _calculate_compliance_score(self, content: str, content_type: ContentType) -> float:
        """Calculate compliance score based on content type"""
        score = 100.0
        
        # Content-specific compliance checks
        if content_type == ContentType.EMAIL:
            if "unsubscribe" not in content.lower():
                score -= 20
            if re.search(r"from:.*@.*\.", content, re.IGNORECASE) is None:
                score -= 10
        
        elif content_type == ContentType.SOCIAL_MEDIA:
            if re.search(r"#(ad|sponsored|paid)", content, re.IGNORECASE) is None:
                # Check if it's promotional content
                if any(word in content.lower() for word in ["buy", "purchase", "sale", "discount"]):
                    score -= 30
        
        elif content_type == ContentType.AD_COPY:
            if "terms and conditions" not in content.lower():
                score -= 15
        
        # General compliance checks
        if re.search(r"guaranteed.*\d+%", content, re.IGNORECASE):
            score -= 25  # Unsubstantiated guarantee claims
        
        return max(0.0, score)
    
    def _calculate_marketing_score(self, content: str, content_type: ContentType) -> float:
        """Calculate marketing effectiveness score"""
        score = 50.0  # Base score
        
        # Positive indicators
        if re.search(r"(call.to.action|cta|click|buy|purchase)", content, re.IGNORECASE):
            score += 10
        
        if re.search(r"(benefit|advantage|value|solution)", content, re.IGNORECASE):
            score += 15
        
        # Negative indicators
        if re.search(r"(spam|junk|fake|scam)", content, re.IGNORECASE):
            score -= 30
        
        if re.search(r"(urgent|hurry|limited time|act now)", content, re.IGNORECASE):
            score -= 10  # Aggressive tactics
        
        return max(0.0, min(100.0, score))
    
    def _generate_suggestions(self, content: str, content_type: ContentType, 
                             attack_types: List[AttackType]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        # Content-type specific suggestions
        if content_type == ContentType.EMAIL:
            if "unsubscribe" not in content.lower():
                suggestions.append("Add clear unsubscribe link for compliance")
            if re.search(r"urgent|important", content, re.IGNORECASE):
                suggestions.append("Consider reducing urgency language to avoid spam filters")
            if not re.search(r"from:.*@.*\.", content, re.IGNORECASE):
                suggestions.append("Include proper sender identification")
        
        elif content_type == ContentType.SOCIAL_MEDIA:
            if any(word in content.lower() for word in ["buy", "purchase", "sale"]):
                if not re.search(r"#(ad|sponsored|paid)", content, re.IGNORECASE):
                    suggestions.append("Add #ad or #sponsored disclosure for promotional content")
            if content.count('#') > 10:
                suggestions.append("Reduce number of hashtags to avoid appearing spammy")
        
        elif content_type == ContentType.AD_COPY:
            if "terms and conditions" not in content.lower():
                suggestions.append("Include reference to terms and conditions")
            if re.search(r"guaranteed.*\d+%", content, re.IGNORECASE):
                suggestions.append("Provide substantiation for percentage claims")
        
        # Attack-type specific suggestions
        if AttackType.FAKE_TESTIMONIALS in attack_types:
            suggestions.append("Ensure all testimonials are from real customers with proper disclosure")
        
        if AttackType.FALSE_ADVERTISING in attack_types:
            suggestions.append("Verify all claims are substantiated and compliant with advertising standards")
        
        if AttackType.CLICKBAIT in attack_types:
            suggestions.append("Use more descriptive headlines that accurately represent content")
        
        if AttackType.DECEPTIVE_CLAIM in attack_types:
            suggestions.append("Remove or modify misleading urgency/scarcity claims")
        
        if AttackType.GDPR_VIOLATION in attack_types:
            suggestions.append("Ensure GDPR compliance with proper consent mechanisms")
        
        if AttackType.UNETHICAL_BIAS in attack_types:
            suggestions.append("Review content for discriminatory language or bias")
        
        return suggestions
    
    def _determine_threat_level(self, pattern_score: float, ml_score: float, 
                               attack_types: List[AttackType]) -> ThreatLevel:
        """Determine overall threat level"""
        max_score = max(pattern_score, ml_score)
        
        # Critical threats
        if any(at in attack_types for at in [
            AttackType.JAILBREAK, AttackType.MALICIOUS_CODE, 
            AttackType.SOCIAL_ENGINEERING, AttackType.UNETHICAL_BIAS
        ]):
            return ThreatLevel.CRITICAL
        
        if max_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif max_score >= 0.7:
            return ThreatLevel.HIGH
        elif max_score >= 0.5:
            return ThreatLevel.MEDIUM
        elif max_score >= 0.3:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.SAFE
    
    def _generate_recommendation(self, threat_level: ThreatLevel, 
                                attack_types: List[AttackType]) -> str:
        """Generate actionable recommendation"""
        if threat_level == ThreatLevel.CRITICAL:
            return "🚨 CRITICAL: Content blocked. Immediate review required before publication."
        elif threat_level == ThreatLevel.HIGH:
            return "⚠️ HIGH RISK: Content requires significant modifications before approval."
        elif threat_level == ThreatLevel.MEDIUM:
            return "🔶 MEDIUM RISK: Review and modify flagged sections before publication."
        elif threat_level == ThreatLevel.LOW:
            return "🔶 LOW RISK: Minor adjustments recommended for better compliance."
        else:
            return "✅ SAFE: Content approved for publication."
    
    def analyze_content(self, content: str, content_type: ContentType = None,
                       client_id: str = "default") -> DetectionResult:
        """
        Comprehensive content analysis for marketing materials
        
        Args:
            content: Text content to analyze
            content_type: Type of marketing content (auto-detected if None)
            client_id: Client identifier for rate limiting
            
        Returns:
            DetectionResult with comprehensive analysis
        """
        start_time = time.time()
        
        # Rate limiting check
        if not self._check_rate_limit(client_id):
            return DetectionResult(
                is_malicious=False,
                threat_level=ThreatLevel.SAFE,
                confidence=0.0,
                attack_types=[],
                flagged_patterns=["Rate limit exceeded"],
                processing_time=time.time() - start_time,
                recommendation="Rate limit exceeded. Please try again later."
            )
        
        # Input validation
        if not content or not content.strip():
            return DetectionResult(
                is_malicious=False,
                threat_level=ThreatLevel.SAFE,
                confidence=0.0,
                attack_types=[],
                flagged_patterns=["Empty content"],
                processing_time=time.time() - start_time,
                recommendation="No content provided for analysis."
            )
        
        # Content length check
        if len(content) > 50000:  # 50KB limit
            content = content[:50000]
            logger.warning("Content truncated to 50KB limit")
        
        # Check cache
        cache_key = self._get_cache_key(content) if self.enable_caching else None
        if cache_key and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        # Auto-detect content type
        if content_type is None:
            content_type = self._detect_content_type(content)
        
        # Thread-safe analysis
        with self.lock:
            # Pattern-based analysis
            pattern_attacks, flagged_patterns, pattern_score = self._pattern_analysis(content, content_type)
            
            # ML-based analysis
            ml_score, ml_attacks = self._ml_analysis(content)
            
            # PII detection
            pii_detected = self._detect_pii(content)
            
            # Combine attack types
            all_attacks = list(set(pattern_attacks + ml_attacks))
            
            # Calculate scores
            compliance_score = self._calculate_compliance_score(content, content_type)
            marketing_score = self._calculate_marketing_score(content, content_type)
            
            # Determine threat level
            threat_level = self._determine_threat_level(pattern_score, ml_score, all_attacks)
            
            # Final confidence score
            confidence = max(pattern_score, ml_score)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(content, content_type, all_attacks)
            
            # Create result
            result = DetectionResult(
                is_malicious=threat_level not in [ThreatLevel.SAFE, ThreatLevel.LOW],
                threat_level=threat_level,
                confidence=confidence,
                attack_types=all_attacks,
                flagged_patterns=flagged_patterns,
                processing_time=time.time() - start_time,
                recommendation=self._generate_recommendation(threat_level, all_attacks),
                pii_detected=pii_detected,
                metadata={
                    "content_length": len(content),
                    "pattern_score": pattern_score,
                    "ml_score": ml_score,
                    "client_id": client_id,
                    "timestamp": datetime.now().isoformat()
                },
                content_type=content_type,
                compliance_score=compliance_score,
                marketing_score=marketing_score,
                suggestions=suggestions
            )
            
            # Cache result
            if self.enable_caching and cache_key:
                if len(self.cache) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[cache_key] = result
            
            return result
    
    def analyze_batch(self, contents: List[str], content_types: List[ContentType] = None,
                     client_id: str = "default") -> List[DetectionResult]:
        """
        Analyze multiple content pieces in batch
        
        Args:
            contents: List of content strings to analyze
            content_types: List of content types (auto-detected if None)
            client_id: Client identifier for rate limiting
            
        Returns:
            List of DetectionResults
        """
        if content_types is None:
            content_types = [None] * len(contents)
        
        results = []
        for content, content_type in zip(contents, content_types):
            result = self.analyze_content(content, content_type, client_id)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            "cache_size": len(self.cache) if self.cache else 0,
            "rate_limiter_entries": len(self.rate_limiter),
            "model_status": {
                "toxicity_classifier": self.toxicity_classifier is not None,
                "offensive_classifier": self.offensive_classifier is not None,
                "injection_classifier": self.injection_classifier is not None,
                "sentiment_analyzer": self.sentiment_analyzer is not None,
                "hate_speech_detector": self.hate_speech_detector is not None,
                "bias_detector": self.bias_detector is not None
            }
        }
    
    def clear_cache(self):
        """Clear analysis cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def update_patterns(self, new_patterns: List[Dict]):
        """Update malicious patterns"""
        self.malicious_patterns.extend(new_patterns)
        logger.info(f"Added {len(new_patterns)} new patterns")
