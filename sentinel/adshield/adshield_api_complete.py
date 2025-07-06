"""
AdShield FastAPI Endpoints
API endpoints for integrating AdShield with main.py
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import json
import tempfile
import os
from datetime import datetime
import asyncio
import uuid

# Import the core components
from adshield_core_part1 import AdShieldCore, ThreatLevel, AttackType, ContentType, DetectionResult
from adshield_utils import DocumentProcessor, BatchProcessor

# Create router
router = APIRouter(prefix="/adshield", tags=["AdShield Marketing Security"])

# Global analyzer instance (initialize once)
analyzer = None

# Task storage for async processing
async_tasks = {}

def get_analyzer():
    """Get or initialize the AdShield analyzer"""
    global analyzer
    if analyzer is None:
        analyzer = AdShieldCore()
    return analyzer

# Request/Response Models
class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis"""
    content: str = Field(..., description="Content to analyze", min_length=1)
    content_type: Optional[str] = Field(None, description="Type of content (email, blog_post, social_media, etc.)")
    client_id: Optional[str] = Field("default", description="Client identifier for rate limiting")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    contents: List[str] = Field(..., description="List of content to analyze")
    content_types: Optional[List[str]] = Field(None, description="List of content types")
    client_id: Optional[str] = Field("default", description="Client identifier for rate limiting")

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    is_malicious: bool
    threat_level: str
    confidence: float
    attack_types: List[str]
    flagged_patterns: List[str]
    processing_time: float
    recommendation: str
    pii_detected: Dict[str, List[str]]
    metadata: Dict[str, Any]
    content_type: str
    compliance_score: float
    marketing_score: float
    suggestions: List[str]

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    results: List[AnalysisResponse]
    summary: Dict[str, Any]

class FileAnalysisResponse(BaseModel):
    """Response model for file analysis"""
    filename: str
    file_info: Dict[str, Any]
    analysis: AnalysisResponse

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    timestamp: str

class AsyncTaskResponse(BaseModel):
    """Response model for async task"""
    task_id: str
    status: str
    message: str
    created_at: str

class AsyncTaskStatus(BaseModel):
    """Response model for async task status"""
    task_id: str
    status: str
    progress: float
    result: Optional[AnalysisResponse] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

# Utility functions
def detection_result_to_response(result: DetectionResult) -> AnalysisResponse:
    """Convert DetectionResult to AnalysisResponse"""
    return AnalysisResponse(
        is_malicious=result.is_malicious,
        threat_level=result.threat_level.value,
        confidence=result.confidence,
        attack_types=[at.value for at in result.attack_types],
        flagged_patterns=result.flagged_patterns,
        processing_time=result.processing_time,
        recommendation=result.recommendation,
        pii_detected=result.pii_detected,
        metadata=result.metadata,
        content_type=result.content_type.value,
        compliance_score=result.compliance_score,
        marketing_score=result.marketing_score,
        suggestions=result.suggestions
    )

def parse_content_type(content_type_str: Optional[str]) -> Optional[ContentType]:
    """Parse content type string to ContentType enum"""
    if not content_type_str:
        return None
    
    try:
        return ContentType(content_type_str.lower())
    except ValueError:
        return None

async def process_async_task(task_id: str, content: str, content_type: Optional[ContentType], client_id: str):
    """Process content analysis asynchronously"""
    try:
        # Update task status
        async_tasks[task_id]["status"] = "processing"
        async_tasks[task_id]["progress"] = 0.1
        
        analyzer = get_analyzer()
        
        # Update progress
        async_tasks[task_id]["progress"] = 0.5
        
        # Perform analysis
        result = analyzer.analyze_content(
            content=content,
            content_type=content_type,
            client_id=client_id
        )
        
        # Update task with result
        async_tasks[task_id]["status"] = "completed"
        async_tasks[task_id]["progress"] = 1.0
        async_tasks[task_id]["result"] = detection_result_to_response(result)
        async_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        async_tasks[task_id]["status"] = "failed"
        async_tasks[task_id]["error"] = str(e)
        async_tasks[task_id]["completed_at"] = datetime.now().isoformat()

# API Endpoints

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        analyzer = get_analyzer()
        stats = analyzer.get_statistics()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=any(stats["model_status"].values()),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            models_loaded=False,
            timestamp=datetime.now().isoformat()
        )

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(request: ContentAnalysisRequest):
    """
    Analyze single marketing content
    
    - **content**: The marketing content to analyze
    - **content_type**: Optional content type (email, blog_post, social_media, etc.)
    - **client_id**: Optional client identifier for rate limiting
    """
    try:
        analyzer = get_analyzer()
        content_type = parse_content_type(request.content_type)
        
        result = analyzer.analyze_content(
            content=request.content,
            content_type=content_type,
            client_id=request.client_id
        )
        
        return detection_result_to_response(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Analyze multiple marketing contents in batch
    
    - **contents**: List of marketing contents to analyze
    - **content_types**: Optional list of content types
    - **client_id**: Optional client identifier for rate limiting
    """
    try:
        analyzer = get_analyzer()
        
        # Parse content types
        content_types = None
        if request.content_types:
            content_types = [parse_content_type(ct) for ct in request.content_types]
        
        results = analyzer.analyze_batch(
            contents=request.contents,
            content_types=content_types,
            client_id=request.client_id
        )
        
        # Convert results
        response_results = [detection_result_to_response(result) for result in results]
        
        # Generate summary
        summary = {
            "total_content": len(results),
            "safe_content": len([r for r in results if r.threat_level == ThreatLevel.SAFE]),
            "high_risk_content": len([r for r in results if r.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]),
            "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
            "average_compliance_score": sum(r.compliance_score for r in results) / len(results) if results else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        return BatchAnalysisResponse(
            results=response_results,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/analyze/file", response_model=FileAnalysisResponse)
async def analyze_file(file: UploadFile = File(...), client_id: str = Form("default")):
    """
    Analyze uploaded file content
    
    - **file**: File to analyze (txt, docx, json, csv, html)
    - **client_id**: Optional client identifier for rate limiting
    """
    try:
        # Check file type
        allowed_extensions = ['.txt', '.docx', '.json', '.csv', '.html']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process file
            processor = DocumentProcessor()
            file_info = processor.process_file(temp_file_path)
            
            # Analyze content
            analyzer = get_analyzer()
            result = analyzer.analyze_content(
                content=file_info['content'],
                client_id=client_id
            )
            
            return FileAnalysisResponse(
                filename=file.filename,
                file_info=file_info,
                analysis=detection_result_to_response(result)
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

@router.post("/analyze/files", response_model=List[FileAnalysisResponse])
async def analyze_multiple_files(files: List[UploadFile] = File(...), client_id: str = Form("default")):
    """
    Analyze multiple uploaded files
    
    - **files**: List of files to analyze
    - **client_id**: Optional client identifier for rate limiting
    """
    try:
        if len(files) > 10:  # Limit to 10 files
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
        
        results = []
        processor = DocumentProcessor()
        analyzer = get_analyzer()
        
        for file in files:
            try:
                # Check file type
                allowed_extensions = ['.txt', '.docx', '.json', '.csv', '.html']
                file_extension = os.path.splitext(file.filename)[1].lower()
                
                if file_extension not in allowed_extensions:
                    continue  # Skip unsupported files
                
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    # Process file
                    file_info = processor.process_file(temp_file_path)
                    
                    # Analyze content
                    result = analyzer.analyze_content(
                        content=file_info['content'],
                        client_id=client_id
                    )
                    
                    results.append(FileAnalysisResponse(
                        filename=file.filename,
                        file_info=file_info,
                        analysis=detection_result_to_response(result)
                    ))
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        
            except Exception as e:
                # Continue with other files if one fails
                continue
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multiple file analysis failed: {str(e)}")

@router.post("/analyze/async", response_model=AsyncTaskResponse)
async def analyze_async(request: ContentAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze content asynchronously
    Returns immediately with task ID
    """
    task_id = str(uuid.uuid4())
    
    # Initialize task
    async_tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    # Add background task
    content_type = parse_content_type(request.content_type)
    background_tasks.add_task(
        process_async_task,
        task_id,
        request.content,
        content_type,
        request.client_id
    )
    
    return AsyncTaskResponse(
        task_id=task_id,
        status="pending",
        message="Analysis task queued successfully",
        created_at=async_tasks[task_id]["created_at"]
    )

@router.get("/analyze/async/{task_id}", response_model=AsyncTaskStatus)
async def get_async_task_status(task_id: str):
    """Get the status of an async analysis task"""
    if task_id not in async_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = async_tasks[task_id]
    return AsyncTaskStatus(**task)

@router.delete("/analyze/async/{task_id}")
async def cancel_async_task(task_id: str):
    """Cancel an async analysis task"""
    if task_id not in async_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = async_tasks[task_id]
    if task["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Task already completed")
    
    # Mark as cancelled
    async_tasks[task_id]["status"] = "cancelled"
    async_tasks[task_id]["completed_at"] = datetime.now().isoformat()
    
    return {"message": "Task cancelled successfully"}

@router.get("/analyze/async")
async def list_async_tasks():
    """List all async tasks"""
    return {
        "tasks": list(async_tasks.values()),
        "total": len(async_tasks)
    }

@router.get("/stats")
async def get_analyzer_stats():
    """Get analyzer statistics"""
    try:
        analyzer = get_analyzer()
        stats = analyzer.get_statistics()
        
        return {
            "statistics": stats,
            "async_tasks": {
                "total": len(async_tasks),
                "pending": len([t for t in async_tasks.values() if t["status"] == "pending"]),
                "processing": len([t for t in async_tasks.values() if t["status"] == "processing"]),
                "completed": len([t for t in async_tasks.values() if t["status"] == "completed"]),
                "failed": len([t for t in async_tasks.values() if t["status"] == "failed"])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/clear-cache")
async def clear_cache():
    """Clear analyzer cache"""
    try:
        analyzer = get_analyzer()
        analyzer.clear_cache()
        
        return {
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.delete("/clear-tasks")
async def clear_completed_tasks():
    """Clear completed async tasks"""
    global async_tasks
    
    # Keep only active tasks
    active_tasks = {
        task_id: task for task_id, task in async_tasks.items()
        if task["status"] in ["pending", "processing"]
    }
    
    cleared_count = len(async_tasks) - len(active_tasks)
    async_tasks = active_tasks
    
    return {
        "message": f"Cleared {cleared_count} completed tasks",
        "remaining_tasks": len(async_tasks),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/supported-types")
async def get_supported_types():
    """Get supported content types"""
    return {
        "content_types": [ct.value for ct in ContentType],
        "attack_types": [at.value for at in AttackType],
        "threat_levels": [tl.value for tl in ThreatLevel],
        "file_formats": ['.txt', '.docx', '.json', '.csv', '.html']
    }

@router.post("/validate-content")
async def validate_content(request: ContentAnalysisRequest):
    """
    Quick validation endpoint (lighter analysis)
    Returns basic safety check without full analysis
    """
    try:
        analyzer = get_analyzer()
        content_type = parse_content_type(request.content_type)
        
        # Perform quick pattern-based analysis only
        detected_types, flagged_patterns, pattern_score = analyzer._pattern_analysis(
            request.content, 
            content_type or ContentType.GENERIC
        )
        
        threat_level = ThreatLevel.SAFE
        if pattern_score >= 0.7:
            threat_level = ThreatLevel.HIGH
        elif pattern_score >= 0.5:
            threat_level = ThreatLevel.MEDIUM
        elif pattern_score >= 0.3:
            threat_level = ThreatLevel.LOW
        
        return {
            "is_safe": threat_level in [ThreatLevel.SAFE, ThreatLevel.LOW],
            "threat_level": threat_level.value,
            "confidence": pattern_score,
            "flagged_patterns": flagged_patterns,
            "processing_time": 0.01,  # Much faster
            "recommendation": "Quick validation complete"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    content: str = Form(...),
    analysis_result: str = Form(...),
    is_correct: bool = Form(...),
    feedback_note: Optional[str] = Form(None),
    client_id: str = Form("default")
):
    """
    Submit feedback for analysis results
    Helps improve the model accuracy
    """
    try:
        analyzer = get_analyzer()
        
        # Store feedback for model improvement
        feedback_data = {
            "content_hash": hash(content),
            "analysis_result": json.loads(analysis_result),
            "is_correct": is_correct,
            "feedback_note": feedback_note,
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # In a real implementation, this would be stored in a database
        # For now, we'll just log it
        print(f"Feedback received: {feedback_data}")
        
        return {
            "message": "Feedback submitted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded models"""
    try:
        analyzer = get_analyzer()
        stats = analyzer.get_statistics()
        
        return {
            "model_status": stats["model_status"],
            "version": "1.0.0",
            "supported_languages": ["en", "es", "fr", "de", "it", "pt"],
            "model_types": [
                "pattern_analysis",
                "ml_classification",
                "pii_detection",
                "sentiment_analysis",
                "compliance_check"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Include the router in your main FastAPI app with:
# app.include_router(router)
