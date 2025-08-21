#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI application for Report Generation Service.

This module provides REST API endpoints and WebSocket support for the 
template-based report generation service with real-time streaming capabilities.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .models import (
    ReportGenerationRequest, 
    ReportGenerationResponse,
    StreamMessage,
    JobProgress
)
from .report_service import get_report_service
from ..logger import logger


# Create FastAPI application
app = FastAPI(
    title="Template Report Generation Service",
    description="A service for generating reports from markdown templates with streaming support",
    version="1.0.0"
)

# Add CORS middleware for frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend dashboard
from pathlib import Path
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Get service instance
report_service = get_report_service()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_jobs: Dict[str, str] = {}  # connection_id -> job_id
    
    async def connect(self, websocket: WebSocket, connection_id: str, job_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        if job_id:
            self.connection_jobs[connection_id] = job_id
        
        # Register with report service
        await report_service.register_connection(connection_id, job_id)
        logger.info(f"WebSocket connected: {connection_id} for job {job_id}")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_jobs:
            del self.connection_jobs[connection_id]
        
        # Unregister from report service
        asyncio.create_task(report_service.unregister_connection(connection_id))
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_to_job(self, job_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections for a specific job."""
        connections_to_remove = []
        for connection_id, conn_job_id in self.connection_jobs.items():
            if conn_job_id == job_id:
                try:
                    await self.send_message(connection_id, message)
                except:
                    connections_to_remove.append(connection_id)
        
        # Clean up failed connections
        for connection_id in connections_to_remove:
            self.disconnect(connection_id)

# Global connection manager
manager = ConnectionManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    await report_service.start()
    logger.info("Report Generation Service API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup service on shutdown."""
    await report_service.stop()
    logger.info("Report Generation Service API stopped")


# REST API Endpoints
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    """Redirect to dashboard."""
    return RedirectResponse(url="/static/dashboard.html")

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API endpoint with service information."""
    return {
        "service": "Template Report Generation Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/jobs", response_model=ReportGenerationResponse)
async def create_job(request: ReportGenerationRequest):
    """
    Create a new report generation job.
    
    Args:
        request: Report generation parameters
        
    Returns:
        ReportGenerationResponse with job details
    """
    try:
        response = await report_service.create_report_job(request)
        return response
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs/{job_id}/start", response_model=ReportGenerationResponse)
async def start_job(job_id: str):
    """
    Start execution of a created job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        ReportGenerationResponse with execution status
    """
    try:
        response = await report_service.start_job(job_id)
        
        if response.status == "not_found":
            raise HTTPException(status_code=404, detail="Job not found")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}/status", response_model=ReportGenerationResponse)
async def get_job_status(job_id: str):
    """
    Get current status of a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        ReportGenerationResponse with current status
    """
    try:
        response = await report_service.get_job_status(job_id)
        
        if response is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}/progress", response_model=Optional[JobProgress])
async def get_job_progress(job_id: str):
    """
    Get detailed progress information for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        JobProgress with detailed task information
    """
    try:
        progress = await report_service._get_job_progress(job_id)
        
        if progress is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job progress {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}/tasks/{task_id}")
async def get_task_details(job_id: str, task_id: str):
    """
    Get detailed information for a specific task.
    
    Args:
        job_id: Job identifier
        task_id: Task identifier
        
    Returns:
        Dictionary with detailed task information including content and metadata
    """
    try:
        # Check if job exists
        if job_id not in report_service.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = report_service.jobs[job_id]
        
        # Check if task exists in job
        if task_id not in job.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = job.tasks[task_id]
        
        # Extract agent messages from metadata for easier frontend access
        agent_messages = []
        if task.metadata and 'memory' in task.metadata:
            memory = task.metadata['memory']
            if isinstance(memory, list):
                agent_messages = memory
        
        # Return detailed task information
        task_details = {
            "task_id": task.task_id,
            "job_id": task.job_id,
            "title": task.title,
            "task_type": task.task_type,
            "status": task.status.value if hasattr(task.status, 'value') else task.status,
            "progress": task.progress,
            "level": task.level,
            "content": task.content or "",
            "content_length": len(task.content) if task.content else 0,
            "dependencies": task.dependencies,
            "metadata": task.metadata or {},
            "agent_messages": agent_messages,  # Extracted for convenience
            "error_message": task.error_message,
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None
        }
        
        return task_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task details {job_id}/{task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs/{job_id}/cancel", response_model=ReportGenerationResponse)
async def cancel_job(job_id: str):
    """
    Cancel a running job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        ReportGenerationResponse with cancellation status
    """
    try:
        response = await report_service.cancel_job(job_id)
        
        if response.status == "not_found":
            raise HTTPException(status_code=404, detail="Job not found")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}/report")
async def download_report(job_id: str):
    """
    Download the generated report file.
    
    Args:
        job_id: Job identifier
        
    Returns:
        FileResponse with the report file
    """
    try:
        job_status = await report_service.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if not job_status.report_path:
            raise HTTPException(status_code=404, detail="Report not yet generated")
        
        report_path = Path(job_status.report_path)
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")
        
        return FileResponse(
            path=str(report_path),
            filename=report_path.name,
            media_type="text/markdown"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_model=List[ReportGenerationResponse])
async def list_jobs():
    """
    List all active jobs.
    
    Returns:
        List of ReportGenerationResponse objects
    """
    try:
        jobs = []
        for job_id in report_service.jobs.keys():
            job_status = await report_service.get_job_status(job_id)
            if job_status:
                jobs.append(job_status)
        
        return jobs
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-template")
async def upload_template(file: UploadFile = File(...)):
    """
    Upload a markdown template file.
    
    Args:
        file: Uploaded template file
        
    Returns:
        Dictionary with upload status and file path
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.md', '.markdown')):
            raise HTTPException(status_code=400, detail="Only markdown files are supported")
        
        # Create upload directory
        upload_dir = Path("workdir/templates")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{file_id}_{file.filename}"
        file_path = upload_dir / filename
        
        # Save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Template uploaded: {file_path}")
        
        return {
            "status": "success",
            "message": "Template uploaded successfully",
            "file_path": str(file_path),
            "original_filename": file.filename,
            "file_size": len(content)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload knowledge base documents.
    
    Args:
        files: List of uploaded document files
        
    Returns:
        Dictionary with upload status
    """
    try:
        # Create documents directory
        docs_dir = Path("workdir/documents")
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        total_size = 0
        
        for file in files:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{file_id}_{file.filename}"
            file_path = docs_dir / filename
            
            # Save file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            uploaded_files.append({
                "original_filename": file.filename,
                "saved_path": str(file_path),
                "file_size": len(content)
            })
            total_size += len(content)
        
        logger.info(f"Uploaded {len(uploaded_files)} documents, total size: {total_size} bytes")
        
        return {
            "status": "success",
            "message": f"Uploaded {len(uploaded_files)} documents successfully",
            "files": uploaded_files,
            "total_size": total_size
        }
    except Exception as e:
        logger.error(f"Failed to upload documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job updates.
    
    Args:
        websocket: WebSocket connection
        job_id: Job ID to subscribe to
    """
    connection_id = str(uuid.uuid4())
    
    try:
        await manager.connect(websocket, connection_id, job_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "connection_id": connection_id,
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get("type") == "get_status":
                    # Send current job status
                    job_status = await report_service.get_job_status(job_id)
                    if job_status:
                        await websocket.send_text(json.dumps({
                            "type": "status_update",
                            "job_status": job_status.dict(),
                            "timestamp": datetime.now().isoformat()
                        }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket {connection_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error for {connection_id}: {e}")
    finally:
        manager.disconnect(connection_id)


# Modify the report service to use the connection manager
original_broadcast_message_to_job = report_service._broadcast_message_to_job

async def enhanced_broadcast_message_to_job(job_id: str, message: StreamMessage) -> None:
    """Enhanced broadcast that uses WebSocket connections."""
    try:
        # Convert StreamMessage to dict for JSON serialization  
        message_dict = {
            "type": "stream_message",
            "message": {
                "message_id": message.message_id,
                "job_id": message.job_id,
                "task_id": message.task_id,
                "message_type": message.message_type.value,
                "timestamp": message.timestamp.isoformat(),
                "content": message.content,
                "metadata": message.metadata
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the message for debugging
        logger.info(f"Broadcasting WebSocket message: {message.message_type} for job {job_id}")
        
        # Broadcast via WebSocket
        await manager.broadcast_to_job(job_id, message_dict)
        
        # Also call original method for any other handling
        await original_broadcast_message_to_job(job_id, message)
        
    except Exception as e:
        logger.error(f"Error in enhanced_broadcast_message_to_job: {e}")
        # Fallback to original method
        await original_broadcast_message_to_job(job_id, message)

# Replace the service method
report_service._broadcast_message_to_job = enhanced_broadcast_message_to_job


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Template Report Generation Service",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(report_service.jobs),
        "active_connections": len(manager.active_connections)
    }


# Development server function
def run_dev_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the development server."""
    uvicorn.run(
        "app.service.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_dev_server()