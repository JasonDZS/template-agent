#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data models for report generation service.

This module defines all data structures used in the report generation service,
including request/response models, task status tracking, and streaming message formats.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status enumeration."""
    PENDING = "pending"          # Task waiting to be executed
    READY = "ready"             # Task ready for execution (dependencies met)
    RUNNING = "running"         # Task currently executing
    STREAMING = "streaming"     # Task producing streaming output
    COMPLETED = "completed"     # Task completed successfully
    FAILED = "failed"           # Task execution failed
    CANCELLED = "cancelled"     # Task was cancelled


class MessageType(str, Enum):
    """Stream message type enumeration."""
    TASK_START = "task_start"           # Task execution started
    TASK_PROGRESS = "task_progress"     # Task progress update
    TASK_CONTENT = "task_content"       # Task content chunk
    TASK_COMPLETE = "task_complete"     # Task completed
    TASK_ERROR = "task_error"           # Task error occurred
    JOB_PROGRESS = "job_progress"       # Overall job progress
    JOB_COMPLETE = "job_complete"       # Job completed
    JOB_ERROR = "job_error"             # Job error occurred


class ReportGenerationRequest(BaseModel):
    """Request model for report generation."""
    template_path: str = Field(..., description="Path to the markdown template file")
    knowledge_base_path: str = Field(default="workdir/documents", description="Path to knowledge base directory")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Maximum concurrent tasks")
    enable_streaming: bool = Field(default=True, description="Enable streaming output")
    enable_model_merge: bool = Field(default=False, description="Enable intelligent model-based merging")
    report_title: Optional[str] = Field(default=None, description="Custom report title")
    output_path: Optional[str] = Field(default=None, description="Custom output path for final report")
    

class ReportTask(BaseModel):
    """Individual task within a report generation job."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    job_id: str = Field(..., description="Parent job identifier")
    task_type: str = Field(..., description="Task type (generation/merge)")
    title: str = Field(..., description="Task title")
    level: int = Field(..., description="Heading level")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    dependencies: List[str] = Field(default_factory=list, description="Task dependency IDs")
    content: Optional[str] = Field(default=None, description="Generated content")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    start_time: Optional[datetime] = Field(default=None, description="Task start time")
    end_time: Optional[datetime] = Field(default=None, description="Task end time")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Task progress percentage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")


class StreamMessage(BaseModel):
    """Message format for streaming output."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Message identifier")
    job_id: str = Field(..., description="Job identifier")
    task_id: Optional[str] = Field(default=None, description="Task identifier (if task-specific)")
    message_type: MessageType = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    content: Union[str, Dict[str, Any]] = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TaskProgress(BaseModel):
    """Task progress information."""
    task_id: str = Field(..., description="Task identifier")
    title: str = Field(..., description="Task title")
    status: TaskStatus = Field(..., description="Task status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    content_length: int = Field(default=0, description="Generated content length")
    error_message: Optional[str] = Field(default=None, description="Error message if any")


class JobProgress(BaseModel):
    """Overall job progress information."""
    job_id: str = Field(..., description="Job identifier")
    total_tasks: int = Field(..., description="Total number of tasks")
    pending_tasks: int = Field(default=0, description="Number of pending tasks")
    ready_tasks: int = Field(default=0, description="Number of ready tasks")
    running_tasks: int = Field(default=0, description="Number of running tasks")
    completed_tasks: int = Field(default=0, description="Number of completed tasks")
    failed_tasks: int = Field(default=0, description="Number of failed tasks")
    overall_progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall progress percentage")
    start_time: datetime = Field(..., description="Job start time")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    tasks: List[TaskProgress] = Field(default_factory=list, description="Individual task progress")


class ReportGenerationResponse(BaseModel):
    """Response model for report generation."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Response message")
    report_path: Optional[str] = Field(default=None, description="Path to generated report")
    progress: Optional[JobProgress] = Field(default=None, description="Job progress information")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ReportJob(BaseModel):
    """Complete report generation job."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Job identifier")
    request: ReportGenerationRequest = Field(..., description="Original request")
    tasks: Dict[str, ReportTask] = Field(default_factory=dict, description="Job tasks")
    status: str = Field(default="pending", description="Job status")
    start_time: datetime = Field(default_factory=datetime.now, description="Job start time")
    end_time: Optional[datetime] = Field(default=None, description="Job end time")
    final_report_path: Optional[str] = Field(default=None, description="Final report file path")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional job metadata")


class ConnectionInfo(BaseModel):
    """WebSocket connection information."""
    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Connection identifier")
    job_id: Optional[str] = Field(default=None, description="Associated job ID")
    connected_at: datetime = Field(default_factory=datetime.now, description="Connection timestamp")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Connection metadata")