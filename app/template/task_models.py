"""Task Models for Template Agent.

This module defines the core data models for task scheduling including:
- Task types and status enums
- Task input/output models
- Task result tracking
- Base task model with dependency management
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator

from .document_types import HeadingNode


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Type of task in the schedule."""
    GENERATION = "generation"  # Generate content for leaf headings
    MERGE = "merge"           # Merge content from child tasks


class TaskResult(BaseModel):
    """Result of task execution."""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Task execution status")
    content: Optional[str] = Field(None, description="Generated content from task execution")
    error: Optional[str] = Field(None, description="Error message if task failed")
    start_time: Optional[datetime] = Field(None, description="Task start timestamp")
    end_time: Optional[datetime] = Field(None, description="Task completion timestamp")
    duration: Optional[float] = Field(None, description="Task execution duration in seconds")
    
    @field_validator('duration')
    def validate_duration(cls, v):
        """Validate that duration is non-negative."""
        if v is not None and v < 0:
            raise ValueError('Duration must be non-negative')
        return v
    
    @model_validator(mode='before')
    def validate_timestamps(cls, values):
        """Validate timestamp consistency."""
        if isinstance(values, dict):
            start_time = values.get('start_time')
            end_time = values.get('end_time')
            duration = values.get('duration')
            
            if start_time and end_time and start_time > end_time:
                raise ValueError('Start time cannot be after end time')
            
            if start_time and end_time and duration is None:
                # Auto-calculate duration if not provided
                values['duration'] = (end_time - start_time).total_seconds()
        
        return values
    
    class Config:
        from_attributes = True


class TaskInput(BaseModel):
    """Input data for task execution."""
    task: 'Task' = Field(..., description="Task to be executed")
    dependencies_content: Dict[str, str] = Field(
        default_factory=dict, 
        description="Content from dependency tasks mapped by task ID"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Execution context and configuration"
    )
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True  # Allow custom Task type


class TaskOutput(BaseModel):
    """Output data from task execution."""
    content: str = Field(..., description="Generated content from task execution")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Task execution metadata and metrics"
    )
    artifacts: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional artifacts generated during execution"
    )
    
    @field_validator('content')
    def validate_content_not_empty(cls, v):
        """Validate that content is not empty."""
        if v is not None and not v.strip():
            raise ValueError('Content cannot be empty string (use None instead)')
        return v
    
    class Config:
        from_attributes = True


class Task(BaseModel):
    """Represents a single task in the schedule."""
    id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task (generation or merge)")
    heading_node: HeadingNode = Field(..., description="Associated heading node from markdown")
    title: str = Field(..., description="Human-readable task title")
    level: int = Field(..., ge=1, le=6, description="Heading level (1-6)")
    dependencies: List[str] = Field(
        default_factory=list, 
        description="List of task IDs this task depends on (ordered)"
    )
    dependents: Set[str] = Field(
        default_factory=set, 
        description="Set of task IDs that depend on this task"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, 
        description="Current task execution status"
    )
    result: Optional[TaskResult] = Field(
        None, 
        description="Task execution result (populated after completion)"
    )
    
    # Task execution context
    section_content: str = Field(
        default="", 
        description="Content from the markdown section for this task"
    )
    priority: int = Field(
        default=0, 
        description="Task priority (higher values executed first)"
    )
    estimated_duration: float = Field(
        default=0.0, 
        ge=0.0, 
        description="Estimated execution time in seconds"
    )
    
    # Task output metadata (populated after execution)
    output_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata from task execution"
    )
    
    @field_validator('title')
    def validate_title_not_empty(cls, v):
        """Validate that title is not empty."""
        if not v or not v.strip():
            raise ValueError('Task title cannot be empty')
        return v
    
    @model_validator(mode='after')
    def populate_section_content(self):
        """Populate section content from heading node."""
        if self.heading_node and not self.section_content:
            self.section_content = self.heading_node.get_content_for_agent()
        return self
    
    class Config:
        from_attributes = True
        arbitrary_types_allowed = True  # Allow HeadingNode type
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return (
            self.status == TaskStatus.PENDING and 
            set(self.dependencies).issubset(completed_tasks)
        )
    
    def mark_completed(self, content: str = "", error: str = "") -> None:
        """Mark task as completed with result."""
        end_time = datetime.now()
        start_time = self.result.start_time if self.result else end_time
        duration = (end_time - start_time).total_seconds()
        
        self.status = TaskStatus.COMPLETED if not error else TaskStatus.FAILED
        self.result = TaskResult(
            task_id=self.id,
            status=self.status,
            content=content,
            error=error,
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )
    
    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        if not self.result:
            self.result = TaskResult(
                task_id=self.id,
                status=self.status,
                start_time=datetime.now()
            )
        else:
            # Update existing result with new start time
            self.result = self.result.model_copy(update={'start_time': datetime.now()})


# Update forward references
Task.model_rebuild()