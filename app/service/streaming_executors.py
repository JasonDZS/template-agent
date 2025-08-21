#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming Task Executors for Report Generation Service.

This module provides task executors that support streaming output during execution,
allowing real-time progress tracking and content streaming for web interfaces.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Awaitable
from pathlib import Path

from ..template.task_executors import TaskExecutor, TaskInput, TaskOutput
from ..template.task_models import Task, TaskType
from ..agent.section_agent import SectionAgent
from ..agent.merge_agent import MergeAgent
from ..logger import logger

from .models import StreamMessage, MessageType, TaskStatus


class StreamingExecutorBase(TaskExecutor):
    """Base class for streaming task executors."""
    
    def __init__(self, 
                 knowledge_base_path: str = "workdir/documents",
                 stream_callback: Optional[Callable[[StreamMessage], Awaitable[None]]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base_path = knowledge_base_path
        self.stream_callback = stream_callback
        
    async def send_stream_message(self, job_id: str, task_id: str, 
                                 message_type: MessageType, 
                                 content: Any, 
                                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send a streaming message."""
        if self.stream_callback:
            message = StreamMessage(
                job_id=job_id,
                task_id=task_id,
                message_type=message_type,
                content=content,
                metadata=metadata or {}
            )
            try:
                await self.stream_callback(message)
            except Exception as e:
                logger.error(f"Failed to send stream message: {e}")


class StreamingSectionExecutor(StreamingExecutorBase):
    """Streaming task executor for section generation using SectionAgent."""
    
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle generation tasks."""
        return task.task_type == TaskType.GENERATION
    
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute generation task with streaming output."""
        task = task_input.task
        job_id = task_input.context.get("job_id", "unknown")
        
        # Send task start message
        await self.send_stream_message(
            job_id=job_id,
            task_id=task.id,
            message_type=MessageType.TASK_START,
            content={
                "task_title": task.title,
                "task_type": "generation",
                "level": task.level
            }
        )
        
        # Log task start
        self.log_task_input(task_input)
        
        try:
            # Create section info for SectionAgent
            section_info = {
                "id": task.id,
                "content": task.title,
                "level": task.level
            }
            
            # Create report context
            report_context = {
                "title": task_input.context.get("report_title", "Report"),
                "type": task_input.context.get("report_type", "document")
            }
            
            # Send progress update
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_PROGRESS,
                content={"progress": 10, "status": "initializing_agent"}
            )
            
            # Create and configure SectionAgent
            agent = SectionAgent(
                name=f"section_{task.id}",
                section_info=section_info,
                report_context=report_context,
                output_format=task.section_content,
                knowledge_base_path=self.knowledge_base_path
            )
            
            # Send progress update
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_PROGRESS,
                content={"progress": 30, "status": "retrieving_knowledge"}
            )
            
            # Create a custom step method that sends streaming updates
            original_step = agent.step
            
            async def streaming_step():
                """Override step method to send streaming updates."""
                result = await original_step()
                
                # Send progress based on agent state
                progress = 50
                if hasattr(agent, 'is_completed') and agent.is_completed:
                    progress = 90
                    
                await self.send_stream_message(
                    job_id=job_id,
                    task_id=task.id,
                    message_type=MessageType.TASK_PROGRESS,
                    content={"progress": progress, "status": "generating_content"}
                )
                
                return result
            
            # Replace the agent's step method
            agent.step = streaming_step
            
            # Run section generation
            logger.info(f"🔧 Starting SectionAgent for: {task.title}")
            agent_content = content = await agent.run_section_generation()
            
            if not content:
                content = f"[Generation failed] {task.title}: No content generated"
            
            # Format content with proper heading
            formatted_content = "#" * task.level + " " + task.title + "\n" + content
            
            # Send content streaming message
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_CONTENT,
                content=formatted_content
            )
            
            # Create task output
            task_output = TaskOutput(
                content=formatted_content,
                metadata={
                    "task_type": "generation",
                    "agent_type": "SectionAgent",
                    "section_title": task.title,
                    "section_level": task.level,
                    "content_length": len(formatted_content),
                    "executor_id": self.executor_id,
                    "agent_content": agent_content,
                    "memory": agent.messages if hasattr(agent, 'messages') else [],
                    "streaming": True
                }
            )
            
            # Send task completion message
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_COMPLETE,
                content={
                    "task_title": task.title,
                    "content_length": len(formatted_content),
                    "status": "completed"
                }
            )
            
            self.log_task_output(task, task_output)
            return task_output
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.id} failed: {error_msg}")
            
            # Send error message
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_ERROR,
                content={
                    "task_title": task.title,
                    "error": error_msg,
                    "status": "failed"
                }
            )
            
            self.log_task_error(task, e)
            
            # Return error task output
            error_content = f"[Generation Error] {task.title}: {error_msg}"
            return TaskOutput(
                content=error_content,
                metadata={
                    "task_type": "generation",
                    "error": error_msg,
                    "executor_id": self.executor_id,
                    "streaming": True
                }
            )


class StreamingMergeExecutor(StreamingExecutorBase):
    """Streaming task executor for content merging using MergeAgent."""
    
    def __init__(self, 
                 knowledge_base_path: str = "workdir/documents",
                 enable_model_merge: bool = True,
                 stream_callback: Optional[Callable[[StreamMessage], Awaitable[None]]] = None,
                 **kwargs):
        super().__init__(knowledge_base_path, stream_callback, **kwargs)
        self.enable_model_merge = enable_model_merge
    
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle merge tasks."""
        return task.task_type == TaskType.MERGE
    
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute merge task with streaming output."""
        task = task_input.task
        job_id = task_input.context.get("job_id", "unknown")
        
        # Send task start message
        await self.send_stream_message(
            job_id=job_id,
            task_id=task.id,
            message_type=MessageType.TASK_START,
            content={
                "task_title": task.title,
                "task_type": "merge",
                "level": task.level,
                "child_count": len(task_input.dependencies_content)
            }
        )
        
        self.log_task_input(task_input)
        
        try:
            # Create section info for MergeAgent
            section_info = {
                "id": task.id,
                "content": task.title,
                "level": task.level
            }
            
            # Create report context
            report_context = {
                "title": task_input.context.get("report_title", "Report"),
                "type": task_input.context.get("report_type", "document")
            }
            
            # Get child contents from dependencies
            child_contents = list(task_input.dependencies_content.values())
            
            # Send progress update
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_PROGRESS,
                content={"progress": 20, "status": "preparing_merge"}
            )
            
            logger.info(f"🔀 Starting MergeAgent for: {task.title}")
            logger.info(f"   Merging {len(child_contents)} child contents")
            
            if not child_contents:
                agent_content = content = f"[Merge Warning] {task.title}: No child contents to merge"
                memory = []
            else:
                # Send progress update
                await self.send_stream_message(
                    job_id=job_id,
                    task_id=task.id,
                    message_type=MessageType.TASK_PROGRESS,
                    content={"progress": 40, "status": "merging_content"}
                )
                
                # Create and run MergeAgent
                agent = MergeAgent(
                    section_info=section_info,
                    report_context=report_context,
                    child_contents=child_contents,
                    knowledge_base_path=self.knowledge_base_path,
                    enable_model_merge=self.enable_model_merge
                )
                
                # Send progress update
                await self.send_stream_message(
                    job_id=job_id,
                    task_id=task.id,
                    message_type=MessageType.TASK_PROGRESS,
                    content={"progress": 70, "status": "generating_merged_content"}
                )
                
                agent_content = content = await agent.run_merge()
                memory = agent.messages if hasattr(agent, 'messages') else []
                
                if not content:
                    content = f"[Merge failed] {task.title}: No content generated"
            
            # Format content with proper heading
            formatted_content = "#" * task.level + " " + task.title + "\n" + content
            
            # Send content streaming message
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_CONTENT,
                content=formatted_content
            )
            
            # Create task output
            task_output = TaskOutput(
                content=formatted_content,
                metadata={
                    "task_type": "merge",
                    "agent_type": "MergeAgent",
                    "section_title": task.title,
                    "section_level": task.level,
                    "child_count": len(child_contents),
                    "expected_dependencies": len(task.dependencies),
                    "received_dependencies": len(task_input.dependencies_content),
                    "content_length": len(formatted_content),
                    "enable_model_merge": self.enable_model_merge,
                    "executor_id": self.executor_id,
                    "agent_content": agent_content,
                    "memory": memory,
                    "streaming": True
                }
            )
            
            # Send task completion message
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_COMPLETE,
                content={
                    "task_title": task.title,
                    "content_length": len(formatted_content),
                    "child_count": len(child_contents),
                    "merge_mode": "model" if self.enable_model_merge else "simple",
                    "status": "completed"
                }
            )
            
            self.log_task_output(task, task_output)
            return task_output
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Merge task {task.id} failed: {error_msg}")
            
            # Send error message
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_ERROR,
                content={
                    "task_title": task.title,
                    "error": error_msg,
                    "status": "failed"
                }
            )
            
            self.log_task_error(task, e)
            
            # Return error task output
            error_content = f"[Merge Error] {task.title}: {error_msg}"
            return TaskOutput(
                content=error_content,
                metadata={
                    "task_type": "merge",
                    "error": error_msg,
                    "executor_id": self.executor_id,
                    "streaming": True
                }
            )