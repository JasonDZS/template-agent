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
from ..schema import AgentState
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
        
        # Check if task was cancelled before execution
        from ..template.task_models import TaskStatus
        if task.status == TaskStatus.CANCELLED:
            logger.info(f"🚫 Task {task.title} execution was cancelled")
            return TaskOutput(
                task_id=task.id,
                content="Task was cancelled",
                metadata={"status": "cancelled"}
            )
        
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
            
            # Create and configure SectionAgent (remove name parameter to avoid conflict)
            agent = SectionAgent(
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
            await agent.run()
            
            # Debug agent state
            logger.info(f"🔍 Agent state after run(): is_finished={agent.is_finished()}, is_completed={getattr(agent, 'is_completed', None)}, state={getattr(agent, 'state', None)}")
            
            if agent.is_finished():
                agent_content = content = agent.get_content()
                if not content:
                    content = f"[Generation failed] {task.title}: No content generated"
                    logger.warning(f"⚠️ Agent finished but generated no content")
                else:
                    logger.info(f"✅ Agent finished successfully, content length: {len(content)}")
            elif getattr(agent, 'state', None) == AgentState.IDLE and getattr(agent, 'current_step', 0) == 0:
                # Agent reached max steps and was reset to IDLE - treat as completed with error
                agent_content = content = agent.get_content() or f"[Generation timeout] {task.title}: Agent reached maximum steps without completion"
                logger.warning(f"⚠️ Agent reached max steps without finishing, treating as completed with timeout")
            else:
                agent_content = content = f"[Generation incomplete] {task.title}: Agent did not finish successfully"
                logger.error(f"❌ Agent did not finish successfully. State details:")
                logger.error(f"   - is_completed: {getattr(agent, 'is_completed', 'N/A')}")
                logger.error(f"   - state: {getattr(agent, 'state', 'N/A')}")
                logger.error(f"   - current_step: {getattr(agent, 'current_step', 'N/A')}")
                logger.error(f"   - max_steps: {getattr(agent, 'max_steps', 'N/A')}")
                logger.error(f"   - generated_content length: {len(getattr(agent, 'generated_content', ''))}")
                logger.error(f"   - memory messages count: {len(agent.memory.messages) if hasattr(agent, 'memory') else 'N/A'}")
                
                # Log last few memory messages for debugging
                if hasattr(agent, 'memory') and agent.memory.messages:
                    logger.error(f"   - Last memory messages:")
                    for i, msg in enumerate(agent.memory.messages[-3:]):  # Last 3 messages
                        logger.error(f"     [{i}] {msg.role}: {str(msg.content)[:100]}...")
            
            # Format content with proper heading (check if already has heading)
            if not content.startswith('#'):
                formatted_content = "#" * task.level + " " + task.title + "\n" + content
            else:
                # Ensure proper heading level
                lines = content.split('\n', 1)
                if lines[0].startswith('#'):
                    header = "#" * task.level + " " + task.title
                    formatted_content = header + ("\n" + lines[1] if len(lines) > 1 else "")
                else:
                    formatted_content = content
            
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
                    "memory": agent.memory.messages if hasattr(agent, 'memory') else [],
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
                },
                metadata=task_output.metadata  # Include metadata in the completion message
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
        
        # Check if task was cancelled before execution
        from ..template.task_models import TaskStatus
        if task.status == TaskStatus.CANCELLED:
            logger.info(f"🚫 Task {task.title} execution was cancelled")
            return TaskOutput(
                task_id=task.id,
                content="Task was cancelled",
                metadata={"status": "cancelled"}
            )
        
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
            
            # Get child contents from dependencies in the correct order
            child_contents = []
            failed_contents = []
            
            for dep_id in task.dependencies:
                if dep_id in task_input.dependencies_content:
                    content = task_input.dependencies_content[dep_id]
                    logger.info(f"   📝 Processing dependency {dep_id}, content preview: '{content[:200]}...' (length: {len(content)})")
                    
                    # Filter out failed content
                    is_failed = ("[Generation incomplete]" in content or 
                                "[Generation failed]" in content or 
                                "[Generation timeout]" in content or
                                "[Merge incomplete]" in content or 
                                "[Merge failed]" in content or
                                "[Merge timeout]" in content or
                                "[No content generated]" in content or
                                "Agent did not finish successfully" in content)
                    
                    if is_failed:
                        failed_contents.append(content)
                        logger.warning(f"   ⚠️  Skipping failed dependency content for task {dep_id}")
                    else:
                        child_contents.append(content)
                        logger.info(f"   ✅ Valid dependency content for task {dep_id}")
                else:
                    logger.warning(f"   ⚠️  Missing dependency content for task {dep_id}")
            
            logger.info(f"   📝 Valid child contents: {len(child_contents)}, Failed contents: {len(failed_contents)}")
            
            # Send progress update
            await self.send_stream_message(
                job_id=job_id,
                task_id=task.id,
                message_type=MessageType.TASK_PROGRESS,
                content={"progress": 20, "status": "preparing_merge"}
            )
            
            logger.info(f"🔀 Starting MergeAgent for: {task.title}")
            logger.info(f"   Expected dependencies: {len(task.dependencies)} {task.dependencies}")
            logger.info(f"   Available dependencies: {len(task_input.dependencies_content)} {list(task_input.dependencies_content.keys())}")
            logger.info(f"   Collected child contents: {len(child_contents)}")
            
            # Log child content previews for debugging
            for i, content in enumerate(child_contents):
                preview = content[:100].replace('\n', '\\n') if content else "[empty]"
                logger.info(f"   Child content {i+1}: {len(content)} chars - '{preview}{'...' if len(content) > 100 else ''}'")
                
            logger.info(f"   Merging {len(child_contents)} child contents")
            
            if not child_contents:
                if failed_contents:
                    agent_content = content = f"[Merge Warning] {task.title}: All child contents failed to generate. Unable to merge."
                    logger.warning(f"⚠️ All {len(failed_contents)} child contents failed for merging {task.title}")
                else:
                    agent_content = content = f"[Merge Warning] {task.title}: No child contents to merge"
                    logger.warning(f"⚠️ No child contents available for merging {task.title}")
                memory = []
            else:
                # Send progress update
                await self.send_stream_message(
                    job_id=job_id,
                    task_id=task.id,
                    message_type=MessageType.TASK_PROGRESS,
                    content={"progress": 40, "status": "merging_content"}
                )
                
                # Create and run MergeAgent (remove unsupported parameters)
                agent = MergeAgent(
                    section_info=section_info,
                    report_context=report_context,
                    child_contents=child_contents,
                    output_format=task.section_content if hasattr(task, 'section_content') else None
                )
                
                # Send progress update
                await self.send_stream_message(
                    job_id=job_id,
                    task_id=task.id,
                    message_type=MessageType.TASK_PROGRESS,
                    content={"progress": 70, "status": "generating_merged_content"}
                )
                
                await agent.run()
                
                # Debug agent state
                logger.info(f"🔍 MergeAgent state after run(): is_finished={agent.is_finished()}, is_completed={getattr(agent, 'is_completed', None)}, state={getattr(agent, 'state', None)}")
                
                if agent.is_finished():
                    agent_content = content = agent.get_content()
                    memory = agent.memory.messages if hasattr(agent, 'memory') else []
                    
                    if not content:
                        content = f"[Merge failed] {task.title}: No content generated"
                        logger.warning(f"⚠️ MergeAgent finished but generated no content")
                    else:
                        logger.info(f"✅ MergeAgent finished successfully, content length: {len(content)}")
                elif getattr(agent, 'state', None) == AgentState.IDLE and getattr(agent, 'current_step', 0) == 0:
                    # Agent reached max steps and was reset to IDLE - treat as completed with error
                    agent_content = content = agent.get_content() or f"[Merge timeout] {task.title}: Agent reached maximum steps without completion"
                    memory = agent.memory.messages if hasattr(agent, 'memory') else []
                    logger.warning(f"⚠️ MergeAgent reached max steps without finishing, treating as completed with timeout")
                else:
                    agent_content = content = f"[Merge incomplete] {task.title}: Agent did not finish successfully"
                    memory = []
                    logger.error(f"❌ MergeAgent did not finish successfully. State details:")
                    logger.error(f"   - is_completed: {getattr(agent, 'is_completed', 'N/A')}")
                    logger.error(f"   - state: {getattr(agent, 'state', 'N/A')}")
                    logger.error(f"   - current_step: {getattr(agent, 'current_step', 'N/A')}")
                    logger.error(f"   - max_steps: {getattr(agent, 'max_steps', 'N/A')}")
                    logger.error(f"   - generated_content length: {len(getattr(agent, 'generated_content', ''))}")
                    logger.error(f"   - memory messages count: {len(agent.memory.messages) if hasattr(agent, 'memory') else 'N/A'}")
                    
                    # Log last few memory messages for debugging
                    if hasattr(agent, 'memory') and agent.memory.messages:
                        logger.error(f"   - Last memory messages:")
                        for i, msg in enumerate(agent.memory.messages[-3:]):  # Last 3 messages
                            logger.error(f"     [{i}] {msg.role}: {str(msg.content)[:100]}...")
            
            # Format content with proper heading (check if already has heading for merge)
            if not content.startswith('#'):
                formatted_content = "#" * task.level + " " + task.title + "\n" + content
            else:
                # Ensure proper heading level
                lines = content.split('\n', 1)
                if lines[0].startswith('#'):
                    header = "#" * task.level + " " + task.title
                    formatted_content = header + ("\n" + lines[1] if len(lines) > 1 else "")
                else:
                    formatted_content = content
            
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
                },
                metadata=task_output.metadata  # Include metadata in the completion message
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