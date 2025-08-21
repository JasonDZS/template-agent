#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generation Service.

This module provides the main service class for generating reports from templates
with full streaming support, concurrent execution, and real-time progress tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Awaitable, Any
import uuid

from ..template import MarkdownTaskSchedule, TaskScheduler
from ..template.task_models import Task, TaskType
from ..logger import logger

from .models import (
    ReportGenerationRequest, ReportJob, ReportTask, TaskStatus,
    StreamMessage, MessageType, JobProgress, TaskProgress,
    ReportGenerationResponse, ConnectionInfo
)
from .streaming_executors import StreamingSectionExecutor, StreamingMergeExecutor


class ReportGenerationService:
    """
    Main service for template-based report generation with streaming support.
    
    This service manages the complete lifecycle of report generation including:
    - Template parsing and task creation
    - Concurrent task execution with streaming output
    - WebSocket connection management
    - Progress tracking and status updates
    - Final report assembly and storage
    """
    
    def __init__(self):
        """Initialize the report generation service."""
        self.jobs: Dict[str, ReportJob] = {}  # Active jobs
        self.connections: Dict[str, ConnectionInfo] = {}  # WebSocket connections
        self.job_connections: Dict[str, Set[str]] = {}  # job_id -> connection_ids
        
        # Background task for cleaning up completed jobs
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 3600  # 1 hour
        
        logger.info("ReportGenerationService initialized")
    
    async def start(self) -> None:
        """Start the service and background tasks."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_completed_jobs())
        logger.info("ReportGenerationService started")
    
    async def stop(self) -> None:
        """Stop the service and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running jobs
        for job_id in list(self.jobs.keys()):
            await self.cancel_job(job_id)
            
        logger.info("ReportGenerationService stopped")
    
    async def create_report_job(self, request: ReportGenerationRequest) -> ReportGenerationResponse:
        """
        Create a new report generation job.
        
        Args:
            request: Report generation request parameters
            
        Returns:
            ReportGenerationResponse with job details
        """
        job_id = str(uuid.uuid4())
        
        try:
            # Validate template file
            template_path = Path(request.template_path)
            if not template_path.exists():
                raise ValueError(f"Template file not found: {request.template_path}")
            
            # Create job
            job = ReportJob(
                job_id=job_id,
                request=request,
                status="initializing"
            )
            
            # Parse template and create tasks
            await self._initialize_job_tasks(job)
            
            # Store job
            self.jobs[job_id] = job
            self.job_connections[job_id] = set()
            
            logger.info(f"Created report job {job_id} with {len(job.tasks)} tasks")
            
            return ReportGenerationResponse(
                job_id=job_id,
                status="created",
                message=f"Report generation job created with {len(job.tasks)} tasks",
                progress=await self._get_job_progress(job_id)
            )
            
        except Exception as e:
            error_msg = f"Failed to create report job: {str(e)}"
            logger.error(error_msg)
            
            return ReportGenerationResponse(
                job_id=job_id,
                status="failed",
                message=error_msg
            )
    
    async def start_job(self, job_id: str) -> ReportGenerationResponse:
        """
        Start executing a report generation job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ReportGenerationResponse with execution status
        """
        if job_id not in self.jobs:
            return ReportGenerationResponse(
                job_id=job_id,
                status="not_found",
                message="Job not found"
            )
        
        job = self.jobs[job_id]
        
        if job.status == "running":
            return ReportGenerationResponse(
                job_id=job_id,
                status="already_running",
                message="Job is already running"
            )
        
        try:
            job.status = "running"
            job.start_time = datetime.now()
            
            # Start job execution in background
            asyncio.create_task(self._execute_job(job))
            
            logger.info(f"Started job execution for {job_id}")
            
            return ReportGenerationResponse(
                job_id=job_id,
                status="started",
                message="Job execution started",
                progress=await self._get_job_progress(job_id)
            )
            
        except Exception as e:
            error_msg = f"Failed to start job: {str(e)}"
            logger.error(error_msg)
            job.status = "failed"
            job.error_message = error_msg
            
            return ReportGenerationResponse(
                job_id=job_id,
                status="failed",
                message=error_msg
            )
    
    async def get_job_status(self, job_id: str) -> Optional[ReportGenerationResponse]:
        """
        Get current status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ReportGenerationResponse with current status
        """
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        progress = await self._get_job_progress(job_id)
        
        return ReportGenerationResponse(
            job_id=job_id,
            status=job.status,
            message=f"Job is {job.status}",
            report_path=job.final_report_path,
            progress=progress,
            created_at=job.start_time
        )
    
    async def cancel_job(self, job_id: str) -> ReportGenerationResponse:
        """
        Cancel a running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ReportGenerationResponse with cancellation status
        """
        if job_id not in self.jobs:
            return ReportGenerationResponse(
                job_id=job_id,
                status="not_found",
                message="Job not found"
            )
        
        job = self.jobs[job_id]
        job.status = "cancelled"
        job.end_time = datetime.now()
        
        # Update all running tasks to cancelled
        for task in job.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
                task.end_time = datetime.now()
        
        # Send cancellation message to connected clients
        await self._broadcast_job_message(
            job_id=job_id,
            message_type=MessageType.JOB_ERROR,
            content={"status": "cancelled", "message": "Job was cancelled"}
        )
        
        logger.info(f"Cancelled job {job_id}")
        
        return ReportGenerationResponse(
            job_id=job_id,
            status="cancelled",
            message="Job cancelled successfully"
        )
    
    async def register_connection(self, connection_id: str, job_id: Optional[str] = None) -> ConnectionInfo:
        """
        Register a new WebSocket connection.
        
        Args:
            connection_id: Connection identifier
            job_id: Optional job ID to associate with connection
            
        Returns:
            ConnectionInfo object
        """
        connection = ConnectionInfo(
            connection_id=connection_id,
            job_id=job_id
        )
        
        self.connections[connection_id] = connection
        
        if job_id and job_id in self.job_connections:
            self.job_connections[job_id].add(connection_id)
        
        logger.info(f"Registered connection {connection_id} for job {job_id}")
        return connection
    
    async def unregister_connection(self, connection_id: str) -> None:
        """
        Unregister a WebSocket connection.
        
        Args:
            connection_id: Connection identifier
        """
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            if connection.job_id and connection.job_id in self.job_connections:
                self.job_connections[connection.job_id].discard(connection_id)
            
            del self.connections[connection_id]
            logger.info(f"Unregistered connection {connection_id}")
    
    async def _initialize_job_tasks(self, job: ReportJob) -> None:
        """Initialize tasks for a job from template."""
        request = job.request
        
        # Create task schedule from template
        schedule = MarkdownTaskSchedule(
            request.template_path,
            max_concurrent=request.max_concurrent
        )
        
        # Convert schedule tasks to report tasks
        for task_id, task in schedule.tasks.items():
            report_task = ReportTask(
                task_id=task.id,
                job_id=job.job_id,
                task_type=task.task_type.value,
                title=task.title,
                level=task.level,
                dependencies=[dep for dep in task.dependencies],
                metadata={
                    "template_task": True,
                    "estimated_duration": task.estimated_duration,
                    "priority": task.priority
                }
            )
            job.tasks[task_id] = report_task
    
    async def _execute_job(self, job: ReportJob) -> None:
        """Execute a complete job."""
        job_id = job.job_id
        request = job.request
        
        try:
            # Send job start message
            await self._broadcast_job_message(
                job_id=job_id,
                message_type=MessageType.JOB_PROGRESS,
                content={"status": "started", "message": "Job execution started"}
            )
            
            # Create task scheduler
            scheduler = TaskScheduler(max_concurrent=request.max_concurrent)
            
            # Create streaming callback
            async def stream_callback(message: StreamMessage) -> None:
                await self._handle_stream_message(message)
            
            # Register streaming executors
            section_executor = StreamingSectionExecutor(
                knowledge_base_path=request.knowledge_base_path,
                stream_callback=stream_callback if request.enable_streaming else None
            )
            merge_executor = StreamingMergeExecutor(
                knowledge_base_path=request.knowledge_base_path,
                enable_model_merge=request.enable_model_merge,
                stream_callback=stream_callback if request.enable_streaming else None
            )
            
            scheduler.register_executor(section_executor)
            scheduler.register_executor(merge_executor)
            
            # Set global context
            scheduler.set_global_context({
                "job_id": job_id,
                "report_title": request.report_title or f"Generated Report",
                "report_type": "template_generated",
                "template_path": request.template_path
            })
            
            # Recreate template tasks for scheduler using the SAME task IDs
            schedule = MarkdownTaskSchedule(
                request.template_path,
                max_concurrent=request.max_concurrent
            )
            
            # Create a mapping from old task IDs to new task IDs for consistency
            old_to_new_task_mapping = {}
            for old_task_id, old_task in schedule.tasks.items():
                for report_task_id, report_task in job.tasks.items():
                    if report_task.title == old_task.title and report_task.task_type == old_task.task_type.value:
                        old_to_new_task_mapping[old_task.id] = report_task_id
                        # Update the scheduler task with the correct ID
                        old_task.id = report_task_id
                        break
            
            # Update task dependencies to use the new task IDs
            for task in schedule.tasks.values():
                new_dependencies = []
                for dep_id in task.dependencies:
                    if dep_id in old_to_new_task_mapping:
                        new_dependencies.append(old_to_new_task_mapping[dep_id])
                    else:
                        new_dependencies.append(dep_id)
                task.dependencies = new_dependencies
            
            # Add tasks to scheduler with corrected IDs and dependencies
            for task in schedule.tasks.values():
                scheduler.add_task(task)
            
            # Execute tasks
            await self._execute_scheduler_tasks(scheduler, job)
            
            # Generate final report
            if job.status != "cancelled":
                await self._generate_final_report(scheduler, job)
            
        except Exception as e:
            error_msg = f"Job execution failed: {str(e)}"
            logger.error(error_msg)
            job.status = "failed"
            job.error_message = error_msg
            job.end_time = datetime.now()
            
            await self._broadcast_job_message(
                job_id=job_id,
                message_type=MessageType.JOB_ERROR,
                content={"status": "failed", "error": error_msg}
            )
    
    async def _execute_scheduler_tasks(self, scheduler: TaskScheduler, job: ReportJob) -> None:
        """Execute tasks using the scheduler."""
        job_id = job.job_id
        max_rounds = 50
        round_num = 1
        
        while round_num <= max_rounds and job.status == "running":
            ready_tasks = scheduler.get_ready_tasks()
            
            if not ready_tasks:
                progress = scheduler.get_progress()
                total_completed = progress["completed_tasks"] + progress["failed_tasks"]
                
                if total_completed == progress["total_tasks"]:
                    break
                else:
                    logger.warning(f"No ready tasks in round {round_num} for job {job_id}")
                    break
            
            # Execute ready tasks
            available_slots = scheduler.max_concurrent - len(scheduler.running_tasks)
            tasks_to_execute = ready_tasks[:available_slots]
            
            # Update task status
            for task in tasks_to_execute:
                if task.id in job.tasks:
                    job.tasks[task.id].status = TaskStatus.RUNNING
                    job.tasks[task.id].start_time = datetime.now()
            
            # Send progress update
            progress = scheduler.get_progress()
            await self._broadcast_job_message(
                job_id=job_id,
                message_type=MessageType.JOB_PROGRESS,
                content={
                    "round": round_num,
                    "executing_tasks": len(tasks_to_execute),
                    "progress": progress
                }
            )
            
            # Execute tasks concurrently
            execution_tasks = []
            for task in tasks_to_execute:
                execution_tasks.append(scheduler.execute_task(task))
            
            if execution_tasks:
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Update task status based on results
                for i, (task, result) in enumerate(zip(tasks_to_execute, results)):
                    if task.id in job.tasks:
                        job_task = job.tasks[task.id]
                        job_task.end_time = datetime.now()
                        
                        if isinstance(result, Exception):
                            job_task.status = TaskStatus.FAILED
                            job_task.error_message = str(result)
                        else:
                            job_task.status = TaskStatus.COMPLETED
                            if hasattr(task, 'result') and task.result:
                                job_task.content = task.result.content
                                # Also copy metadata from task result
                                if hasattr(task.result, 'metadata') and task.result.metadata:
                                    job_task.metadata.update(task.result.metadata)
            
            round_num += 1
            await asyncio.sleep(0.1)  # Small delay between rounds
        
        # Mark job as completed
        if job.status == "running":
            job.status = "completed"
            job.end_time = datetime.now()
    
    async def _generate_final_report(self, scheduler: TaskScheduler, job: ReportJob) -> None:
        """Generate the final report file."""
        job_id = job.job_id
        request = job.request
        
        try:
            # Get all task results
            task_results = scheduler.get_task_results()
            
            if not task_results:
                raise ValueError("No task results available for final report")
            
            # Find root tasks and sort by document order
            all_dependencies = set()
            for task in scheduler.tasks.values():
                all_dependencies.update(task.dependencies)
            
            root_tasks = []
            for task_id, task in scheduler.tasks.items():
                if (task_id not in all_dependencies and 
                    task.result and task.result.content):
                    root_tasks.append((task, task.result))
            
            # Sort by level and document order
            def extract_line_number(task_obj):
                try:
                    heading_node = task_obj.heading_node
                    if hasattr(heading_node, 'attributes') and heading_node.attributes:
                        return heading_node.attributes.get('line_number', 0)
                    return 0
                except (AttributeError, KeyError):
                    return 0
            
            root_tasks.sort(key=lambda x: (x[0].level, extract_line_number(x[0])))
            
            # Build final report
            template_name = Path(request.template_path).stem
            report_title = request.report_title or f"Generated Report - {template_name}"
            
            final_report = f"# {report_title}\n\n"
            final_report += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            
            for task, result in root_tasks:
                content = result.content.strip()
                if not content.startswith('#') and task.title:
                    final_report += f"## {task.title}\n\n{content}\n\n"
                else:
                    final_report += f"{content}\n\n"
            
            # Save report
            if request.output_path:
                output_path = Path(request.output_path)
            else:
                output_path = Path(f"workdir/output/report_{template_name}_{job_id[:8]}.md")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            job.final_report_path = str(output_path)
            
            # Send completion message
            await self._broadcast_job_message(
                job_id=job_id,
                message_type=MessageType.JOB_COMPLETE,
                content={
                    "status": "completed",
                    "report_path": str(output_path),
                    "report_length": len(final_report),
                    "task_count": len(root_tasks)
                }
            )
            
            logger.info(f"Generated final report for job {job_id}: {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to generate final report: {str(e)}"
            logger.error(error_msg)
            job.status = "failed"
            job.error_message = error_msg
            
            await self._broadcast_job_message(
                job_id=job_id,
                message_type=MessageType.JOB_ERROR,
                content={"status": "failed", "error": error_msg}
            )
    
    async def _handle_stream_message(self, message: StreamMessage) -> None:
        """Handle incoming stream message."""
        job_id = message.job_id
        
        # Update task status if applicable
        if message.task_id and job_id in self.jobs:
            job = self.jobs[job_id]
            if message.task_id in job.tasks:
                task = job.tasks[message.task_id]
                
                if message.message_type == MessageType.TASK_START:
                    task.status = TaskStatus.STREAMING
                elif message.message_type == MessageType.TASK_COMPLETE:
                    task.status = TaskStatus.COMPLETED
                    task.progress = 100.0
                elif message.message_type == MessageType.TASK_ERROR:
                    task.status = TaskStatus.FAILED
                    if isinstance(message.content, dict):
                        task.error_message = message.content.get("error", "Unknown error")
                elif message.message_type == MessageType.TASK_PROGRESS:
                    if isinstance(message.content, dict):
                        task.progress = message.content.get("progress", 0.0)
        
        # Broadcast to connected clients
        await self._broadcast_message_to_job(job_id, message)
    
    async def _broadcast_job_message(self, job_id: str, message_type: MessageType, content: Any) -> None:
        """Broadcast a job-level message."""
        message = StreamMessage(
            job_id=job_id,
            message_type=message_type,
            content=content
        )
        await self._broadcast_message_to_job(job_id, message)
    
    async def _broadcast_message_to_job(self, job_id: str, message: StreamMessage) -> None:
        """Broadcast message to all connections subscribed to a job."""
        if job_id not in self.job_connections:
            return
        
        message_json = message.json()
        connection_ids = list(self.job_connections[job_id])
        
        for connection_id in connection_ids:
            try:
                # This would be replaced with actual WebSocket send in the API layer
                logger.debug(f"Broadcasting to connection {connection_id}: {message.message_type}")
                # await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to send message to connection {connection_id}: {e}")
                # Remove failed connection
                self.job_connections[job_id].discard(connection_id)
                self.connections.pop(connection_id, None)
    
    async def _get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get current job progress."""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        tasks = list(job.tasks.values())
        
        total_tasks = len(tasks)
        pending_tasks = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
        ready_tasks = sum(1 for t in tasks if t.status == TaskStatus.READY)
        running_tasks = sum(1 for t in tasks if t.status in [TaskStatus.RUNNING, TaskStatus.STREAMING])
        completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        
        if total_tasks > 0:
            overall_progress = (completed_tasks / total_tasks) * 100.0
        else:
            overall_progress = 0.0
        
        task_progress_list = [
            TaskProgress(
                task_id=task.task_id,
                title=task.title,
                status=task.status,
                progress=task.progress,
                content_length=len(task.content) if task.content else 0,
                error_message=task.error_message
            )
            for task in tasks
        ]
        
        return JobProgress(
            job_id=job_id,
            total_tasks=total_tasks,
            pending_tasks=pending_tasks,
            ready_tasks=ready_tasks,
            running_tasks=running_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            overall_progress=overall_progress,
            start_time=job.start_time,
            tasks=task_progress_list
        )
    
    async def _cleanup_completed_jobs(self) -> None:
        """Background task to cleanup old completed jobs."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)  # Keep jobs for 24 hours
                jobs_to_remove = []
                
                for job_id, job in self.jobs.items():
                    if (job.status in ["completed", "failed", "cancelled"] and 
                        job.end_time and job.end_time < cutoff_time):
                        jobs_to_remove.append(job_id)
                
                for job_id in jobs_to_remove:
                    del self.jobs[job_id]
                    if job_id in self.job_connections:
                        del self.job_connections[job_id]
                    logger.info(f"Cleaned up old job {job_id}")
                
                await asyncio.sleep(self._cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retry


# Global service instance
_service_instance: Optional[ReportGenerationService] = None


def get_report_service() -> ReportGenerationService:
    """Get the global report generation service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ReportGenerationService()
    return _service_instance