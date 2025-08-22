"""Task Scheduler for Template Agent.

This module provides task scheduling functionality including:
- Core task scheduler with dependency resolution
- Progress tracking and monitoring
- Concurrent task execution management
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any, Callable

from .task_models import Task, TaskResult, TaskStatus, TaskType
from .task_executors import ExecutorRegistry, GenerationTaskExecutor, MergeTaskExecutor


class TaskScheduler:
    """Task scheduler that manages task execution with dependency resolution."""
    
    def __init__(self, max_concurrent: int = 3, logger: Optional[logging.Logger] = None):
        """Initialize task scheduler.
        
        Args:
            max_concurrent: Maximum number of concurrent tasks
            logger: Optional logger instance
        """
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()
        
        # Task execution system
        self.executor_registry = ExecutorRegistry()
        
        # Logging
        self.logger = logger or logging.getLogger(f"TaskScheduler")
    
    def set_generation_handler(self, handler: Callable[[Task], str]) -> None:
        """Set handler for generation tasks."""
        gen_executor = GenerationTaskExecutor(generation_handler=handler)
        self.executor_registry.register(gen_executor)
    
    def set_merge_handler(self, handler: Callable[[Task, List[str]], str]) -> None:
        """Set handler for merge tasks."""
        merge_executor = MergeTaskExecutor(merge_handler=handler)
        self.executor_registry.register(merge_executor)
    
    def register_executor(self, executor) -> None:
        """Register a task executor.
        
        Args:
            executor: TaskExecutor to register
        """
        self.executor_registry.register(executor)
    
    def set_global_context(self, context: Dict[str, Any]) -> None:
        """Set global context for all executors.
        
        Args:
            context: Context data to set
        """
        self.executor_registry.set_global_context(context)
    
    def add_task(self, task: Task) -> None:
        """Add task to scheduler."""
        self.tasks[task.id] = task
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute."""
        ready_tasks = []
        for task in self.tasks.values():
            if task.is_ready(self.completed_tasks):
                ready_tasks.append(task)
        
        # Sort by priority and level (higher level tasks first for generation)
        ready_tasks.sort(key=lambda t: (-t.priority, t.level if t.task_type == TaskType.GENERATION else -t.level))
        return ready_tasks
    
    def can_schedule_more(self) -> bool:
        """Check if we can schedule more tasks."""
        return len(self.running_tasks) < self.max_concurrent
    
    async def execute_task(self, task: Task) -> None:
        """Execute a single task."""
        # Check if task was cancelled before execution
        if task.status == TaskStatus.CANCELLED:
            self.logger.info(f"🚫 Task {task.title} was cancelled before execution")
            return
        
        # Log task start
        self.logger.info(f"🚀 Starting Task - {task.title} ({task.task_type.value})")
        self.logger.info(f"   Task ID: {task.id}")
        self.logger.info(f"   Dependencies: {len(task.dependencies)}")
        self.logger.info(f"   Running Tasks: {len(self.running_tasks)}/{self.max_concurrent}")
        
        task.mark_running()
        self.running_tasks.add(task.id)
        
        try:
            # Check if task was cancelled after being marked as running
            if task.status == TaskStatus.CANCELLED:
                self.logger.info(f"🚫 Task {task.title} was cancelled during execution")
                return
                
            # Try to use new executor system first
            executor = self.executor_registry.get_executor(task)
            
            if executor:
                self.logger.debug(f"   Using executor: {executor.__class__.__name__}")
                
                # Use new executor system
                dependencies_results = {}
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if dep_task and dep_task.result:
                        dependencies_results[dep_id] = dep_task.result
                    else:
                        self.logger.warning(f"   ⚠️  Missing dependency result: {dep_id}")
                
                # Log dependency collection
                self.logger.debug(f"   Collected {len(dependencies_results)} dependency results")
                
                # Final cancellation check before expensive execution
                if task.status == TaskStatus.CANCELLED:
                    self.logger.info(f"🚫 Task {task.title} was cancelled before execution")
                    return
                
                task_input = executor.prepare_input(task, dependencies_results)
                task_output = await executor.execute(task_input)
                content = task_output.content
                metadata = task_output.metadata
                
                # Store output metadata in task result
                task.output_metadata = task_output.metadata
                
            else:
                raise RuntimeError(f"No executor found for task type: {task.task_type}")
            
            # Check if task was cancelled after execution but before completion
            if task.status == TaskStatus.CANCELLED:
                self.logger.info(f"🚫 Task {task.title} was cancelled after execution")
                return
            
            task.mark_completed(content=content, metadata=metadata)
            self.completed_tasks.add(task.id)
            
            # Log successful completion
            self.logger.info(f"✅ Completed Task - {task.title}")
            self.logger.info(f"   Content Length: {len(content) if content else 0} characters")
            self.logger.info(f"   Duration: {task.result.duration:.3f}s")
            
        except Exception as e:
            task.mark_completed(error=str(e))
            self.failed_tasks.add(task.id)
            
            # Log failure
            self.logger.error(f"❌ Failed Task - {task.title}")
            self.logger.error(f"   Error: {str(e)}")
            self.logger.error(f"   Duration: {task.result.duration:.3f}s")
        
        finally:
            self.running_tasks.discard(task.id)
            
            # Log final status
            self.logger.debug(f"📊 Status Update:")
            self.logger.debug(f"   Running: {len(self.running_tasks)}")
            self.logger.debug(f"   Completed: {len(self.completed_tasks)}")
            self.logger.debug(f"   Failed: {len(self.failed_tasks)}")
    
    
    def get_progress(self) -> Dict[str, Any]:
        """Get execution progress information."""
        total_tasks = len(self.tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        running = len(self.running_tasks)
        pending = total_tasks - completed - failed - running
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "running_tasks": running,
            "pending_tasks": pending,
            "progress_percentage": (completed / total_tasks * 100) if total_tasks > 0 else 0,
            "success_rate": (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0
        }
    
    def get_task_results(self) -> Dict[str, TaskResult]:
        """Get all task results."""
        results = {}
        for task_id, task in self.tasks.items():
            if task.result:
                results[task_id] = task.result
        return results