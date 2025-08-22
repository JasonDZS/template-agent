"""Task Executors for Template Agent.

This module provides task execution functionality including:
- Abstract base executor interface
- Generation task executor for leaf headings
- Merge task executor for combining content
- Executor registry for managing multiple executors
"""

from abc import ABC, abstractmethod
import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union

from .task_models import (
    Task, TaskInput, TaskOutput, TaskResult, TaskStatus, TaskType
)


class TaskExecutor(ABC):
    """Base class for task executors that handle task execution logic.
    
    TaskExecutor defines the interface for executing different types of tasks,
    handling inputs/outputs, and managing execution context.
    """
    
    def __init__(self, executor_id: str = None, logger: Optional[logging.Logger] = None):
        self.executor_id = executor_id or str(uuid.uuid4())
        self.execution_context: Dict[str, Any] = {}
        self.logger = logger or logging.getLogger(f"{self.__class__.__name__}[{self.executor_id[:8]}]")
    
    @abstractmethod
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute a task with given input.
        
        Args:
            task_input: Input data for task execution
            
        Returns:
            TaskOutput with execution results
            
        Raises:
            Exception: If task execution fails
        """
        pass
    
    @abstractmethod
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle the given task.
        
        Args:
            task: Task to check
            
        Returns:
            True if executor can handle the task, False otherwise
        """
        pass
    
    def prepare_input(self, task: Task, dependencies_results: Dict[str, TaskResult]) -> TaskInput:
        """Prepare input for task execution.
        
        Args:
            task: Task to prepare input for
            dependencies_results: Results from dependency tasks
            
        Returns:
            Prepared TaskInput
        """
        dependencies_content = {}
        for dep_id, result in dependencies_results.items():
            if result.status == TaskStatus.COMPLETED and result.content:
                dependencies_content[dep_id] = result.content
        
        return TaskInput(
            task=task,
            dependencies_content=dependencies_content,
            context=self.execution_context.copy()
        )
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set execution context.
        
        Args:
            context: Context data for execution
        """
        self.execution_context.update(context)
    
    def log_task_input(self, task_input: TaskInput) -> None:
        """Log task input details.
        
        Args:
            task_input: Input data for task execution
        """
        task = task_input.task
        
        self.logger.info(f"📥 Task Input - {task.title} ({task.task_type.value})")
        self.logger.info(f"   Task ID: {task.id}")
        self.logger.info(f"   Task Level: {task.level}")
        self.logger.info(f"   Dependencies: {len(task.dependencies)}")
        self.logger.info(f"   Dependencies Content: {len(task_input.dependencies_content)}")
        self.logger.info(f"   Section Content Length: {len(task.section_content)}")
        
        if task_input.dependencies_content:
            self.logger.debug("   📋 Dependencies Details:")
            for i, (dep_id, content) in enumerate(task_input.dependencies_content.items(), 1):
                content_preview = content[:100].replace('\n', ' ') if content else ""
                self.logger.debug(f"      {i}. {dep_id[:8]}... ({len(content)} chars): {content_preview}...")
        
        if task_input.context:
            self.logger.debug(f"   🎛️  Context: {task_input.context}")
    
    def log_task_output(self, task: Task, task_output: TaskOutput, duration: float = None) -> None:
        """Log task output details.
        
        Args:
            task: The executed task
            task_output: Output data from task execution  
            duration: Execution duration in seconds
        """
        self.logger.info(f"📤 Task Output - {task.title} ({task.task_type.value})")
        self.logger.info(f"   Output Content Length: {len(task_output.content)}")
        self.logger.info(f"   Output Metadata: {task_output.metadata}")
        
        if duration is not None:
            self.logger.info(f"   Execution Duration: {duration:.3f}s")
        
        if task_output.artifacts:
            self.logger.debug(f"   🎁 Artifacts: {task_output.artifacts}")
        
        # Log content preview
        content_preview = task_output.content[:200].replace('\n', ' ') if task_output.content else ""
        self.logger.debug(f"   📄 Content Preview: {content_preview}...")
    
    def log_task_error(self, task: Task, error: Exception, duration: float = None) -> None:
        """Log task execution error.
        
        Args:
            task: The failed task
            error: The exception that occurred
            duration: Execution duration in seconds before failure
        """
        self.logger.error(f"❌ Task Error - {task.title} ({task.task_type.value})")
        self.logger.error(f"   Task ID: {task.id}")
        self.logger.error(f"   Error Type: {type(error).__name__}")
        self.logger.error(f"   Error Message: {str(error)}")
        
        if duration is not None:
            self.logger.error(f"   Failed After: {duration:.3f}s")


class GenerationTaskExecutor(TaskExecutor):
    """Task executor for generation tasks that create content for leaf headings."""
    
    def __init__(self, generation_handler: Callable[[Task], Union[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.generation_handler = generation_handler
    
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle generation tasks."""
        return task.task_type == TaskType.GENERATION
    
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute generation task.
        
        Args:
            task_input: Input data containing task and context
            
        Returns:
            TaskOutput with generated content
            
        Raises:
            RuntimeError: If no generation handler is set
            Exception: If generation fails
        """
        if not self.generation_handler:
            raise RuntimeError("No generation handler set for GenerationTaskExecutor")
        
        task = task_input.task
        start_time = datetime.now()
        
        # Log input details
        self.log_task_input(task_input)
        
        try:
            # Call generation handler (could be async or sync)
            if asyncio.iscoroutinefunction(self.generation_handler):
                content = await self.generation_handler(task)
            else:
                content = self.generation_handler(task)
            
            # Convert result to string if needed
            if not isinstance(content, str):
                content = str(content)
            
            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            task_output = TaskOutput(
                content=content,
                metadata={
                    "task_type": "generation",
                    "heading_level": task.level,
                    "heading_title": task.title,
                    "section_length": len(task.section_content),
                    "execution_time": duration,
                    "executor_id": self.executor_id
                }
            )
            
            # Log output details
            self.log_task_output(task, task_output, duration)
            
            return task_output
            
        except Exception as e:
            # Calculate failure time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log error details
            self.log_task_error(task, e, duration)
            
            raise Exception(f"Generation task failed for {task.title}: {str(e)}")


class MergeTaskExecutor(TaskExecutor):
    """Task executor for merge tasks that combine content from child tasks."""
    
    def __init__(self, merge_handler: Callable[[Task, List[str]], Union[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.merge_handler = merge_handler
    
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle merge tasks."""
        return task.task_type == TaskType.MERGE
    
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute merge task.
        
        Args:
            task_input: Input data containing task and dependencies content
            
        Returns:
            TaskOutput with merged content
            
        Raises:
            RuntimeError: If no merge handler is set
            Exception: If merge fails
        """
        if not self.merge_handler:
            raise RuntimeError("No merge handler set for MergeTaskExecutor")
        
        task = task_input.task
        start_time = datetime.now()
        
        # Log input details
        self.log_task_input(task_input)
        
        try:
            # Get content from dependencies in correct order
            child_contents = []
            for dep_id in task.dependencies:
                if dep_id in task_input.dependencies_content:
                    child_contents.append(task_input.dependencies_content[dep_id])
            
            # Log dependency analysis
            self.logger.info(f"📊 Dependency Analysis - {task.title}")
            self.logger.info(f"   Expected Dependencies: {len(task.dependencies)}")
            self.logger.info(f"   Received Content: {len(task_input.dependencies_content)}")
            self.logger.info(f"   Matched Dependencies: {len(child_contents)}")
            
            if len(child_contents) != len(task.dependencies):
                missing_deps = set(task.dependencies) - set(task_input.dependencies_content.keys())
                self.logger.warning(f"   ⚠️  Missing Dependencies: {missing_deps}")
            
            # Call merge handler (could be async or sync)
            if asyncio.iscoroutinefunction(self.merge_handler):
                content = await self.merge_handler(task, child_contents)
            else:
                content = self.merge_handler(task, child_contents)
            
            # Convert result to string if needed
            if not isinstance(content, str):
                content = str(content)
            
            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            task_output = TaskOutput(
                content=content,
                metadata={
                    "task_type": "merge",
                    "heading_level": task.level,
                    "heading_title": task.title,
                    "child_count": len(child_contents),
                    "expected_dependencies": len(task.dependencies),
                    "received_dependencies": len(task_input.dependencies_content),
                    "dependencies": task.dependencies,
                    "execution_time": duration,
                    "executor_id": self.executor_id
                }
            )
            
            # Log output details
            self.log_task_output(task, task_output, duration)
            
            return task_output
            
        except Exception as e:
            # Calculate failure time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log error details
            self.log_task_error(task, e, duration)
            
            raise Exception(f"Merge task failed for {task.title}: {str(e)}")


class ExecutorRegistry:
    """Registry for managing multiple task executors."""
    
    def __init__(self):
        self.executors: List[TaskExecutor] = []
    
    def register(self, executor: TaskExecutor) -> None:
        """Register a task executor.
        
        Args:
            executor: TaskExecutor to register
        """
        self.executors.append(executor)
    
    def get_executor(self, task: Task) -> Optional[TaskExecutor]:
        """Get appropriate executor for a task.
        
        Args:
            task: Task to find executor for
            
        Returns:
            TaskExecutor that can handle the task, or None
        """
        for executor in self.executors:
            if executor.can_execute(task):
                return executor
        return None
    
    def set_global_context(self, context: Dict[str, Any]) -> None:
        """Set context for all registered executors.
        
        Args:
            context: Context data to set
        """
        for executor in self.executors:
            executor.set_context(context)