"""Document Task Scheduler for Template Agent.

This module provides markdown document-specific scheduling functionality:
- Converting markdown documents to task graphs
- Task graph visualization and analysis
- Document-aware task execution
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .document_types import MarkdownDocument, HeadingNode
from .converters import parse_markdown_file_to_document_tree
from .task_models import Task, TaskResult, TaskType, TaskStatus
from .task_scheduler import TaskScheduler


class MarkdownTaskSchedule:
    """Task schedule generator for markdown documents."""
    
    def __init__(self, markdown_path: str, max_concurrent: int = 3):
        """Initialize markdown task schedule.
        
        Args:
            markdown_path: Path to markdown file
            max_concurrent: Maximum concurrent tasks
        """
        self.markdown_path = Path(markdown_path)
        self.max_concurrent = max_concurrent
        
        # Parse document
        self.document = parse_markdown_file_to_document_tree(str(self.markdown_path))
        
        # Task management
        self.scheduler = TaskScheduler(max_concurrent=max_concurrent)
        self.tasks: Dict[str, Task] = {}
        self.heading_to_task: Dict[str, str] = {}  # heading_id -> task_id mapping
        
        # Build task graph
        self._build_task_graph()
    
    def _build_task_graph(self) -> None:
        """Build task graph from document structure."""
        all_headings = self.document.get_all_headings()
        leaf_headings = self.document.get_leaf_headings()
        
        # Create generation tasks for leaf headings
        for heading in leaf_headings:
            task = self._create_generation_task(heading)
            self.tasks[task.id] = task
            self.scheduler.add_task(task)
            self.heading_to_task[heading.id] = task.id
        
        # Create merge tasks for non-leaf headings
        for heading in all_headings:
            if not heading.is_heading_leaf():
                task = self._create_merge_task(heading)
                if task:  # Only add if has dependencies
                    self.tasks[task.id] = task
                    self.scheduler.add_task(task)
                    self.heading_to_task[heading.id] = task.id
    
    def _create_generation_task(self, heading: HeadingNode) -> Task:
        """Create generation task for leaf heading."""
        task_id = f"gen_{heading.id}"
        
        return Task(
            id=task_id,
            task_type=TaskType.GENERATION,
            heading_node=heading,
            title=heading.content or f"Section {heading.level}",
            level=heading.level,
            section_content=heading.children_content(),
            priority=100 - heading.level,  # Higher level = higher priority
            estimated_duration=30.0  # Default 30 seconds

        )
    
    def _create_merge_task(self, heading: HeadingNode) -> Optional[Task]:
        """Create merge task for non-leaf heading."""
        child_headings = heading.get_child_headings()
        if not child_headings:
            return None
        
        task_id = f"merge_{heading.id}"
        dependencies = []  # List to maintain order
        
        # Sort child headings by their order in the document
        sorted_children = sorted(child_headings, key=lambda h: h.order)
        
        # Find dependencies (child tasks) in correct order
        for child in sorted_children:
            child_task_id = self.heading_to_task.get(child.id)
            if child_task_id:
                dependencies.append(child_task_id)
            else:
                # Child might be processed later, use child heading ID
                potential_gen_id = f"gen_{child.id}"
                potential_merge_id = f"merge_{child.id}"
                
                # Check which type of task this child will be
                if child.is_heading_leaf():
                    dependencies.append(potential_gen_id)
                else:
                    dependencies.append(potential_merge_id)
        
        if not dependencies:
            return None
        
        task = Task(
            id=task_id,
            task_type=TaskType.MERGE,
            heading_node=heading,
            title=heading.content or f"Merge {heading.level}",
            level=heading.level,
            dependencies=dependencies,
            priority=50 - heading.level,  # Lower priority than generation
            estimated_duration=10.0  # Merge is faster
        )
        
        # Update dependent relationships
        for dep_id in dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task:
                dep_task.dependents.add(task_id)
        
        return task
    
    def set_handlers(self, 
                    generation_handler: Callable[[Task], str],
                    merge_handler: Callable[[Task, List[str]], str]) -> None:
        """Set task execution handlers."""
        self.scheduler.set_generation_handler(generation_handler)
        self.scheduler.set_merge_handler(merge_handler)
    
    async def execute_all(self) -> Dict[str, TaskResult]:
        """Execute all tasks in dependency order."""
        while not self._is_complete():
            # Get ready tasks
            ready_tasks = self.scheduler.get_ready_tasks()
            
            # Schedule tasks up to concurrency limit
            tasks_to_run = []
            for task in ready_tasks:
                if self.scheduler.can_schedule_more():
                    tasks_to_run.append(self.scheduler.execute_task(task))
                else:
                    break
            
            if not tasks_to_run:
                # No ready tasks and no running tasks = deadlock or completion
                if not self.scheduler.running_tasks:
                    break
                # Wait a bit for running tasks
                await asyncio.sleep(0.1)
                continue
            
            # Run tasks concurrently
            await asyncio.gather(*tasks_to_run, return_exceptions=True)
        
        return self.scheduler.get_task_results()
    
    def _is_complete(self) -> bool:
        """Check if all tasks are complete."""
        total_tasks = len(self.tasks)
        finished_tasks = len(self.scheduler.completed_tasks) + len(self.scheduler.failed_tasks)
        return finished_tasks >= total_tasks
    
    def get_task_graph_info(self) -> Dict[str, Any]:
        """Get information about the task graph."""
        generation_tasks = [t for t in self.tasks.values() if t.task_type == TaskType.GENERATION]
        merge_tasks = [t for t in self.tasks.values() if t.task_type == TaskType.MERGE]
        
        # Calculate graph statistics
        max_depth = max([len(t.dependencies) for t in self.tasks.values()], default=0)
        avg_dependencies = sum([len(t.dependencies) for t in self.tasks.values()]) / len(self.tasks) if self.tasks else 0
        
        return {
            "total_tasks": len(self.tasks),
            "generation_tasks": len(generation_tasks),
            "merge_tasks": len(merge_tasks),
            "max_dependency_depth": max_depth,
            "avg_dependencies_per_task": avg_dependencies,
            "leaf_headings": len(self.document.get_leaf_headings()),
            "total_headings": len(self.document.get_all_headings()),
            "document_title": self.document.title,
            "tasks_by_level": self._get_tasks_by_level()
        }
    
    def _get_tasks_by_level(self) -> Dict[int, Dict[str, int]]:
        """Get task distribution by heading level."""
        level_stats = {}
        for task in self.tasks.values():
            level = task.level
            if level not in level_stats:
                level_stats[level] = {"generation": 0, "merge": 0}
            
            if task.task_type == TaskType.GENERATION:
                level_stats[level]["generation"] += 1
            else:
                level_stats[level]["merge"] += 1
        
        return level_stats
    
    def get_progress(self) -> Dict[str, Any]:
        """Get execution progress."""
        base_progress = self.scheduler.get_progress()
        
        # Add task graph specific information
        base_progress.update({
            "document_title": self.document.title,
            "task_graph_info": self.get_task_graph_info(),
            "estimated_total_duration": sum([t.estimated_duration for t in self.tasks.values()]),
            "actual_duration": sum([
                t.result.duration for t in self.tasks.values() 
                if t.result and t.result.duration
            ] or [0.0])
        })
        
        return base_progress
    
    def visualize_task_graph(self) -> str:
        """Generate a text visualization of the task graph."""
        lines = []
        lines.append(f"Task Graph for: {self.document.title}")
        lines.append("=" * 50)
        
        # Group tasks by level
        levels = {}
        for task in self.tasks.values():
            level = task.level
            if level not in levels:
                levels[level] = []
            levels[level].append(task)
        
        # Display by level
        for level in sorted(levels.keys()):
            lines.append(f"\nLevel {level}:")
            for task in levels[level]:
                task_type_symbol = "🔧" if task.task_type == TaskType.GENERATION else "🔀"
                status_symbol = {
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.READY: "✅",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌",
                    TaskStatus.CANCELLED: "🚫"
                }.get(task.status, "❓")
                
                deps_info = f" (depends on: {len(task.dependencies)})" if task.dependencies else ""
                lines.append(f"  {status_symbol} {task_type_symbol} {task.title}{deps_info}")
                
                if task.dependencies:
                    for dep_id in task.dependencies:
                        dep_task = self.tasks.get(dep_id)
                        if dep_task:
                            lines.append(f"    ⬅️ {dep_task.title}")
        
        return "\n".join(lines)


def create_task_schedule(markdown_path: str, max_concurrent: int = 3) -> MarkdownTaskSchedule:
    """Create a task schedule from markdown file.
    
    Args:
        markdown_path: Path to markdown file
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        MarkdownTaskSchedule instance
    """
    return MarkdownTaskSchedule(markdown_path, max_concurrent)