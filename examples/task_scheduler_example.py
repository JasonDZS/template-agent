#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete workflow example.txt using TaskScheduler and TaskExecutors
Supports arbitrary markdown template input with SectionAgent and MergeAgent integration
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.template.task_scheduler import TaskScheduler
from app.template.task_executors import TaskExecutor, TaskInput, TaskOutput
from app.template.task_models import Task, TaskType, TaskStatus
from app.agent.section_agent import SectionAgent
from app.agent.merge_agent import MergeAgent
from app.template import MarkdownTaskSchedule
from app.logger import logger


class SectionAgentExecutor(TaskExecutor):
    """Task executor that uses SectionAgent for content generation"""
    
    def __init__(self, knowledge_base_path: str = "workdir/documents", **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base_path = knowledge_base_path
    
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle generation tasks"""
        return task.task_type == TaskType.GENERATION
    
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute generation task using SectionAgent"""
        task = task_input.task
        
        # Log input details
        self.log_task_input(task_input)
        
        try:
            # Create section info for SectionAgent following tutorial format
            section_info = {
                "id": task.id,
                "content": task.title,
                "level": task.level,
                "description": f"Generate content for section: {task.title}"
            }
            
            # Create report context
            report_context = {
                "title": task_input.context.get("report_title", "Report"),
                "type": task_input.context.get("report_type", "document")
            }
            logger.info(f"🔧 Starting SectionAgent for: {task.title}")
            logger.info(f"   Section ID: {task.id}, Level: {task.level}")
            logger.info(f"   Knowledge base: {self.knowledge_base_path}")
            # Create and run SectionAgent with proper parameters
            agent = SectionAgent(
                section_info=section_info,
                report_context=report_context,
                output_format=task.section_content,
                knowledge_base_path=self.knowledge_base_path
            )
            
            await agent.run()
            
            if agent.is_finished():
                agent_content = content = agent.get_content()
                if not content:
                    content = f"[Generation failed] {task.title}: No content generated"
                    logger.warning(f"⚠️ SectionAgent generated empty content for {task.title}")
                else:
                    logger.info(f"✅ SectionAgent completed successfully for {task.title}")
                    logger.info(f"   Generated content length: {len(content)} characters")
            else:
                agent_content = content = f"[Generation incomplete] {task.title}: Agent did not finish successfully"
                logger.warning(f"⚠️ SectionAgent did not complete for {task.title}")

            # Add section header if not already present
            if not content.startswith('#'):
                content = "#" * task.level + " " + task.title + "\n" + content
            else:
                # Ensure proper heading level
                lines = content.split('\n', 1)
                if lines[0].startswith('#'):
                    # Replace existing heading with proper level
                    header = "#" * task.level + " " + task.title
                    content = header + ("\n" + lines[1] if len(lines) > 1 else "")

            task_output = TaskOutput(
                content=content,
                metadata={
                    "task_type": "generation",
                    "agent_type": "SectionAgent",
                    "section_title": task.title,
                    "section_level": task.level,
                    "content_length": len(content),
                    "executor_id": self.executor_id,
                    "agent_content": agent_content,
                    "memory": agent.memory.messages if hasattr(agent, 'memory') else []
                }
            )
            
            # Log output details
            self.log_task_output(task, task_output)
            
            return task_output
            
        except Exception as e:
            self.log_task_error(task, e)
            logger.error(f"❌ SectionAgent execution failed for {task.title}: {str(e)}")
            # Return error content instead of raising exception
            error_content = f"[Generation Error] {task.title}: {str(e)}"
            return TaskOutput(
                content=error_content,
                metadata={
                    "task_type": "generation",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "executor_id": self.executor_id,
                    "section_title": task.title,
                    "section_level": task.level
                }
            )


class MergeAgentExecutor(TaskExecutor):
    """Task executor that uses MergeAgent for content merging"""
    
    def __init__(self, knowledge_base_path: str = "workdir/documents", **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base_path = knowledge_base_path
    
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle merge tasks"""
        return task.task_type == TaskType.MERGE
    
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute merge task using MergeAgent"""
        task = task_input.task
        
        # Log input details
        self.log_task_input(task_input)
        
        try:
            # Create section info for MergeAgent following tutorial format  
            section_info = {
                "id": task.id,
                "content": task.title,
                "level": task.level,
                "description": f"Merge content for section: {task.title}"
            }
            
            # Create report context
            report_context = {
                "title": task_input.context.get("report_title", "Report"),
                "type": task_input.context.get("report_type", "document")
            }
            
            # Get child contents from dependencies
            child_contents = list(task_input.dependencies_content.values())
            
            logger.info(f"🔀 Starting MergeAgent for: {task.title}")
            logger.info(f"   Section ID: {task.id}, Level: {task.level}")
            logger.info(f"   Merging {len(child_contents)} child contents")
            logger.info(f"   Expected dependencies: {len(task.dependencies)}, Received: {len(task_input.dependencies_content)}")
            
            if not child_contents:
                agent_content = content = f"[Merge Warning] {task.title}: No child contents to merge"
                memory = []
                logger.warning(f"⚠️ No child contents available for merging {task.title}")
            else:
                # Create and run MergeAgent (remove knowledge_base_path as it's not supported)
                agent = MergeAgent(
                    section_info=section_info,
                    report_context=report_context,
                    child_contents=child_contents,
                    output_format=task.section_content if hasattr(task, 'section_content') else None
                )
                
                await agent.run()
                
                if agent.is_finished():
                    agent_content = content = agent.get_content()
                    memory = agent.memory.messages if hasattr(agent, 'memory') else []
                    
                    if not content:
                        content = f"[Merge failed] {task.title}: No content generated"
                        logger.warning(f"⚠️ MergeAgent generated empty content for {task.title}")
                    else:
                        logger.info(f"✅ MergeAgent completed successfully for {task.title}")
                        logger.info(f"   Merged content length: {len(content)} characters")
                        logger.info(f"   Successfully merged {len(child_contents)} child contents")
                else:
                    agent_content = content = f"[Merge incomplete] {task.title}: Agent did not finish successfully"
                    memory = []
                    logger.warning(f"⚠️ MergeAgent did not complete for {task.title}")

            # Add section header if not already present (for merge tasks)
            if not content.startswith('#'):
                content = "#" * task.level + " " + task.title + "\n" + content
            else:
                # Ensure proper heading level
                lines = content.split('\n', 1)
                if lines[0].startswith('#'):
                    # Replace existing heading with proper level
                    header = "#" * task.level + " " + task.title
                    content = header + ("\n" + lines[1] if len(lines) > 1 else "")

            task_output = TaskOutput(
                content=content,
                metadata={
                    "task_type": "merge",
                    "agent_type": "MergeAgent",
                    "section_title": task.title,
                    "section_level": task.level,
                    "child_count": len(child_contents),
                    "expected_dependencies": len(task.dependencies),
                    "received_dependencies": len(task_input.dependencies_content),
                    "content_length": len(content),
                    "executor_id": self.executor_id,
                    "agent_content": agent_content,
                    "memory": memory
                }
            )
            
            # Log output details
            self.log_task_output(task, task_output)
            
            return task_output
            
        except Exception as e:
            self.log_task_error(task, e)
            logger.error(f"❌ MergeAgent execution failed for {task.title}: {str(e)}")
            # Return error content instead of raising exception
            error_content = f"[Merge Error] {task.title}: {str(e)}"
            return TaskOutput(
                content=error_content,
                metadata={
                    "task_type": "merge",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "executor_id": self.executor_id,
                    "section_title": task.title,
                    "section_level": task.level,
                    "child_count": len(child_contents) if 'child_contents' in locals() else 0
                }
            )


async def run_markdown_template_workflow(template_path: str, max_concurrent: int = 3, knowledge_base_path: str = None):
    """
    Run complete workflow with arbitrary markdown template
    
    Args:
        template_path: Path to markdown template file
        max_concurrent: Maximum concurrent tasks
        knowledge_base_path: Optional custom knowledge base path
    """
    print("=" * 70)
    print("基于TaskScheduler的完整工作流程示例")
    print("=" * 70)
    
    # Check template file exists
    if not Path(template_path).exists():
        print(f"❌ Template file not found: {template_path}")
        return
    
    print(f"📄 Loading template: {template_path}")
    
    # Auto-detect knowledge base path if not provided
    if knowledge_base_path is None:
        knowledge_base_path = "workdir/documents"
        if Path("workdir/finance/documents").exists():
            knowledge_base_path = "workdir/finance/documents"
        elif Path("workdir/documents").exists():
            knowledge_base_path = "workdir/documents"
    
    print(f"📚 Using knowledge base: {knowledge_base_path}")
    
    # Load template and create task schedule
    schedule = MarkdownTaskSchedule(template_path, max_concurrent=max_concurrent)
    
    # Get task information
    graph_info = schedule.get_task_graph_info()
    print(f"📊 Task Statistics:")
    print(f"   Total tasks: {graph_info['total_tasks']}")
    print(f"   Generation tasks: {graph_info['generation_tasks']}")
    print(f"   Merge tasks: {graph_info['merge_tasks']}")
    print(f"   Max dependency depth: {graph_info.get('max_dependency_depth', 0)}")
    
    if graph_info['total_tasks'] == 0:
        print("❌ No tasks found in template")
        return
    
    # Create task scheduler
    scheduler = TaskScheduler(max_concurrent=max_concurrent)
    
    # Register executors with detected knowledge base path
    section_executor = SectionAgentExecutor(
        knowledge_base_path=knowledge_base_path,
        executor_id="section_executor_1"
    )
    merge_executor = MergeAgentExecutor(
        executor_id="merge_executor_1"
    )
    
    scheduler.register_executor(section_executor)
    scheduler.register_executor(merge_executor)
    
    # Set global context for all executors
    scheduler.set_global_context({
        "report_title": f"基于{Path(template_path).stem}的报告",
        "report_type": "template_generated",
        "template_path": template_path
    })
    
    # Add all tasks to scheduler
    for task in schedule.tasks.values():
        scheduler.add_task(task)
    
    print(f"\n🚀 Starting task execution...")
    print(f"   Max concurrent: {max_concurrent}")
    print(f"   Total tasks: {len(schedule.tasks)}")
    
    # Execute tasks in rounds
    round_num = 1
    max_rounds = 20  # Safety limit
    
    while round_num <= max_rounds:
        # Get ready tasks
        ready_tasks = scheduler.get_ready_tasks()
        
        if not ready_tasks:
            # Check if all tasks are completed
            progress = scheduler.get_progress()
            if progress["completed_tasks"] + progress["failed_tasks"] == progress["total_tasks"]:
                print(f"\n✅ All tasks completed!")
                break
            else:
                print(f"\n⚠️  No ready tasks but execution not complete - possible circular dependency")
                break
        
        # Limit concurrent tasks
        available_slots = scheduler.max_concurrent - len(scheduler.running_tasks)
        tasks_to_execute = ready_tasks[:available_slots]
        
        print(f"\n📋 Round {round_num}: Executing {len(tasks_to_execute)} tasks")
        for task in tasks_to_execute:
            icon = "🔧" if task.task_type == TaskType.GENERATION else "🔀"
            print(f"   {icon} {task.title} (Level {task.level})")
        
        # Execute tasks concurrently
        execution_tasks = []
        for task in tasks_to_execute:
            execution_tasks.append(scheduler.execute_task(task))
        
        # Wait for completion
        if execution_tasks:
            await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Show progress
        progress = scheduler.get_progress()
        print(f"   Progress: {progress['completed_tasks']}/{progress['total_tasks']} completed")
        print(f"   Success rate: {progress['success_rate']:.1f}%")
        
        round_num += 1
        
        # Short delay between rounds
        await asyncio.sleep(0.1)
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    progress = scheduler.get_progress()
    print(f"   Total tasks: {progress['total_tasks']}")
    print(f"   Completed: {progress['completed_tasks']}")
    print(f"   Failed: {progress['failed_tasks']}")
    print(f"   Success rate: {progress['success_rate']:.1f}%")
    print(f"   Execution rounds: {round_num - 1}")
    
    # Generate final report
    await generate_final_report(scheduler, template_path)
    
    return scheduler


async def generate_final_report(scheduler: TaskScheduler, template_path: str):
    """Generate final report from task results"""
    print(f"\n📄 Generating final report...")
    
    # Get all task results
    task_results = scheduler.get_task_results()
    
    if not task_results:
        print("❌ No task results available")
        return
    
    # Find root tasks (tasks with no dependents) and sort by document order
    all_dependencies = set()
    for task in scheduler.tasks.values():
        all_dependencies.update(task.dependencies)
    
    # Extract heading node line number for proper ordering
    def extract_line_number(task_obj):
        try:
            heading_node = task_obj.heading_node
            if hasattr(heading_node, 'attributes') and heading_node.attributes:
                line_num = heading_node.attributes.get('line_number', 0)
                return line_num
            return 0
        except (AttributeError, KeyError):
            return 0
    
    # Find root tasks (tasks with no dependents) and sort by document order
    all_dependencies = set()
    for task in scheduler.tasks.values():
        all_dependencies.update(task.dependencies)
    
    # Collect only root tasks with results
    root_tasks = []
    for task_id, task in scheduler.tasks.items():
        if (task_id not in all_dependencies and 
            task.result and task.result.content):
            root_tasks.append((task, task.result))
    
    print(f"📋 Found {len(root_tasks)} root tasks with content")
    if len(root_tasks) == 0:
        print("⚠️  No root tasks with content found!")
        # Debug: show all task statuses
        status_summary = {}
        for task in scheduler.tasks.values():
            status = str(task.status)
            status_summary[status] = status_summary.get(status, 0) + 1
        print(f"📊 Task status summary: {status_summary}")
        return
    
    # Sort root tasks by level first, then by document order (line number)
    root_tasks.sort(key=lambda x: (x[0].level, extract_line_number(x[0])))
    
    # Build final report with proper title
    template_name = Path(template_path).stem
    final_report = ""

    # Add content with proper structure
    for task, result in root_tasks:
        # The content from executors already includes proper headings, just add it directly
        content = result.content.strip()
        if content:
            final_report += f"{content}\n\n"
    
    # Save report
    output_path = f"workdir/output/task_scheduler_report_{template_name}.md"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    print(f"✅ Final report saved: {output_path}")
    print(f"📊 Report length: {len(final_report)} characters")
    
    # Show preview
    print(f"\n📖 Report preview (first 500 characters):")
    print("-" * 50)
    print(final_report[:500])
    if len(final_report) > 500:
        print("...")


async def main():
    """Main function demonstrating different template examples"""
    print("TaskScheduler + TaskExecutor 工作流程演示")
    print("支持任意markdown模板输入")
    
    # Example 1: 企业信贷评估模板
    template1 = "workdir/template/企业信贷评估模板.md"
    if Path(template1).exists():
        print(f"\n🎯 示例1: 企业信贷评估模板")
        await run_markdown_template_workflow(
            template1, 
            max_concurrent=1,
            knowledge_base_path="workdir/finance/documents"
        )

if __name__ == "__main__":
    asyncio.run(main())