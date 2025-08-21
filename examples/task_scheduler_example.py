#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete workflow example using TaskScheduler and TaskExecutors
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
            logger.info(f"🔧 Starting SectionAgent for: {task.title}, Task: {task}, ")
            # Create and run SectionAgent
            agent = SectionAgent(
                name=f"section_{task.id}",
                section_info=section_info,
                report_context=report_context,
                output_format = task.section_content,
                knowledge_base_path=self.knowledge_base_path
            )
            
            agent_content = content = await agent.run_section_generation()
            
            if not content:
                content = f"[Generation failed] {task.title}: No content generated"

            content = "#" * task.level + " " + task.title + "\n" + content

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
                    "memory": agent.messages
                }
            )
            
            # Log output details
            self.log_task_output(task, task_output)
            
            return task_output
            
        except Exception as e:
            self.log_task_error(task, e)
            # Return error content instead of raising exception
            error_content = f"[Generation Error] {task.title}: {str(e)}"
            return TaskOutput(
                content=error_content,
                metadata={
                    "task_type": "generation",
                    "error": str(e),
                    "executor_id": self.executor_id
                }
            )


class MergeAgentExecutor(TaskExecutor):
    """Task executor that uses MergeAgent for content merging"""
    
    def __init__(self, knowledge_base_path: str = "workdir/documents", enable_model_merge: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base_path = knowledge_base_path
        self.enable_model_merge = enable_model_merge
    
    def can_execute(self, task: Task) -> bool:
        """Check if this executor can handle merge tasks"""
        return task.task_type == TaskType.MERGE
    
    async def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute merge task using MergeAgent"""
        task = task_input.task
        
        # Log input details
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
            
            logger.info(f"🔀 Starting MergeAgent for: {task.title}")
            logger.info(f"   Merging {len(child_contents)} child contents")
            
            if not child_contents:
                agent_content = content = f"[Merge Warning] {task.title}: No child contents to merge"
                memory = []
            else:
                # Create and run MergeAgent
                agent = MergeAgent(
                    section_info=section_info,
                    report_context=report_context,
                    child_contents=child_contents,
                    knowledge_base_path=self.knowledge_base_path,
                    enable_model_merge=self.enable_model_merge
                )
                
                agent_content = content = await agent.run_merge()
                memory = agent.messages
                
                if not content:
                    content = f"[Merge failed] {task.title}: No content generated"

            content = "#" * task.level + " " + task.title + "\n" + content

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
                    "enable_model_merge": self.enable_model_merge,
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
            # Return error content instead of raising exception
            error_content = f"[Merge Error] {task.title}: {str(e)}"
            return TaskOutput(
                content=error_content,
                metadata={
                    "task_type": "merge",
                    "error": str(e),
                    "executor_id": self.executor_id
                }
            )


async def run_markdown_template_workflow(template_path: str, max_concurrent: int = 3):
    """
    Run complete workflow with arbitrary markdown template
    
    Args:
        template_path: Path to markdown template file
        max_concurrent: Maximum concurrent tasks
    """
    print("=" * 70)
    print("基于TaskScheduler的完整工作流程示例")
    print("=" * 70)
    
    # Check template file exists
    if not Path(template_path).exists():
        print(f"❌ Template file not found: {template_path}")
        return
    
    print(f"📄 Loading template: {template_path}")
    
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
    
    # Register executors
    section_executor = SectionAgentExecutor(knowledge_base_path="workdir/documents/finance")
    merge_executor = MergeAgentExecutor(
        knowledge_base_path="workdir/documents/finance",
        enable_model_merge=False
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
    
    # Collect root tasks (top-level sections that aren't dependencies of others)
    root_tasks = []
    for task_id, task in scheduler.tasks.items():
        if task_id not in all_dependencies and task.result and task.result.content:
            root_tasks.append((task, task.result))
    
    # Sort root tasks by level first, then by document order (line number)
    root_tasks.sort(key=lambda x: (x[0].level, extract_line_number(x[0])))
    
    # Build final report with proper title
    template_name = Path(template_path).stem
    final_report = ""

    # Add content with proper structure
    for task, result in root_tasks:
        # Ensure content includes the section title if not already present
        content = result.content.strip()
        if not content.startswith('#') and task.title:
            final_report += f"## {task.title}\n\n{content}\n\n"
        else:
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
        await run_markdown_template_workflow(template1, max_concurrent=1)

if __name__ == "__main__":
    asyncio.run(main())