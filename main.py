#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器主模块
整合Agent、工具和转换器功能
"""

import asyncio
import argparse
from pathlib import Path
from typing import Optional

from app.agent.report_multi import ReportGeneratorAgent
from app.tool.knowledge_retrieval import get_knowledge_retrieval_tool
from app.converter import MarkdownConverter, ConversionRequest
from app.logger import logger
from app.config import settings

# TaskScheduler imports
from app.template.task_scheduler import TaskScheduler
from app.template.task_executors import TaskExecutor, TaskInput, TaskOutput
from app.template.task_models import Task, TaskType, TaskStatus
from app.agent.section_agent import SectionAgent
from app.agent.merge_agent import MergeAgent
from app.template import MarkdownTaskSchedule


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


class ReportGenerationSystem:
    """报告生成系统"""

    def __init__(self,
                 knowledge_base_path: str = "workdir/documents",
                 template_base_path: str = "workdir/template"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.template_base_path = Path(template_base_path)
        self.converter = MarkdownConverter()

        # 确保目录存在
        self.knowledge_base_path.mkdir(parents = True, exist_ok = True)
        self.template_base_path.mkdir(parents = True, exist_ok = True)

    async def generate_report(self,
                              template_name: str,
                              output_name: Optional[str] = None,
                              max_steps: int = 20,
                              use_schedule: bool = False,
                              confirm_execution: bool = True) -> str:
        """生成报告"""
        try:
            # 查找模板文件
            template_path = self._find_template(template_name)
            if not template_path:
                raise FileNotFoundError(f"未找到模板文件: {template_name}")

            # 设置输出路径
            if not output_name:
                output_name = f"generated_report_{template_path.stem}"

            output_path = Path("workdir") / "output" / output_name
            output_path.parent.mkdir(parents = True, exist_ok = True)

            # 创建并配置Agent
            agent = ReportGeneratorAgent(
                name = f"report_generator_{template_path.stem}",
                description = f"基于模板 {template_path.name} 生成报告",
                template_path = str(template_path),
                knowledge_base_path = str(self.knowledge_base_path),
                output_path = str(output_path),
                max_steps = max_steps,
                parallel_sections = settings.parallel_sections,
                max_concurrent = settings.max_concurrent,
            )

            logger.info(f"开始生成报告，模板: {template_path.name}")
            logger.info(f"知识库路径: {self.knowledge_base_path}")
            logger.info(f"输出路径: {output_path}")
            logger.info(f"使用调度模式: {use_schedule}")

            # 运行Agent
            if use_schedule:
                result = await self._run_with_task_scheduler(
                    str(template_path), 
                    str(output_path),
                    max_steps,
                    confirm_execution
                )
                # 获取进度信息（从scheduler中获取）
                return result
            else:
                result = await agent.run_with_template(str(template_path), str(output_path))

            # 获取进度信息
            progress = agent.get_progress()

            success_message = f"""报告生成完成！

模板: {template_path.name}
进度: {progress['completed_sections']}/{progress['total_sections']} 
      ({progress['progress_percentage']:.1f}%)
输出: {output_path}.json, {output_path}.md

详细执行结果:
{result}"""

            logger.info("报告生成任务完成")
            return success_message

        except Exception as e:
            error_message = f"报告生成失败: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)

    def _find_template(self, template_name: str) -> Optional[Path]:
        """查找模板文件"""
        # 直接路径
        if Path(template_name).exists():
            return Path(template_name)

        # 在模板目录中查找
        possible_paths = [
            self.template_base_path / template_name,
            self.template_base_path / f"{template_name}.md",
            self.template_base_path / f"{template_name}.json"
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    async def _run_with_task_scheduler(self, template_path: str, output_path: str, 
                                       max_steps: int, confirm_execution: bool) -> str:
        """使用TaskScheduler运行工作流程"""
        try:
            logger.info(f"开始使用TaskScheduler执行工作流程")
            logger.info(f"模板路径: {template_path}")
            logger.info(f"输出路径: {output_path}")
            logger.info(f"最大并发: {settings.max_concurrent}")
            
            # Check template file exists
            if not Path(template_path).exists():
                raise FileNotFoundError(f"模板文件不存在: {template_path}")
            
            # Load template and create task schedule
            schedule = MarkdownTaskSchedule(template_path, max_concurrent=settings.max_concurrent)
            
            # Get task information
            graph_info = schedule.get_task_graph_info()
            logger.info(f"任务统计信息:")
            logger.info(f"   总任务数: {graph_info['total_tasks']}")
            logger.info(f"   生成任务: {graph_info['generation_tasks']}")
            logger.info(f"   合并任务: {graph_info['merge_tasks']}")
            logger.info(f"   最大依赖深度: {graph_info.get('max_dependency_depth', 0)}")
            
            if graph_info['total_tasks'] == 0:
                raise ValueError("模板中未找到任务")
            
            # Ask for user confirmation if needed
            if confirm_execution:
                print(f"\n📋 任务队列信息:")
                print(f"   总任务数: {graph_info['total_tasks']}")
                print(f"   生成任务: {graph_info['generation_tasks']}")
                print(f"   合并任务: {graph_info['merge_tasks']}")
                print(f"   最大并发: {settings.max_concurrent}")
                
                user_input = input("\n是否继续执行? (y/N): ").strip().lower()
                if user_input not in ['y', 'yes', '是']:
                    return "用户取消了任务执行"
            
            # Create task scheduler
            scheduler = TaskScheduler(max_concurrent=settings.max_concurrent)
            
            # Register executors with detected knowledge base path
            section_executor = SectionAgentExecutor(
                knowledge_base_path=str(self.knowledge_base_path),
                executor_id="section_executor_1"
            )
            merge_executor = MergeAgentExecutor(
                executor_id="merge_executor_1"
            )
            
            scheduler.register_executor(section_executor)
            scheduler.register_executor(merge_executor)
            
            # Set global context for all executors
            template_name = Path(template_path).stem
            scheduler.set_global_context({
                "report_title": f"基于{template_name}的报告",
                "report_type": "template_generated",
                "template_path": template_path
            })
            
            # Add all tasks to scheduler
            for task in schedule.tasks.values():
                scheduler.add_task(task)
            
            logger.info(f"开始任务执行...")
            logger.info(f"   最大并发: {settings.max_concurrent}")
            logger.info(f"   总任务数: {len(schedule.tasks)}")
            
            # Execute tasks in rounds
            round_num = 1
            max_rounds = min(max_steps, 20)  # Safety limit
            
            while round_num <= max_rounds:
                # Get ready tasks
                ready_tasks = scheduler.get_ready_tasks()
                
                if not ready_tasks:
                    # Check if all tasks are completed
                    progress = scheduler.get_progress()
                    if progress["completed_tasks"] + progress["failed_tasks"] == progress["total_tasks"]:
                        logger.info(f"所有任务已完成!")
                        break
                    else:
                        logger.warning(f"没有就绪任务但执行未完成 - 可能存在循环依赖")
                        break
                
                # Limit concurrent tasks
                available_slots = scheduler.max_concurrent - len(scheduler.running_tasks)
                tasks_to_execute = ready_tasks[:available_slots]
                
                logger.info(f"第{round_num}轮: 执行{len(tasks_to_execute)}个任务")
                for task in tasks_to_execute:
                    icon = "🔧" if task.task_type == TaskType.GENERATION else "🔀"
                    logger.info(f"   {icon} {task.title} (Level {task.level})")
                
                # Execute tasks concurrently
                execution_tasks = []
                for task in tasks_to_execute:
                    execution_tasks.append(scheduler.execute_task(task))
                
                # Wait for completion
                if execution_tasks:
                    await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Show progress
                progress = scheduler.get_progress()
                logger.info(f"   进度: {progress['completed_tasks']}/{progress['total_tasks']} 已完成")
                logger.info(f"   成功率: {progress['success_rate']:.1f}%")
                
                round_num += 1
                
                # Short delay between rounds
                await asyncio.sleep(0.1)
            
            # Final statistics
            progress = scheduler.get_progress()
            logger.info(f"最终统计:")
            logger.info(f"   总任务数: {progress['total_tasks']}")
            logger.info(f"   已完成: {progress['completed_tasks']}")
            logger.info(f"   失败: {progress['failed_tasks']}")
            logger.info(f"   成功率: {progress['success_rate']:.1f}%")
            logger.info(f"   执行轮次: {round_num - 1}")
            
            # Generate final report
            final_report_path = await self._generate_final_report(scheduler, template_path, output_path)
            
            success_message = f"""TaskScheduler报告生成完成！

模板: {Path(template_path).name}
进度: {progress['completed_tasks']}/{progress['total_tasks']} 
      ({progress['success_rate']:.1f}%)
输出: {final_report_path}
执行轮次: {round_num - 1}

详细执行结果:
- 总任务数: {progress['total_tasks']}
- 已完成: {progress['completed_tasks']}
- 失败: {progress['failed_tasks']}
- 成功率: {progress['success_rate']:.1f}%"""

            return success_message
            
        except Exception as e:
            error_message = f"TaskScheduler工作流程执行失败: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)

    async def _generate_final_report(self, scheduler: TaskScheduler, template_path: str, output_path: str) -> str:
        """从任务结果生成最终报告"""
        logger.info(f"生成最终报告...")
        
        # Get all task results
        task_results = scheduler.get_task_results()
        
        if not task_results:
            logger.warning("没有可用的任务结果")
            return None
        
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
        
        logger.info(f"找到{len(root_tasks)}个根级任务并有内容")
        if len(root_tasks) == 0:
            logger.warning("没有找到根级任务且有内容的!")
            # Debug: show all task statuses
            status_summary = {}
            for task in scheduler.tasks.values():
                status = str(task.status)
                status_summary[status] = status_summary.get(status, 0) + 1
            logger.info(f"任务状态摘要: {status_summary}")
            return None
        
        # Sort root tasks by level first, then by document order (line number)
        root_tasks.sort(key=lambda x: (x[0].level, extract_line_number(x[0])))
        
        # Build final report
        template_name = Path(template_path).stem
        final_report = ""

        # Add content with proper structure
        for task, result in root_tasks:
            # The content from agents already includes proper headings, just add it directly
            content = result.content.strip()
            if content:
                final_report += f"{content}\n\n"
        
        # Save report as markdown
        output_md_path = f"{output_path}.md"
        Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        # Also save detailed task results as JSON
        output_json_path = f"{output_path}.json"
        import json
        detailed_results = {}
        for task, result in root_tasks:
            # Clean metadata to remove non-serializable objects
            clean_metadata = {}
            for key, value in result.metadata.items():
                if key == "memory":
                    # Convert memory messages to serializable format
                    if value and isinstance(value, list):
                        clean_metadata[key] = [
                            {
                                "role": getattr(msg, 'role', 'unknown'),
                                "content": str(getattr(msg, 'content', ''))
                            } for msg in value
                        ]
                    else:
                        clean_metadata[key] = []
                else:
                    try:
                        json.dumps(value)  # Test if value is JSON serializable
                        clean_metadata[key] = value
                    except (TypeError, ValueError):
                        clean_metadata[key] = str(value)  # Convert to string if not serializable
            
            detailed_results[task.id] = {
                "title": task.title,
                "level": task.level,
                "task_type": task.task_type.value,
                "status": task.status.value,
                "content": result.content,
                "metadata": clean_metadata
            }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"最终报告已保存: {output_md_path}")
        logger.info(f"详细结果已保存: {output_json_path}")
        logger.info(f"报告长度: {len(final_report)} 字符")
        
        return output_md_path

    async def convert_template(self,
                               template_path: str,
                               output_format: str = "json") -> str:
        """转换模板格式"""
        try:
            template_file = Path(template_path)
            if not template_file.exists():
                raise FileNotFoundError(f"模板文件不存在: {template_path}")

            # 读取模板内容
            template_content = template_file.read_text(encoding = 'utf-8')

            # 确定源格式
            source_format = "markdown" if template_content.strip().startswith('#') else "json"

            # 执行转换
            conversion_request = ConversionRequest(
                source_format = source_format,
                target_format = output_format,
                content = template_content
            )

            result = self.converter.convert(conversion_request)

            if result.success:
                # 保存转换结果
                output_path = template_file.with_suffix(f'.{output_format}')
                if output_format == "json":
                    import json
                    with open(output_path, 'w', encoding = 'utf-8') as f:
                        json.dump(result.result, f, ensure_ascii = False, indent = 2)
                else:
                    with open(output_path, 'w', encoding = 'utf-8') as f:
                        f.write(result.result)

                return f"模板转换成功: {output_path}"
            else:
                raise Exception(result.error)

        except Exception as e:
            error_message = f"模板转换失败: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)

    async def test_knowledge_base(self, query: str = "智能鞋垫") -> str:
        """测试知识库检索"""
        try:
            knowledge_tool = get_knowledge_retrieval_tool(str(self.knowledge_base_path))
            result = await knowledge_tool.execute(query = query, threshold=settings.distance, top_k = settings.top_k)

            if result.error:
                return f"知识库检索测试失败: {result.error}"
            else:
                stats = knowledge_tool.get_statistics()
                return f"""知识库检索测试成功！

统计信息:
- 文档总数: {stats['total_documents']}
- 总大小: {stats['total_size']} 字符
- 平均大小: {stats['average_size']} 字符
- 文件类型: {stats['file_types']}

检索结果 (查询: "{query}"):
{result.output}"""

        except Exception as e:
            error_message = f"知识库测试失败: {str(e)}"
            logger.error(error_message)
            return error_message

    def list_templates(self) -> str:
        """列出可用模板"""
        templates = []

        if self.template_base_path.exists():
            for file_path in self.template_base_path.iterdir():
                if file_path.is_file() and file_path.suffix in ['.md', '.json']:
                    templates.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "type": file_path.suffix[1:]
                    })

        if not templates:
            return "未找到可用模板"

        result = "可用模板:\n"
        for i, template in enumerate(templates, 1):
            result += f"{i}. {template['name']} ({template['type']}, {template['size']} bytes)\n"

        return result


async def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description = "智能报告生成器")
    parser.add_argument("action", choices = ["generate", "convert", "test", "list"],
                        help = "执行的操作")
    parser.add_argument("--template", "-t", help = "模板名称或路径")
    parser.add_argument("--output", "-o", help = "输出文件名")
    parser.add_argument("--format", "-f", default = "json",
                        choices = ["json", "markdown"], help = "转换目标格式")
    parser.add_argument("--query", "-q", default = "智能鞋垫", help = "测试查询")
    parser.add_argument("--max-steps", type = int, default = 200, help = "最大执行步数")
    parser.add_argument("--knowledge-base", default = "workdir/documents",
                        help = "知识库路径")
    parser.add_argument("--template-base", default = "workdir/template",
                        help = "模板基础路径")
    parser.add_argument("--schedule", action="store_true", 
                        help = "使用任务调度模式")
    parser.add_argument("--no-confirm", action="store_true",
                        help = "跳过用户确认（自动执行）")

    args = parser.parse_args()

    # 创建报告生成系统
    system = ReportGenerationSystem(
        knowledge_base_path = args.knowledge_base,
        template_base_path = args.template_base
    )

    try:
        if args.action == "generate":
            if not args.template:
                print("错误: 生成报告需要指定模板 (--template)")
                return

            result = await system.generate_report(
                template_name = args.template,
                output_name = args.output,
                max_steps = args.max_steps,
                use_schedule = args.schedule,
                confirm_execution = not args.no_confirm
            )
            print(result)

        elif args.action == "convert":
            if not args.template:
                print("错误: 转换模板需要指定模板路径 (--template)")
                return

            result = await system.convert_template(
                template_path = args.template,
                output_format = args.format
            )
            print(result)

        elif args.action == "test":
            result = await system.test_knowledge_base(query = args.query)
            print(result)

        elif args.action == "list":
            result = system.list_templates()
            print(result)

    except Exception as e:
        print(f"执行失败: {e}")
        logger.error(f"命令执行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())