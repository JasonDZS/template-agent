#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Section Report Generator Agent

This module provides a sophisticated agent for generating structured reports
by coordinating multiple section agents. It supports parallel and sequential
processing modes, content quality assessment, and automatic polishing.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base import BaseAgent
from .section_agent_react import SectionAgentReAct
from .section_agent import SectionAgent
from ..config import settings
from ..tool.knowledge_retrieval import get_knowledge_retrieval_tool
from ..converter import MarkdownConverter, ConversionRequest
from ..schema import AgentState
from ..logger import logger
from ..prompt import report_multi as prompts
from ..template import create_task_schedule, TaskType, TaskStatus


class ReportGeneratorAgent(BaseAgent):
    """
    Structured Report Generator Coordinator Agent
    
    This agent coordinates multiple SectionAgents to generate structured reports
    based on templates. It supports both parallel and sequential processing modes,
    automatic content quality assessment, and polishing capabilities.
    
    Attributes:
        template_path (Optional[Path]): Path to the report template file
        output_path (Optional[Path]): Path where generated reports will be saved
        knowledge_base_path (str): Path to the knowledge base for content retrieval
        parallel_sections (bool): Whether to process sections in parallel
        max_concurrent (int): Maximum number of concurrent section processing
        enable_polishing (bool): Whether to enable content polishing
        
    Example:
        >>> agent = ReportGeneratorAgent(
        ...     template_path="templates/report.md",
        ...     knowledge_base_path="data/documents",
        ...     output_path="output/report.md",
        ...     parallel_sections=True,
        ...     max_concurrent=3
        ... )
        >>> result = await agent.run_with_template("template.md", "output.md")
    """
    
    def __init__(self, 
                 template_path: Optional[str] = None,
                 knowledge_base_path: str = "workdir/documents",
                 output_path: Optional[str] = None,
                 parallel_sections: bool = False,
                 max_concurrent: int = 3,
                 enable_polishing: bool = True,
                 **kwargs):
        """
        Initialize the ReportGeneratorAgent.
        
        Args:
            template_path (Optional[str]): Path to the report template file
            knowledge_base_path (str): Path to the knowledge base directory
            output_path (Optional[str]): Path where the generated report will be saved
            parallel_sections (bool): Whether to process sections in parallel
            max_concurrent (int): Maximum number of concurrent section processing
            enable_polishing (bool): Whether to enable content polishing and quality checks
            **kwargs: Additional keyword arguments passed to the base class
        """
        super().__init__(**kwargs)
        
        self.template_path = Path(template_path) if template_path else None
        self.output_path = Path(output_path) if output_path else None
        self.knowledge_base_path = knowledge_base_path
        self.parallel_sections = parallel_sections
        self.max_concurrent = max_concurrent
        
        # Initialize tools
        self.knowledge_tool = get_knowledge_retrieval_tool(knowledge_base_path)
        self.converter = MarkdownConverter()
        
        # Report template and content
        self.template_structure = None
        self.report_content = {}
        
        # Task scheduling system
        self.task_schedule = None
        self.use_schedule_mode = False
        
        # Section agent management
        self.section_agents: List[SectionAgent | SectionAgentReAct] = []
        self.completed_sections = set()
        self.active_agents: Dict[int, SectionAgentReAct] = {}
        
        # Content quality control
        self.enable_polishing = enable_polishing
        self.quality_check_passed = False
        self.polishing_completed = False
        
        # Set agent default parameters
        if not self.name:
            self.name = "report_coordinator"
        if not self.description:
            self.description = "Intelligent agent that coordinates multiple SectionAgents to generate structured reports"
        
        # Set system prompt
        self.system_prompt = prompts.SYSTEM_PROMPT
    
    async def initialize_from_template(self, template_content: str):
        """
        Initialize the agent from template content.
        
        This method parses the template content (either Markdown or JSON format)
        and sets up the report structure and section agents accordingly.
        
        Args:
            template_content (str): The template content in Markdown or JSON format
            
        Raises:
            ValueError: If template conversion fails
            Exception: If template initialization fails
        """
        try:
            # If it's a Markdown template, convert to JSON structure
            if template_content.strip().startswith('#'):
                conversion_request = ConversionRequest(
                    source_format="markdown",
                    target_format="json",
                    content=template_content,
                    options=None
                )
                conversion_result = self.converter.convert(conversion_request)
                
                if conversion_result.success:
                    self.template_structure = conversion_result.result
                else:
                    raise ValueError(f"Template conversion failed: {conversion_result.error}")
            else:
                # Assume JSON format
                self.template_structure = json.loads(template_content)
            
            # Initialize report content structure
            self._initialize_report_structure()
            
            logger.info(f"Template initialization completed with {len(self.template_structure.get('content', []))} sections")
            
        except Exception as e:
            logger.error(f"Template initialization failed: {e}")
            raise
    
    async def initialize_from_schedule(self, template_path: str, confirm_execution: bool = True):
        """
        Initialize the agent using task schedule system.
        
        This method creates a task schedule from the template, displays the task queue
        for user confirmation, and sets up agents for each task.
        
        Args:
            template_path (str): Path to the template file
            confirm_execution (bool): Whether to ask for user confirmation before execution
            
        Raises:
            ValueError: If task schedule creation fails
            Exception: If schedule initialization fails
        """
        try:
            logger.info(f"Initializing task schedule from template: {template_path}")
            
            # Create task schedule
            self.task_schedule = create_task_schedule(
                template_path, 
                max_concurrent=self.max_concurrent
            )
            self.use_schedule_mode = True
            
            # Display task queue information
            self._display_task_queue()
            
            # Ask for user confirmation if needed
            if confirm_execution:
                confirmed = self._ask_user_confirmation()
                if not confirmed:
                    logger.info("User cancelled task execution")
                    return False
            
            # Initialize report structure from schedule
            self._initialize_from_task_schedule()
            
            logger.info(f"Task schedule initialization completed with {len(self.task_schedule.tasks)} tasks")
            return True
            
        except Exception as e:
            logger.error(f"Task schedule initialization failed: {e}")
            raise
    
    def _initialize_report_structure(self):
        """
        Initialize report content structure and section agents.
        
        This method creates the basic report structure and instantiates
        the appropriate section agents for each heading in the template.
        Only creates agents for top-level headings (not contained within other headings).
        """
        self.report_content = {
            "title": self.template_structure.get("title", "Report"),
            "metadata": self.template_structure.get("metadata", {}),
            "content": []
        }
        
        # Create section structure and corresponding agents for each template part
        report_context = {
            "title": self.report_content["title"],
            "metadata": self.report_content["metadata"]
        }
        
        # Extract all headings recursively from the hierarchical structure
        all_headings = []
        self._extract_all_headings(self.template_structure.get("content", []), all_headings)
        
        # Find leaf headings (those without child headings)
        leaf_headings = self._find_leaf_headings(all_headings)
        
        # Create sections and agents only for leaf headings
        for heading_info in leaf_headings:
            element = heading_info["element"]
            section = {
                "type": "heading",
                "level": element.get("level", 1),
                "content": element.get("content", ""),
                "id": len(self.report_content["content"]),
                "completed": False,
                "generated_content": ""
            }
            self.report_content["content"].append(section)
            
            # Create dedicated SectionAgent for each top-level section
            section_title = section.get("content", f"Section {section['id']}")
            
            # Extract children_content from template element for output format
            output_format = None
            element_attributes = element.get("attributes", {})
            if element_attributes and "children_content" in element_attributes:
                output_format = element_attributes["children_content"]
            
            if settings.section_agent_react:
                section_agent = SectionAgentReAct(
                    # name=f"section_agent_{section['id']}",
                    section_info=section,
                    report_context=report_context,
                    knowledge_base_path=self.knowledge_base_path,
                    output_format=output_format,
                    llm=self.llm
                )
            else:
                section_agent = SectionAgent(
                    name=f"section_agent_{section['id']}",
                    section_info=section,
                    report_context=report_context,
                    knowledge_base_path=self.knowledge_base_path,
                )
            self.section_agents.append(section_agent)
    
    def _find_leaf_headings(self, all_headings: List[Dict]) -> List[Dict]:
        """
        Find leaf headings that have no child headings under them.
        
        A heading is considered a leaf if it has no children_content with headings.
        
        Args:
            all_headings: List of all heading elements with their metadata
            
        Returns:
            List of leaf headings that should have agents created
        """
        if not all_headings:
            return []
        
        leaf_headings = []
        
        for heading in all_headings:
            element = heading["element"]
            element_attributes = element.get("attributes", {})
            
            # Check if this heading has child headings
            has_child_headings = False
            if element_attributes and "children_content" in element_attributes:
                children_content = element_attributes["children_content"]
                if children_content:
                    # Check if any child element is a heading
                    for child in children_content:
                        if child.get("type") == "heading":
                            has_child_headings = True
                            break
            
            # If no child headings, this is a leaf
            if not has_child_headings:
                leaf_headings.append(heading)
        
        logger.info(f"Found {len(leaf_headings)} leaf headings")
        for heading in leaf_headings:
            logger.info(f"  - {heading['content']} (level {heading['level']})")
        
        return leaf_headings
    
    def _extract_all_headings(self, content_list: List[Dict], all_headings: List[Dict], parent_index: int = 0) -> None:
        """
        Recursively extract all headings from the hierarchical structure.
        
        Args:
            content_list: List of content elements to process
            all_headings: List to collect all found headings
            parent_index: Index offset for tracking position
        """
        for i, element in enumerate(content_list):
            if element.get("type") == "heading":
                # Get level from attributes or direct level field
                level = element.get("level", 1)
                if "attributes" in element and element["attributes"]:
                    level = element["attributes"].get("level", level)
                
                heading_info = {
                    "index": parent_index + i,
                    "element": element,
                    "level": level,
                    "content": element.get("content", "")
                }
                all_headings.append(heading_info)
                
                # Recursively extract headings from children_content
                element_attributes = element.get("attributes", {})
                if element_attributes and "children_content" in element_attributes:
                    children_content = element_attributes["children_content"]
                    if children_content:
                        self._extract_all_headings(children_content, all_headings, parent_index + i + 1)
    
    def _display_task_queue(self):
        """Display task queue information to console."""
        if not self.task_schedule:
            return
        
        print("\n" + "="*80)
        print(f"📋 Task Queue - {self.task_schedule.document.title}")
        print("="*80)
        
        # Get task graph info
        graph_info = self.task_schedule.get_task_graph_info()
        
        print(f"\n📊 Task Statistics:")
        print(f"   Total tasks: {graph_info['total_tasks']}")
        print(f"   Generation tasks: {graph_info['generation_tasks']}")
        print(f"   Merge tasks: {graph_info['merge_tasks']}")
        print(f"   Max concurrent: {self.max_concurrent}")
        
        print(f"\n📈 Task Distribution by Level:")
        for level, counts in graph_info['tasks_by_level'].items():
            total = counts['generation'] + counts['merge']
            print(f"   Level {level}: Generation({counts['generation']}) + Merge({counts['merge']}) = {total}")
        
        print("\n" + "-"*80)
        print("🔍 Task Details and Agents:")
        print("-"*80)
        
        # Group tasks by level
        tasks_by_level = {}
        for task in self.task_schedule.tasks.values():
            level = task.level
            if level not in tasks_by_level:
                tasks_by_level[level] = []
            tasks_by_level[level].append(task)
        
        # Display tasks by level
        for level in sorted(tasks_by_level.keys()):
            tasks = tasks_by_level[level]
            print(f"\n🏷️  Level {level} ({len(tasks)} tasks)")
            print("-" * 60)
            
            for i, task in enumerate(tasks, 1):
                task_type_icon = "🔧" if task.task_type == TaskType.GENERATION else "🔀"
                agent_type = "SectionAgentReAct (Generation)" if task.task_type == TaskType.GENERATION else "SectionAgentReAct (Merge)"
                
                print(f"   {i:2d}. {task_type_icon} {task.title}")
                print(f"       Agent: {agent_type}")
                print(f"       Task Type: {task.task_type.value}")
                print(f"       Dependencies: {len(task.dependencies)}")
                print(f"       Estimated Duration: {task.estimated_duration}s")
                
                if task.dependencies:
                    print(f"       Depends on:")
                    for dep_id in task.dependencies:
                        dep_task = self.task_schedule.tasks.get(dep_id)
                        if dep_task:
                            print(f"         - {dep_task.title} ({dep_task.task_type.value})")
                print()
        
        # Show execution sequence preview
        self._display_execution_preview()
    
    def _display_execution_preview(self):
        """Display execution sequence preview."""
        if not self.task_schedule:
            return
        
        print("\n" + "-"*80)
        print("⚡ Execution Sequence Preview")
        print("-"*80)
        
        # Simulate task execution order (simplified version)
        completed_tasks = set()
        execution_rounds = []
        max_rounds = 20
        
        for round_num in range(max_rounds):
            ready_tasks = []
            for task in self.task_schedule.tasks.values():
                if (task.id not in completed_tasks and 
                    set(task.dependencies).issubset(completed_tasks)):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                break
            
            # Sort by priority and limit to max concurrent
            ready_tasks.sort(key=lambda t: (-t.priority, t.level if t.task_type == TaskType.GENERATION else -t.level))
            ready_tasks = ready_tasks[:self.max_concurrent]
            
            execution_rounds.append(ready_tasks)
            
            for task in ready_tasks:
                completed_tasks.add(task.id)
        
        print(f"\n🚀 Estimated execution rounds: {len(execution_rounds)}")
        print(f"📊 Total tasks: {len(self.task_schedule.tasks)}")
        
        for round_num, tasks in enumerate(execution_rounds[:5], 1):  # Show first 5 rounds
            print(f"\nRound {round_num} - Parallel execution ({len(tasks)} tasks):")
            for i, task in enumerate(tasks, 1):
                task_type = "Generation" if task.task_type == TaskType.GENERATION else "Merge"
                print(f"   {i:2d}. [{task_type}] {task.title} (Level {task.level})")
        
        if len(execution_rounds) > 5:
            print(f"\n   ... and {len(execution_rounds) - 5} more rounds")
        
        print(f"\n✅ All tasks will complete in {len(execution_rounds)} rounds")
    
    def _ask_user_confirmation(self) -> bool:
        """Ask user for confirmation to proceed with task execution."""
        print("\n" + "="*80)
        print("❓ User Confirmation Required")
        print("="*80)
        print("\nThe task queue is ready for execution.")
        print("This will create and run multiple agents to generate report content.")
        print(f"Estimated total duration: {sum(t.estimated_duration for t in self.task_schedule.tasks.values()):.1f}s")
        
        while True:
            try:
                response = input("\nDo you want to proceed with execution? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    print("✅ User confirmed. Starting task execution...")
                    return True
                elif response in ['n', 'no']:
                    print("❌ User cancelled task execution.")
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                print("\n❌ User cancelled with Ctrl+C.")
                return False
            except Exception:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
    
    def _initialize_from_task_schedule(self):
        """Initialize report structure and agents from task schedule."""
        if not self.task_schedule:
            return
        
        self.report_content = {
            "title": self.task_schedule.document.title,
            "metadata": {},
            "content": []
        }
        
        report_context = {
            "title": self.report_content["title"],
            "metadata": self.report_content["metadata"]
        }
        
        # Create agents for each task
        for task_id, task in self.task_schedule.tasks.items():
            # Create section info from task
            section = {
                "type": "heading",
                "level": task.level,
                "content": task.title,
                "id": len(self.report_content["content"]),
                "completed": False,
                "generated_content": "",
                "task_id": task_id  # Link to schedule task
            }
            self.report_content["content"].append(section)
            
            # Determine task type for agent
            agent_task_type = task.task_type

            # Create section agent with appropriate task type
            if settings.section_agent_react:
                section_agent = SectionAgentReAct(
                    section_info=section,
                    report_context=report_context,
                    knowledge_base_path=self.knowledge_base_path,
                    output_format=task.section_content,  # Use task content as output format
                    task_type=agent_task_type,
                    llm=self.llm
                )
            else:
                section_agent = SectionAgent(
                    name=f"section_agent_{section['id']}",
                    section_info=section,
                    report_context=report_context,
                    knowledge_base_path=self.knowledge_base_path,
                    task_type=agent_task_type,
                    output_format=task.section_content  # Use task content as output format
                )
            
            self.section_agents.append(section_agent)
            logger.info(f"Created {agent_task_type.value} agent for task: {task.title}")
        
        logger.info(f"Initialized {len(self.section_agents)} agents from task schedule")
    
    async def step(self) -> str:
        """
        Execute a single step operation - coordinate section generation.
        
        This method manages the overall report generation process,
        including section generation, content polishing, and quality checks.
        
        Returns:
            str: Status message describing the current step result
            
        Raises:
            Exception: If coordination step execution fails
        """
        try:
            # Check if there are pending sections
            pending_sections = [
                section for section in self.report_content["content"]
                if not section.get("completed", False)
            ]
            
            if not pending_sections:
                # All sections completed, proceed with content polishing and quality check
                if self.enable_polishing and not self.polishing_completed:
                    return await self._polish_and_check_content()
                else:
                    # Polishing completed, save report and terminate
                    await self._save_report()
                    logger.info("All sections completed, saving report")
                    self.state = AgentState.FINISHED
                    return "Report generation completed"
            
            if self.parallel_sections:
                return await self._handle_parallel_generation()
            else:
                return await self._handle_sequential_generation()
                
        except Exception as e:
            logger.error(f"Coordination step execution failed: {e}")
            return f"Execution failed: {str(e)}"
    
    async def _handle_sequential_generation(self) -> str:
        """
        Handle sequential section generation.
        
        Process sections one by one in order, ensuring each section
        is completed before moving to the next.
        
        Returns:
            str: Status message describing the current section processing
        """
        # Find the next pending section
        for i, section in enumerate(self.report_content["content"]):
            if not section.get("completed", False):
                section_agent = self.section_agents[i]
                section_title = section.get("content", f"Section {i+1}")
                
                logger.info(f"Starting section generation: {section_title}")
                
                try:
                    # Run section agent
                    await section_agent.run()
                    generated_content = section_agent.get_content()
                    
                    if generated_content:
                        section["generated_content"] = generated_content
                        section["completed"] = True
                        self.completed_sections.add(section["id"])
                        
                        progress = f"{len(self.completed_sections)}/{len(self.report_content['content'])}"
                        logger.info(f"Section '{section_title}' generation completed, progress: {progress}")
                        
                        return f"Completed section '{section_title}' ({progress})"
                    else:
                        return f"Generating section '{section_title}'"
                        
                except Exception as e:
                    logger.error(f"Section '{section_title}' generation failed: {e}")
                    section["generated_content"] = f"[Generation failed] {str(e)}"
                    section["completed"] = True
                    return f"Section '{section_title}' generation failed: {str(e)}"
        
        return "All sections processed"
    
    async def _handle_parallel_generation(self) -> str:
        """
        Handle parallel section generation.
        
        Process multiple sections concurrently with a limit on
        the maximum number of concurrent operations.
        
        Returns:
            str: Status message describing the parallel processing result
        """
        # Get all incomplete sections
        pending_sections = [
            (i, section) for i, section in enumerate(self.report_content["content"])
            if not section.get("completed", False)
        ]
        
        if not pending_sections:
            return "All sections completed"
        
        # Limit concurrency
        batch_size = min(len(pending_sections), self.max_concurrent)
        current_batch = pending_sections[:batch_size]
        
        logger.info(f"Starting parallel generation of {len(current_batch)} sections")
        
        # Create concurrent tasks
        tasks = []
        for section_idx, section in current_batch:
            section_agent = self.section_agents[section_idx]
            task = self._generate_section_with_agent(section_idx, section, section_agent)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        completed_count = 0
        for i, result in enumerate(results):
            section_idx, section = current_batch[i]
            if isinstance(result, Exception):
                logger.error(f"Parallel section generation failed: {result}")
                section["generated_content"] = f"[Generation failed] {str(result)}"
            else:
                section["generated_content"] = result
            
            section["completed"] = True
            self.completed_sections.add(section["id"])
            completed_count += 1
        
        progress = f"{len(self.completed_sections)}/{len(self.report_content['content'])}"
        logger.info(f"Parallel generation completed {completed_count} sections, total progress: {progress}")
        
        return f"Parallel completed {completed_count} sections ({progress})"
    
    async def _generate_section_with_agent(self, section_idx: int, section: Dict[str, Any], agent: SectionAgentReAct) -> str:
        """
        Generate section content using the specified agent.
        
        Args:
            section_idx (int): Index of the section being generated
            section (Dict[str, Any]): Section information dictionary
            agent (SectionAgentReAct): The agent responsible for generating this section
            
        Returns:
            str: Generated content for the section
            
        Raises:
            Exception: If section generation fails
        """
        try:
            section_title = section.get("content", f"Section {section_idx + 1}")
            logger.info(f"Starting agent-based section generation: {section_title}")
            
            await agent.run()
            content = agent.get_content()
            return content if content else f"[Empty content] {section_title}"
        except Exception as e:
            logger.error(f"Agent generation for section {section_idx + 1} failed: {e}")
            raise
    
    async def _polish_and_check_content(self) -> str:
        """
        Polish and check report content quality.
        
        This method performs comprehensive quality assessment and improvement
        of the generated report content, including:
        - Content quality evaluation
        - Content polishing based on identified issues
        - Consistency checking across sections
        - Final quality validation
        
        Returns:
            str: Status message describing the polishing result
        """
        try:
            logger.info("Starting overall report content polishing and quality check")
            
            # 1. Content quality assessment
            quality_issues = await self._assess_content_quality()
            
            # 2. Content polishing
            if quality_issues:
                logger.info(f"Found {len(quality_issues)} quality issues, starting polishing")
                await self._polish_content(quality_issues)
            
            # 3. Consistency check
            consistency_issues = await self._check_content_consistency()
            
            # 4. Fix consistency issues
            if consistency_issues:
                logger.info(f"Found {len(consistency_issues)} consistency issues, starting fixes")
                await self._fix_consistency_issues(consistency_issues)
            
            # 5. Final quality validation
            final_quality = await self._final_quality_check()
            
            self.polishing_completed = True
            self.quality_check_passed = final_quality
            
            if final_quality:
                logger.info("✅ Content polishing and quality check completed, report quality is good")
                return "Content polishing and quality check completed"
            else:
                logger.warning("⚠️ Content polishing completed, but some quality issues remain")
                return "Content polishing completed (minor quality issues remain)"
                
        except Exception as e:
            logger.error(f"Content polishing and checking failed: {e}")
            self.polishing_completed = True
            return f"Content polishing failed: {str(e)}"
    
    async def _assess_content_quality(self) -> List[Dict[str, Any]]:
        """
        Assess content quality for each section.
        
        Evaluates the quality of generated content using LLM-based assessment
        and identifies sections that need improvement.
        
        Returns:
            List[Dict[str, Any]]: List of quality issues found in sections
        """
        quality_issues = []
        
        # Check content quality for each section
        for i, section in enumerate(self.report_content["content"]):
            if not section.get("generated_content"):
                continue
                
            section_title = section.get("content", f"Section {i+1}")
            content = section["generated_content"]
            
            # Build quality assessment prompt
            quality_prompt = prompts.QUALITY_ASSESSMENT_PROMPT.format(
                section_title=section_title,
                content=content
            )
            
            self.update_memory("user", quality_prompt)
            
            try:
                response = await self.llm.ask(self.memory.messages)
                if response:
                    # Parse assessment results
                    import re
                    json_match = re.search(r'\{[^{}]*}', response, re.DOTALL)
                    if json_match:
                        evaluation = json.loads(json_match.group())
                        
                        # If quality score is below threshold, record as issue
                        if evaluation.get("overall_score", 5) < 4:
                            quality_issues.append({
                                "section_idx": i,
                                "section_title": section_title,
                                "evaluation": evaluation,
                                "content": content
                            })
                            
                self.update_memory("assistant", response)
                
            except Exception as e:
                logger.error(f"Quality assessment for section '{section_title}' failed: {e}")
        
        return quality_issues
    
    async def _polish_content(self, quality_issues: List[Dict[str, Any]]) -> None:
        """
        Polish content based on identified quality issues.
        
        Args:
            quality_issues (List[Dict[str, Any]]): List of quality issues to address
        """
        for issue_info in quality_issues:
            section_idx = issue_info["section_idx"]
            section_title = issue_info["section_title"]
            evaluation = issue_info["evaluation"]
            original_content = issue_info["content"]
            
            logger.info(f"Polishing section: {section_title}")
            
            # Build polishing prompt
            polish_prompt = prompts.POLISH_CONTENT_PROMPT.format(
                section_title=section_title,
                original_content=original_content,
                issues=', '.join(evaluation.get('issues', [])),
                suggestions=', '.join(evaluation.get('suggestions', []))
            )
            
            self.update_memory("user", polish_prompt)
            
            try:
                response = await self.llm.ask(self.memory.messages)
                if response and response.strip():
                    # Update section content
                    polished_content = response.strip()
                    self.report_content["content"][section_idx]["generated_content"] = polished_content
                    
                    logger.info(f"Section '{section_title}' polishing completed")
                    self.update_memory("assistant", polished_content)
                else:
                    logger.warning(f"Section '{section_title}' polishing received no valid response")
                    
            except Exception as e:
                logger.error(f"Section '{section_title}' polishing failed: {e}")
    
    async def _check_content_consistency(self) -> List[Dict[str, str]]:
        """
        Check content consistency across sections.
        
        Analyzes the overall report for consistency issues such as
        contradictory information, inconsistent terminology, or
        misaligned content flow.
        
        Returns:
            List[Dict[str, str]]: List of consistency issues found
        """
        consistency_issues = []
        
        # Collect all section content
        all_content = ""
        section_summaries = []
        
        for section in self.report_content["content"]:
            if section.get("generated_content"):
                section_title = section.get("content", "")
                content = section["generated_content"]
                all_content += f"\n\n## {section_title}\n{content}"
                section_summaries.append(f"- {section_title}: {content[:100]}...")
        
        # Build consistency check prompt
        consistency_prompt = prompts.CONSISTENCY_CHECK_PROMPT.format(
            report_title=self.report_content.get('title', ''),
            section_summaries=chr(10).join(section_summaries),
            full_content=all_content
        )
        
        self.update_memory("user", consistency_prompt)
        
        try:
            response = await self.llm.ask(self.memory.messages)
            if response:
                # Parse consistency check results
                import re
                json_match = re.search(r'\{.*}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    consistency_issues = result.get("issues", [])
                    
                self.update_memory("assistant", response)
                
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
        
        return consistency_issues
    
    async def _fix_consistency_issues(self, consistency_issues: List[Dict[str, str]]) -> None:
        """
        Fix consistency issues identified in the report.
        
        Args:
            consistency_issues (List[Dict[str, str]]): List of consistency issues to fix
        """
        for issue in consistency_issues:
            issue_type = issue.get("type", "Unknown issue")
            description = issue.get("description", "")
            affected_sections = issue.get("affected_sections", [])
            suggestion = issue.get("suggestion", "")
            
            logger.info(f"Fixing consistency issue: {issue_type}")
            
            # Find affected sections
            for section in self.report_content["content"]:
                section_title = section.get("content", "")
                if any(affected_section in section_title for affected_section in affected_sections):
                    
                    original_content = section.get("generated_content", "")
                    if not original_content:
                        continue
                    
                    # Build fix prompt
                    fix_prompt = prompts.FIX_CONSISTENCY_PROMPT.format(
                        section_title=section_title,
                        original_content=original_content,
                        issue_type=issue_type,
                        description=description,
                        suggestion=suggestion
                    )
                    
                    self.update_memory("user", fix_prompt)
                    
                    try:
                        response = await self.llm.ask(self.memory.messages)
                        if response and response.strip():
                            fixed_content = response.strip()
                            section["generated_content"] = fixed_content
                            logger.info(f"Section '{section_title}' consistency issue fixed")
                            self.update_memory("assistant", fixed_content)
                        
                    except Exception as e:
                        logger.error(f"Section '{section_title}' consistency fix failed: {e}")
    
    async def _final_quality_check(self) -> bool:
        """
        Perform final quality check on the complete report.
        
        Conducts a comprehensive final assessment of the entire report
        to determine if it meets quality standards.
        
        Returns:
            bool: True if quality check passes, False otherwise
        """
        try:
            # Build final report content
            full_report = f"# {self.report_content.get('title', '')}\\n"
            
            for section in self.report_content["content"]:
                section_title = section.get("content", "")
                content = section.get("generated_content", "")
                level_prefix = "#" * section.get("level", 1)
                full_report += f"\n{level_prefix} {section_title}\n\n{content}\n"
            
            # Final quality assessment prompt
            final_check_prompt = prompts.FINAL_QUALITY_CHECK_PROMPT.format(
                full_report=full_report
            )
            
            self.update_memory("user", final_check_prompt)
            response = await self.llm.ask(self.memory.messages)
            
            if response:
                self.update_memory("assistant", response)
                # Simple pass/fail determination
                return "PASS" in response.upper()
            else:
                return False
                
        except Exception as e:
            logger.error(f"Final quality check failed: {e}")
            return False
    
    
    async def _save_report(self):
        """
        Save the generated report to files.
        
        Saves the complete report in both JSON and Markdown formats
        to the specified output path.
        
        Raises:
            Exception: If report saving fails
        """
        try:
            # Build complete report structure
            report_document = {
                "title": self.report_content["title"],
                "metadata": self.report_content["metadata"],
                "content": []
            }
            
            # Add generated content
            for section in self.report_content["content"]:
                # Add heading
                report_document["content"].append({
                    "type": "heading",
                    "level": section["level"],
                    "content": section["content"],
                    "attributes": {"level": section["level"]}
                })
                
                # Add generated content as paragraph
                if section.get("generated_content"):
                    report_document["content"].append({
                        "type": "paragraph",
                        "content": section["generated_content"],
                        "attributes": {}
                    })
            
            # Save JSON format
            if self.output_path:
                json_path = self.output_path.with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(report_document, f, ensure_ascii=False, indent=2)
                
                # Convert and save Markdown format
                conversion_request = ConversionRequest(
                    source_format="json",
                    target_format="markdown",
                    content=report_document,
                    options=None
                )
                conversion_result = self.converter.convert(conversion_request)
                
                if conversion_result.success:
                    md_path = self.output_path.with_suffix('.md')
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(conversion_result.result)
                    
                    logger.info(f"Report saved: {json_path}, {md_path}")
                else:
                    logger.warning(f"Markdown conversion failed: {conversion_result.error}")
            else:
                logger.info("No output path specified, report not saved to file")
            
        except Exception as e:
            logger.error(f"Report saving failed: {e}")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get report generation progress information.
        
        Returns comprehensive progress information including section completion,
        overall progress percentage, active sections, and current processing phase.
        
        Returns:
            Dict[str, Any]: Progress information dictionary containing:
                - total_sections: Total number of sections in the report
                - completed_sections: Number of completed sections
                - progress_percentage: Overall progress as a percentage
                - active_sections: List of currently active section titles
                - remaining_sections: Number of remaining sections
                - mode: Processing mode ("parallel" or "sequential")
                - polishing_enabled: Whether polishing is enabled
                - polishing_completed: Whether polishing phase is completed
                - quality_check_passed: Whether quality check passed
                - current_phase: Current processing phase ("generation" or "polishing")
        """
        total_sections = len(self.report_content.get("content", []))
        completed_sections = len(self.completed_sections)
        
        # Get currently processing sections
        active_sections = []
        for i, agent in enumerate(self.section_agents):
            if agent.state == AgentState.RUNNING and not self.report_content["content"][i].get("completed", False):
                active_sections.append(self.report_content["content"][i].get("content", f"Section {i+1}"))
        
        # Calculate overall progress (including polishing phase)
        base_progress = (completed_sections / total_sections * 80) if total_sections > 0 else 0
        if self.polishing_completed:
            base_progress += 20  # Additional 20% for completed polishing
        elif completed_sections == total_sections and self.enable_polishing:
            base_progress += 10  # Additional 10% for polishing in progress
        
        return {
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "progress_percentage": min(base_progress, 100),
            "active_sections": active_sections,
            "remaining_sections": total_sections - completed_sections,
            "mode": "parallel" if self.parallel_sections else "sequential",
            "polishing_enabled": self.enable_polishing,
            "polishing_completed": self.polishing_completed,
            "quality_check_passed": self.quality_check_passed,
            "current_phase": "polishing" if (completed_sections == total_sections and not self.polishing_completed and self.enable_polishing) else "generation"
        }
    
    async def run_with_template(self, template_path: str, output_path: Optional[str] = None) -> str:
        """
        Run the agent with a specified template.
        
        This is a convenience method that loads a template file,
        initializes the agent, and runs the report generation process.
        
        Args:
            template_path (str): Path to the template file to use
            output_path (Optional[str]): Path where the generated report will be saved
            
        Returns:
            str: Result message from the agent execution
            
        Raises:
            Exception: If template running fails
        """
        try:
            # Load template
            template_content = Path(template_path).read_text(encoding='utf-8')
            await self.initialize_from_template(template_content)
            
            # Set output path
            if output_path:
                self.output_path = Path(output_path)
            
            # Run agent
            result = await self.run()
            
            return result
            
        except Exception as e:
            logger.error(f"Template running failed: {e}")
            raise
    
    async def run_with_schedule(self, template_path: str, output_path: Optional[str] = None,
                               confirm_execution: bool = True) -> str:
        """
        Run the agent with schedule-based task management.
        
        This method uses the task scheduling system to coordinate report generation
        with proper dependency management and task type specialization.
        
        Args:
            template_path (str): Path to the template file to use
            output_path (Optional[str]): Path where the generated report will be saved
            confirm_execution (bool): Whether to ask for user confirmation before execution
            
        Returns:
            str: Result message from the agent execution
            
        Raises:
            Exception: If schedule-based running fails
        """
        try:
            # Initialize from schedule
            confirmed = await self.initialize_from_schedule(template_path, confirm_execution)
            if not confirmed:
                return "Task execution cancelled by user"
            
            # Set output path
            if output_path:
                self.output_path = Path(output_path)
            
            # Run agent with schedule coordination
            result = await self.run()
            
            return result
            
        except Exception as e:
            logger.error(f"Schedule-based running failed: {e}")
            raise