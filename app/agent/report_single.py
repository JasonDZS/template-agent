#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured Report Generator Agent

This module contains the single-agent implementation for generating structured reports
based on templates. It processes sections sequentially and generates comprehensive
reports with knowledge retrieval capabilities.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from app.agent.base import BaseAgent
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.converter import MarkdownConverter, ConversionRequest
from app.schema import AgentState
from app.logger import logger
from app.prompt import report_single as prompts


class ReportGeneratorAgentSingle(BaseAgent):
    """
    Structured Report Generator Agent - Single Agent Implementation
    
    This agent generates structured reports based on templates using a single-agent
    approach. It processes sections sequentially, retrieves relevant knowledge,
    and produces comprehensive reports in both JSON and Markdown formats.
    
    Attributes:
        template_path (Optional[Path]): Path to the report template file
        output_path (Optional[Path]): Path where generated reports will be saved
        knowledge_base_path (str): Path to the knowledge base for content retrieval
        knowledge_tool (KnowledgeRetrievalTool): Tool for retrieving relevant information
        converter (MarkdownConverter): Tool for format conversion between JSON and Markdown
        template_structure (Optional[Dict]): Parsed template structure
        report_content (Dict[str, Any]): Current report content being generated
        current_section (Optional[Dict]): Currently processing section
        completed_sections (Set[int]): Set of completed section IDs
        
    Example:
        >>> agent = ReportGeneratorAgentSingle(
        ...     template_path="templates/report.md",
        ...     knowledge_base_path="data/documents",
        ...     output_path="output/report.md"
        ... )
        >>> result = await agent.run_with_template("template.md", "output.md")
    """

    def __init__(self,
                 template_path: Optional[str] = None,
                 knowledge_base_path: str = "workdir/documents",
                 output_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Report Generator Agent.
        
        Args:
            template_path (Optional[str]): Path to the report template file
            knowledge_base_path (str): Path to the knowledge base directory
            output_path (Optional[str]): Path where the generated report will be saved
            **kwargs: Additional arguments passed to the base agent
        """
        super().__init__(**kwargs)

        self.template_path = Path(template_path) if template_path else None
        self.output_path = Path(output_path) if output_path else None
        self.knowledge_base_path = knowledge_base_path

        # Initialize tools
        self.knowledge_tool = KnowledgeRetrievalTool(knowledge_base_path)
        self.converter = MarkdownConverter()

        # Report template and content structures
        self.template_structure = None
        self.report_content = {}
        self.current_section = None
        self.completed_sections = set()

        # Set agent default parameters
        if not self.name:
            self.name = "report_generator"
        if not self.description:
            self.description = "Intelligent agent for generating structured reports"

        # Set system prompts
        self.system_prompt = prompts.SYSTEM_PROMPT

        self.next_step_prompt = prompts.NEXT_STEP_PROMPT

    async def initialize_from_template(self, template_content: str):
        """
        Initialize the agent from template content.
        
        This method parses the template content (either Markdown or JSON format)
        and sets up the report structure accordingly.
        
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
                    source_format = "markdown",
                    target_format = "json",
                    content = template_content
                )
                conversion_result = self.converter.convert(conversion_request)

                if conversion_result.success:
                    self.template_structure = conversion_result.result
                else:
                    raise ValueError(f"Template conversion failed: {conversion_result.error}")
            else:
                # Assume it's JSON format
                self.template_structure = json.loads(template_content)

            # Initialize report content structure
            self._initialize_report_structure()

            logger.info(f"Template initialization completed, {len(self.template_structure.get('content', []))} sections found")

        except Exception as e:
            logger.error(f"Template initialization failed: {e}")
            raise

    def _initialize_report_structure(self):
        """
        Initialize the report content structure.
        
        This method creates the basic report structure and prepares
        empty content sections for each heading in the template.
        """
        self.report_content = {
            "title": self.template_structure.get("title", "Report"),
            "metadata": self.template_structure.get("metadata", {}),
            "content": []
        }

        # Create empty content structure for each template section
        for element in self.template_structure.get("content", []):
            if element.get("type") == "heading":
                section = {
                    "type": "heading",
                    "level": element.get("level", 1),
                    "content": element.get("content", ""),
                    "id": len(self.report_content["content"]),
                    "completed": False,
                    "generated_content": ""
                }
                self.report_content["content"].append(section)

    async def step(self) -> str:
        """
        Execute a single step operation.
        
        This method processes one section at a time, generating content
        and managing the overall report generation workflow.
        
        Returns:
            str: Status message describing the current step result
            
        Raises:
            Exception: If step execution fails
        """
        try:
            # Check if there are pending sections
            pending_sections = [
                section for section in self.report_content["content"]
                if not section.get("completed", False)
            ]

            if not pending_sections:
                # All sections completed, save report and terminate
                await self._save_report()
                logger.info("All sections completed, saving report")
                self.state = AgentState.FINISHED
                return "Report generation completed"

            # Select the next section to process
            current_section = pending_sections[0]
            self.current_section = current_section

            section_title = current_section.get("content", f"Section {current_section['id'] + 1}")
            logger.info(f"Starting section processing: {section_title}")

            # Build current step prompt
            progress_info = f"{len(self.completed_sections)}/{len(self.report_content['content'])}"
            pending_info = [s.get("content", f"Section {s['id'] + 1}") for s in pending_sections[:3]]

            step_prompt = self.next_step_prompt.format(
                progress = progress_info,
                pending_sections = ", ".join(pending_info)
            )

            # Update system message
            self.update_memory("system", f"{self.system_prompt}\n\n{step_prompt}")

            # Generate content for current section
            content_result = await self._generate_section_content(current_section)

            if content_result:
                current_section["generated_content"] = content_result
                current_section["completed"] = True
                self.completed_sections.add(current_section["id"])

                return f"Completed section '{section_title}': {content_result[:100]}..."
            else:
                return f"Processing section '{section_title}'"

        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return f"Execution failed: {str(e)}"

    async def _generate_section_content(self, section: Dict[str, Any]) -> str:
        """
        Generate content for the specified section.
        
        This method retrieves relevant knowledge from the knowledge base
        and uses LLM to generate appropriate content for the section.
        
        Args:
            section (Dict[str, Any]): Section information dictionary containing
                title, level, and other metadata
                
        Returns:
            str: Generated content for the section
            
        Raises:
            Exception: If content generation fails
        """
        section_title = section.get("content", "")
        section_level = section.get("level", 1)

        # Build retrieval query
        query = f"{section_title} {self.report_content.get('title', '')}"

        try:
            # Retrieve relevant information from knowledge base
            retrieval_result = await self.knowledge_tool.execute(
                query = query,
                top_k = 3,
                threshold = 0.01
            )

            if retrieval_result.error:
                logger.warning(f"Knowledge base retrieval failed: {retrieval_result.error}")
                knowledge_context = "No relevant knowledge base information retrieved"
            else:
                knowledge_context = retrieval_result.output

            # Build content generation prompt
            content_prompt = prompts.CONTENT_GENERATION_PROMPT.format(
                report_title=self.report_content.get('title', ''),
                section_title=section_title,
                section_level=section_level,
                knowledge_context=knowledge_context
            )

            self.update_memory("user", content_prompt)

            # Call LLM to generate content
            response = await self.llm.ask(self.memory.messages)

            if response:
                generated_content = response.strip()
                self.update_memory("assistant", generated_content)
                return generated_content
            else:
                logger.warning(f"LLM returned no valid content")
                return f"[To be completed] Content related to {section_title}"

        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return f"[Generation failed] {section_title}: {str(e)}"

    async def _save_report(self):
        """
        Save the generated report to files.
        
        This method builds the complete report structure and saves it
        in both JSON and Markdown formats to the specified output path.
        
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
                with open(json_path, 'w', encoding = 'utf-8') as f:
                    json.dump(report_document, f, ensure_ascii = False, indent = 2)

                # Convert and save Markdown format
                conversion_request = ConversionRequest(
                    source_format = "json",
                    target_format = "markdown",
                    content = report_document
                )
                conversion_result = self.converter.convert(conversion_request)

                if conversion_result.success:
                    md_path = self.output_path.with_suffix('.md')
                    with open(md_path, 'w', encoding = 'utf-8') as f:
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
        
        Returns comprehensive progress information including section completion
        status, progress percentage, and current processing section.
        
        Returns:
            Dict[str, Any]: Progress information dictionary containing:
                - total_sections: Total number of sections in the report
                - completed_sections: Number of completed sections
                - progress_percentage: Progress as a percentage (0-100)
                - current_section: Title of the currently processing section
                - remaining_sections: Number of remaining sections to process
        """
        total_sections = len(self.report_content.get("content", []))
        completed_sections = len(self.completed_sections)

        return {
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "progress_percentage": (completed_sections / total_sections * 100) if total_sections > 0 else 0,
            "current_section": self.current_section.get("content", "") if self.current_section else None,
            "remaining_sections": total_sections - completed_sections
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
            template_content = Path(template_path).read_text(encoding = 'utf-8')
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