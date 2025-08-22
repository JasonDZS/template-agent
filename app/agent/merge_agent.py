#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Agent based on BaseAgent.

This module provides a merge agent that merges content from child sections
using model-based intelligent merging. It's specialized for merging tasks only.
"""

from typing import Dict, Any, List, Optional
from app.agent.base import BaseAgent
from app.prompt.section_agent import NEXT_STEP_PROMPT, TaskPrompts
from app.schema import AgentState
from app.logger import logger


class MergeAgent(BaseAgent):
    """
    Merge agent specialized for merging content from child sections using model-based merging.
    """

    def __init__(self, 
                 section_info: Dict[str, Any],
                 report_context: Dict[str, Any],
                 child_contents: List[str],
                 output_format: Optional[str] = None,
                 **kwargs):
        """
        Initialize the MergeAgent.
        
        Args:
            section_info: Information about the section to merge
            report_context: Context about the overall report
            child_contents: List of content from child sections to merge
            output_format: Expected output format for the merged section
            **kwargs: Additional keyword arguments passed to the parent class
        """
        
        section_title = section_info.get("content", "Untitled Section")
        section_level = section_info.get("level", 1)
        section_id = section_info.get("id", 0)
        report_title = report_context.get("title", "")
        
        # Set basic information
        name = f"merge_{section_id}_{section_title[:10]}"
        description = f"Dedicated merge agent for section '{section_title}'"
        
        # Initialize parent class
        super().__init__(
            name=name,
            description=description,
            **kwargs
        )
        
        self.section_info = section_info
        self.report_context = report_context  
        self.child_contents = child_contents


        self.generated_content = ""
        self.is_completed = False

        # Set up merge prompt with child content for model-based merging
        self.system_prompt = TaskPrompts.MERGE_PROMPT.format(
            section_title=section_title,
            report_title=report_title,
            section_level=section_level,
            child_count=len(child_contents),
            output_format=output_format
        )

        logger.info(f"System prompt for merge section '{self.system_prompt}'")
        logger.info(f"Initialized MergeAgent: {section_title}")

    async def step(self) -> str:
        """Execute a single step in the merge workflow."""
        try:
            if self.is_completed:
                self.state = AgentState.FINISHED
                return f"Section '{self.section_info.get('content', '')}' generation completed"

            section_title = self.section_info.get('content', '')
            # Add system prompt if not already added
            if not any(msg.role == "system" for msg in self.memory.messages):
                self.update_memory("system", self.system_prompt)
                logger.info(f"Starting section merge: {section_title}")

            # Get LLM response for merging
            try:
                user_input = "\n".join(self.child_contents)

                if self.llm.model in ["Qwen/Qwen3-4B", "Qwen/Qwen3-32B"]:
                    logger.info(f"Using Qwen model, disabling think tag in prompt")
                    user_input = user_input + "\no_think<think></think>"

                self.update_memory("user", user_input)

                response = await self.llm.ask(messages=self.memory.messages)

                if response:
                    if "</think>" in response:
                        response = response.split("</think>")[1].strip()
                    self.generated_content = response.strip()
                    self.update_memory("assistant", self.generated_content)

                    self.is_completed = True
                    self.state = AgentState.FINISHED
                    logger.info(f"Section {section_title} merge completed successfully")
                    logger.info(f"=== Merged Content for Section '{section_title}' ===")
                    logger.info(self.generated_content)
                    logger.info("=================================================")
                    return  f"Section '{section_title}' merge completed"

                else:
                    logger.warning(f"LLM returned no valid content for section merge: {section_title}")
                    self.generated_content = f"[No content generated] LLM returned empty response for merge: {section_title}"
                    self.is_completed = True
                    self.state = AgentState.FINISHED
                    return f"Section '{section_title}' merge completed with no content from LLM"

            except Exception as e:
                logger.error(f"Section merge failed: {e}")
                self.state = AgentState.ERROR
                return f"Error during merging: {str(e)}"

        except Exception as e:
            logger.error(f"Merge step execution failed: {e}")
            self.state = AgentState.ERROR
            return f"Merge step failed: {str(e)}"

    def get_content(self) -> str:
        """Get the merged content."""
        return self.generated_content
    
    def is_finished(self) -> bool:
        """Check if the merge is finished."""
        return self.is_completed and self.state == AgentState.FINISHED