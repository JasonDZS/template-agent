#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Generation Agent.

This module provides a section generation agent specialized for content generation
tasks using knowledge retrieval and LLM generation.
"""

from typing import Dict, Any, Optional
from app.agent.base import BaseAgent
from app.config import settings
from app.tool.knowledge_retrieval import get_knowledge_retrieval_tool
from app.schema import AgentState
from app.logger import logger
from app.prompt.section_agent import TaskPrompts


class SectionAgent(BaseAgent):
    """
    Section generation agent specialized for content generation tasks.
    
    This agent is responsible for generating individual sections of a report.
    It retrieves relevant information from a knowledge base and uses an LLM
    to generate high-quality, structured section content.
    
    Attributes:
        section_info (Dict[str, Any]): Information about the section to generate.
        report_context (Dict[str, Any]): Context information about the overall report.
        knowledge_base_path (str): Path to the knowledge base for retrieval.
        knowledge_tool (KnowledgeRetrievalTool): Tool for retrieving knowledge.
        generated_content (str): The generated section content.
        is_completed (bool): Flag indicating if section generation is complete.
    """
    
    def __init__(self, 
                 section_info: Dict[str, Any],
                 report_context: Dict[str, Any],
                 knowledge_base_path: str = "workdir/documents",
                 output_format: Optional[str] = None,
                 **kwargs):
        """
        Initialize the SectionAgent.
        
        Args:
            section_info (Dict[str, Any]): Information about the section including
                title, level, id, etc.
            report_context (Dict[str, Any]): Context about the overall report
                including title and other metadata.
            knowledge_base_path (str, optional): Path to the knowledge base directory.
                Defaults to "workdir/documents".
            output_format (str, optional): Expected output format for the section.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        # Extract required fields from section_info
        section_title = section_info.get('content', 'Untitled Section')
        section_id = section_info.get('id', 0)
        
        # Set basic information for BaseAgent
        name = f'section_{section_id}_{section_title[:10]}'
        description = f'Dedicated agent for generating section: {section_title}'
        
        super().__init__(name=name, description=description, **kwargs)
        
        self.section_info = section_info
        self.report_context = report_context
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize tools
        self.knowledge_tool = get_knowledge_retrieval_tool(knowledge_base_path)
        
        # Generated content
        self.generated_content = ""
        self.is_completed = False
        
        # Set agent default parameters
        section_title = section_info.get("content", "Untitled Section")
        section_level = section_info.get("level", 1)
        report_title = report_context.get("title", "")
        
        if not self.description:
            self.description = f"Dedicated agent for generating section '{section_title}'"

        # Set up generation prompt
        self.system_prompt = TaskPrompts.GENERATION_PROMPT.format(
            section_title=section_title,
            report_title=report_title,
            section_level=section_level,
            output_format=output_format
        )
        self.update_memory("system", self.system_prompt)
    
    async def step(self) -> str:
        """
        Execute a single step operation.
        
        This method performs one iteration of the section generation process,
        including knowledge retrieval and content generation using the LLM.
        
        Returns:
            str: Status message indicating the current step result.
            
        Raises:
            Exception: If the step execution fails.
        """
        try:
            if self.is_completed:
                self.state = AgentState.FINISHED
                return f"Section '{self.section_info.get('content', '')}' generation completed"
            
            # Build retrieval query
            section_title = self.section_info.get("content", "")
            report_title = self.report_context.get("title", "")
            query = f"{section_title} {report_title}"
            
            logger.info(f"Starting section generation: {section_title}, Model: {self.llm.model}")
            
            # Retrieve relevant information from knowledge base
            try:
                retrieval_result = await self.knowledge_tool.execute(
                    query=query,
                    top_k=settings.top_k,
                    threshold=settings.distance
                )
                
                if retrieval_result.error:
                    logger.warning(f"Knowledge base retrieval failed: {retrieval_result.error}")
                    knowledge_context = "No relevant knowledge base information obtained, please generate content based on common sense and professional knowledge."
                else:
                    knowledge_context = retrieval_result.output
            except Exception as e:
                logger.error(f"Knowledge retrieval exception: {e}")
                knowledge_context = "Knowledge base retrieval encountered problems, please generate content based on professional judgment."
            
            # Build knowledge context prompt
            knowledge_prompt = f"""Relevant Knowledge Base Information:
{knowledge_context}

Please generate content for this section based on the above knowledge and your system instructions."""
            if self.llm.model in ["Qwen/Qwen3-4B", "Qwen/Qwen3-32B"]:
                logger.info(f"Using Qwen model, disabling think tag in prompt")
                knowledge_prompt = knowledge_prompt + "\no_think<think>\n\n</think>" # disable think tag for Qwen models
            self.update_memory("user", knowledge_prompt)
            
            # Call LLM to generate content
            response = await self.llm.ask(self.memory.messages)
            
            if response:
                if "</think>" in response:
                    response = response.split("</think>")[1].strip()
                self.generated_content = response.strip()
                self.is_completed = True
                self.state = AgentState.FINISHED
                self.update_memory("assistant", self.generated_content)
                
                logger.info(f"Section '{section_title}' content generation completed, length: {len(self.generated_content)}")
                logger.info("=== Generated Content ===")
                logger.info(self.generated_content)
                logger.info("=========================")
                return f"Section '{section_title}' generation completed"
            else:
                logger.warning(f"LLM returned no valid content for section: {section_title}")
                self.generated_content = f"[No content generated] LLM returned empty response for section: {section_title}"
                self.is_completed = True
                self.state = AgentState.FINISHED
                return f"Section '{section_title}' completed with no content from LLM"
                
        except Exception as e:
            logger.error(f"Section generation step execution failed: {e}")
            self.state = AgentState.ERROR
            return f"Section generation failed: {str(e)}"
    
    def get_content(self) -> str:
        """
        Get the generated content.
        
        Returns:
            str: The generated section content.
        """
        return self.generated_content
    
    def is_finished(self) -> bool:
        """
        Check if the section generation is finished.
        
        Returns:
            bool: True if the section generation is complete and the agent is finished.
        """
        return self.is_completed and self.state == AgentState.FINISHED
    
