#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Generation Agent.

This module provides a basic section generation agent that creates individual
sections of a report using knowledge retrieval and LLM generation.
"""

from typing import Dict, Any, Optional
from app.agent.base import BaseAgent
from app.config import settings
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.schema import AgentState
from app.logger import logger


class SectionAgent(BaseAgent):
    """
    Single section generation agent.
    
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
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        
        self.section_info = section_info
        self.report_context = report_context
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize tools
        self.knowledge_tool = KnowledgeRetrievalTool(knowledge_base_path)
        
        # Generated content
        self.generated_content = ""
        self.is_completed = False
        
        # Set agent default parameters
        section_title = section_info.get("content", "Untitled Section")
        if not self.description:
            self.description = f"Dedicated agent for generating section '{section_title}'"
        
        # Set system prompt
        self.system_prompt = f"""You are an agent specialized in generating the report section "{section_title}".

Report Background Information:
- Report Title: {report_context.get('title', '')}
- Current Section: {section_title} (Level {section_info.get('level', 1)})
- Section ID: {section_info.get('id', 0)}

Your Tasks:
1. Thoroughly understand the role and positioning of this section within the overall report
2. Use the knowledge_retrieval tool to obtain relevant information
3. Generate high-quality, structured section content
4. Ensure content is highly relevant to the section title and fits the overall report style

Content Requirements:
- If high-level heading (1-2 levels): provide comprehensive, strategic content
- If low-level heading (3+ levels): provide specific, detailed implementation content
- Moderate content length (200-800 words)
- Use professional language, ensuring accuracy and readability
- Include data, cases, or specific recommendations when necessary

Completion Conditions:
- Use the terminate tool to end the task after generating complete section content
"""
    
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
            
            logger.info(f"Starting section generation: {section_title}")
            
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
            
            # Build content generation prompt
            content_prompt = f"""Please generate professional content for the following section:

Section Title: {section_title}
Section Level: {self.section_info.get('level', 1)}
Report Title: {report_title}

Relevant Knowledge Base Information:
{knowledge_context}

Please generate content for this section based on the above information. Requirements:

1. Content Structure:
   - If 1-2 level heading: provide overview, importance, and main points for this section
   - If 3+ level heading: provide specific implementation details, methods, or cases

2. Content Quality:
   - Ensure high relevance to the section title
   - Content should be accurate, professional, and valuable
   - Language should be clear and understandable with strong logic
   - Moderate length (200-800 words)

3. Format Requirements:
   - Output section body content directly
   - Do not include section title (will be added automatically)
   - May use sub-headings, lists, and other formats
   - Include specific data or recommendations when necessary

Please start generating the content for this section:"""
            
            self.update_memory("user", content_prompt)
            
            # Call LLM to generate content
            response = await self.llm.ask(self.memory.messages)
            
            if response and response.strip():
                self.generated_content = response.strip()
                self.is_completed = True
                self.update_memory("assistant", self.generated_content)
                
                logger.info(f"Section '{section_title}' content generation completed, length: {len(self.generated_content)}")
                return f"Section '{section_title}' generation completed"
            else:
                logger.warning(f"LLM returned no valid content for section: {section_title}")
                return f"Generating content for section '{section_title}'..."
                
        except Exception as e:
            logger.error(f"Section generation step execution failed: {e}")
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
    
    async def run_section_generation(self) -> str:
        """
        Run the section generation task.
        
        This method orchestrates the complete section generation process by
        running the agent until completion and returning the generated content.
        
        Returns:
            str: The generated section content.
            
        Raises:
            Exception: If section generation fails.
        """
        try:
            # Run agent until completion
            while not self.is_finished():
                result = await self.step()
                if self.state == AgentState.ERROR:
                    break
            
            return self.generated_content
            
        except Exception as e:
            logger.error(f"Section generation execution failed: {e}")
            self.state = AgentState.ERROR
            raise