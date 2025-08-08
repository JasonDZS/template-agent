#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Generation Agent based on ToolCallAgent.

This module provides a section generation agent that supports multiple knowledge base
retrievals until the section content is complete.
"""

from typing import Dict, Any, Optional
from pydantic import Field
from app.agent.toolcall import ToolCallAgent
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.tool import ToolCollection, Terminate
from app.prompt.section_agent import get_system_prompt, NEXT_STEP_PROMPT
from app.schema import AgentState
from app.logger import logger


class SectionAgentReAct(ToolCallAgent):
    """
    Section generation agent based on ToolCall, supporting multiple knowledge base retrievals.
    
    This agent is designed to generate specific sections of a report by leveraging
    knowledge retrieval tools. It can perform multiple iterations of knowledge
    searches until the section content is complete.
    
    Attributes:
        section_info (Dict[str, Any]): Information about the section to generate.
        report_context (Dict[str, Any]): Context information about the overall report.
        knowledge_base_path (str): Path to the knowledge base for retrieval.
        generated_content (str): The final generated content for the section.
        is_content_complete (bool): Flag indicating if content generation is complete.
    """
    
    # Pydantic fields
    section_info: Dict[str, Any] = Field(default_factory=dict)
    report_context: Dict[str, Any] = Field(default_factory=dict)
    knowledge_base_path: str = Field(default="workdir/documents")
    generated_content: str = Field(default="")
    is_content_complete: bool = Field(default=False)
    
    def __init__(self, 
                 section_info: Dict[str, Any],
                 report_context: Dict[str, Any],
                 knowledge_base_path: str = "workdir/documents",
                 **kwargs):
        """
        Initialize the SectionAgentReAct.
        
        Args:
            section_info (Dict[str, Any]): Information about the section including
                title, level, id, etc.
            report_context (Dict[str, Any]): Context about the overall report
                including title and other metadata.
            knowledge_base_path (str, optional): Path to the knowledge base directory.
                Defaults to "workdir/documents".
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        
        section_title = section_info.get("content", "Untitled Section")
        section_level = section_info.get("level", 1)
        section_id = section_info.get("id", 0)
        report_title = report_context.get("title", "")
        
        # Set basic information
        name = f"section_{section_id}_{section_title[:10]}"
        description = f"Dedicated agent for generating section '{section_title}'"
        
        # Initialize parent class
        super().__init__(
            name=name,
            description=description,
            **kwargs
        )
        
        # Set field values
        self.section_info = section_info
        self.report_context = report_context  
        self.knowledge_base_path = knowledge_base_path
        self.generated_content = ""
        self.is_content_complete = False
        
        # Set up prompts
        self.system_prompt = get_system_prompt(
            section_title=section_title,
            report_title=report_title,
            section_level=section_level,
            section_id=section_id
        )
        self.next_step_prompt = NEXT_STEP_PROMPT
        
        # Initialize tools
        knowledge_tool = KnowledgeRetrievalTool(knowledge_base_path)
        terminate_tool = Terminate()
        
        self.available_tools = ToolCollection(
            knowledge_tool,
            terminate_tool
        )
        
        logger.info(f"Initialized SectionAgent: {section_title}")
    
    async def execute_tool(self, command) -> str:
        """
        Override tool execution method to handle content generation logic.
        
        Args:
            command: The tool command to execute.
            
        Returns:
            str: The result of the tool execution.
        """
        result = await super().execute_tool(command)
        
        # If this is a knowledge retrieval tool result, log retrieval information
        if command.function.name == "knowledge_retrieval":
            logger.info(f"Section '{self.section_info.get('content', '')}' completed knowledge retrieval")
        
        # If this is terminate tool, handle content completion
        elif command.function.name == "terminate":
            self.is_content_complete = True
            logger.info(f"Section '{self.section_info.get('content', '')}' generation completed")
        
        return result
    
    async def think(self) -> bool:
        """
        Override thinking method with content generation logic.
        
        Returns:
            bool: True if the agent should continue thinking, False otherwise.
        """
        # Continue thinking if final content hasn't been generated and agent isn't finished
        if not self.is_content_complete and self.state != AgentState.FINISHED:
            return await super().think()
        
        return False
    
    def get_final_content(self) -> str:
        """
        Extract the final generated content from conversation history.
        
        This method searches through the conversation history to find the actual
        section content, filtering out tool usage instructions and other metadata.
        
        Returns:
            str: The final generated section content, or empty string if not found.
        """
        if not self.memory or not self.memory.messages:
            return ""
        
        # Find the last assistant message with content
        for message in reversed(self.memory.messages):
            if message.role == "assistant" and message.content:
                content = message.content.strip()
                # Filter out tool-related content
                if (not content.startswith("我需要") and 
                    not content.startswith("让我") and
                    not content.startswith("现在我") and
                    len(content) > 50):  # Ensure it's actual section content, not brief tool instructions
                    return content
        
        return ""
    
    async def run_section_generation(self) -> str:
        """
        Run the section generation task.
        
        This method orchestrates the entire section generation process, including
        running the agent, extracting the final content, and handling errors.
        
        Returns:
            str: The generated section content.
            
        Raises:
            Exception: If section generation fails.
        """
        try:
            logger.info(f"Starting section generation: {self.section_info.get('content', '')}")
            
            # Run agent until completion
            result = await self.run()
            
            # Get the generated content
            final_content = self.get_final_content()
            
            if final_content:
                self.generated_content = final_content
                logger.info(f"Section generation successful, content length: {len(final_content)}")
                return final_content
            else:
                logger.warning(f"Section generation completed but no valid content found")
                return f"[Section generation incomplete] {self.section_info.get('content', '')}"
            
        except Exception as e:
            logger.error(f"Section generation failed: {e}")
            self.state = AgentState.ERROR
            raise
    
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
        return self.is_content_complete and self.state == AgentState.FINISHED