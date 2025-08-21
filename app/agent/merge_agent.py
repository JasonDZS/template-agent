#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Agent based on ToolCallAgent.

This module provides a merge agent that merges content from child sections
using ToolCallAgent architecture. It's specialized for merging tasks only.
"""

from typing import Dict, Any, List
from pydantic import Field
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection, Terminate
from app.prompt.section_agent import NEXT_STEP_PROMPT, TaskPrompts
from app.schema import AgentState
from app.logger import logger


class MergeAgent(ToolCallAgent):
    """
    Merge agent specialized for merging content from child sections.
    
    This agent supports two modes:
    1. Model-based merging: Uses LLM to intelligently merge child contents
    2. Simple concatenation: Directly concatenates child contents with separators
    
    The mode is controlled by the enable_model_merge parameter.
    """
    
    # Pydantic fields
    section_info: Dict[str, Any] = Field(default_factory=dict)
    report_context: Dict[str, Any] = Field(default_factory=dict)
    knowledge_base_path: str = Field(default="workdir/documents")
    child_contents: List[str] = Field(default_factory=list)
    generated_content: str = Field(default="")
    is_content_complete: bool = Field(default=False)
    enable_model_merge: bool = Field(default=True)

    def __init__(self, 
                 section_info: Dict[str, Any],
                 report_context: Dict[str, Any],
                 child_contents: List[str],
                 knowledge_base_path: str = "workdir/documents",
                 enable_model_merge: bool = True,
                 **kwargs):
        """
        Initialize the MergeAgent.
        
        Args:
            section_info: Information about the section to merge
            report_context: Context about the overall report
            child_contents: List of content from child sections to merge
            knowledge_base_path: Path to the knowledge base directory
            enable_model_merge: Whether to use model for intelligent merging or simple concatenation
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
        
        # Set field values
        self.section_info = section_info
        self.report_context = report_context  
        self.knowledge_base_path = knowledge_base_path
        self.child_contents = child_contents
        self.generated_content = ""
        self.is_content_complete = False
        self.enable_model_merge = enable_model_merge
        
        # Setup based on merge mode
        if self.enable_model_merge:
            # Set up merge prompt with child content for model-based merging
            self.system_prompt = TaskPrompts.MERGE_PROMPT.format(
                section_title=section_title,
                report_title=report_title,
                section_level=section_level,
                child_count=len(child_contents)
            )
            
            # Add child content to the system prompt
            if child_contents:
                child_content_text = "\n\n".join([
                    f"Sub-content {i+1}:\n{content}" 
                    for i, content in enumerate(child_contents)
                ])
                self.system_prompt += f"\n\nSub-contents to merge:\n{child_content_text}"
            
            logger.info(f"System prompt for merge section '{section_title}'")
            self.next_step_prompt = NEXT_STEP_PROMPT
            
            # Initialize tools for model-based merging
            terminate_tool = Terminate()
            self.available_tools = ToolCollection(terminate_tool)
        else:
            # For simple concatenation, no tools or prompts needed
            self.system_prompt = ""
            self.next_step_prompt = ""
            self.available_tools = None
        
        logger.info(f"Initialized MergeAgent: {section_title} (model_merge={enable_model_merge})")

    async def execute_tool(self, command) -> str:
        """Override tool execution method to handle merge logic."""
        result = await super().execute_tool(command)

        # If this is terminate tool, handle content completion
        if command.function.name == "terminate":
            self.is_content_complete = True
            logger.info(f"Merge section merging completed")
        
        return result
    
    async def think(self) -> bool:
        """Override thinking method with merge logic."""
        # Continue thinking if final content hasn't been merged and agent isn't finished
        if not self.is_content_complete and self.state != AgentState.FINISHED:
            return await super().think()
        
        return False
    
    def get_final_content(self) -> str:
        """Extract the final merged content from conversation history."""
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
                    len(content) > 50):
                    return content
        
        return ""
    
    def _simple_concatenate(self) -> str:
        """
        Perform simple concatenation of child contents.
        
        Returns:
            str: The concatenated content.
        """
        if not self.child_contents:
            logger.warning("No child contents to concatenate")
            return ""
        
        # Filter out empty contents
        valid_contents = [content.strip() for content in self.child_contents if content.strip()]
        
        if not valid_contents:
            logger.warning("No valid child contents to concatenate")
            return ""
        
        # Simple concatenation with double newlines as separator
        concatenated = "\n\n".join(valid_contents)
        
        logger.info(f"Simple concatenation completed, total length: {len(concatenated)}")
        return concatenated

    async def run_merge(self) -> str:
        """Run the merge task."""
        try:
            section_title = self.section_info.get('content', '')
            logger.info(f"Starting section merge: {section_title} (model_merge={self.enable_model_merge})")
            
            if not self.enable_model_merge:
                # Simple concatenation mode
                self.generated_content = self._simple_concatenate()
                self.is_content_complete = True
                self.state = AgentState.FINISHED
                
                logger.info(f"Simple concatenation completed, content length: {len(self.generated_content)}")
                return self.generated_content
            else:
                # Model-based merging mode
                # Run agent until completion
                result = await self.run()
                
                # Get the merged content
                final_content = self.get_final_content()
                
                if final_content:
                    self.generated_content = final_content
                    logger.info(f"Model-based merge successful, content length: {len(final_content)}")
                    return final_content
                else:
                    logger.warning(f"Model-based merge completed but no valid content found")
                    return f"[Section merge incomplete] {section_title}"
            
        except Exception as e:
            logger.error(f"Section merge failed: {e}")
            self.state = AgentState.ERROR
            raise
    
    def get_content(self) -> str:
        """Get the merged content."""
        return self.generated_content
    
    def is_finished(self) -> bool:
        """Check if the merge is finished."""
        return self.is_content_complete and self.state == AgentState.FINISHED