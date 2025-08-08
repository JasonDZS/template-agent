#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section Agent Prompt Templates

This module contains prompt templates for section agents that are responsible
for generating individual sections of structured reports. It provides system
prompts and next step prompts with proper formatting and instructions.
"""

def get_system_prompt(section_title: str, report_title: str, section_level: int, section_id: int) -> str:
    """
    Generate system prompt for section agent.
    
    This function creates a detailed system prompt that instructs the section agent
    on its role, responsibilities, and the specific requirements for generating
    content for a particular report section.
    
    Args:
        section_title (str): The title of the section to be generated
        report_title (str): The title of the overall report
        section_level (int): The hierarchical level of the section (1-6)
        section_id (int): Unique identifier for the section
        
    Returns:
        str: Formatted system prompt for the section agent
        
    Example:
        >>> prompt = get_system_prompt("Executive Summary", "Quarterly Report", 1, 0)
        >>> print(prompt[:50])
        You are an intelligent agent specifically responsible...
    """
    return f"""You are an intelligent agent specifically responsible for generating the report section "{section_title}".

Report Background Information:
- Report Title: {report_title}
- Current Section: {section_title} (Level {section_level})
- Section ID: {section_id}

Your Tasks:
1. Deeply understand the role and positioning of this section in the overall report
2. Use the knowledge_retrieval tool to obtain relevant information, multiple retrievals are allowed
3. Generate high-quality, structured section content based on retrieved information
4. Ensure content is highly relevant to the section title and consistent with the overall report style
5. Use the terminate tool to end the task after completing content generation

Content Generation Requirements:
- For high-level titles (levels 1-2), provide overview and strategic content
- For low-level titles (level 3 and above), provide specific detailed implementation content
- Moderate content length (200-800 words)
- Use professional language, ensure accuracy and readability
- Include data, cases, or specific recommendations when necessary

Tool Usage Guide:
- knowledge_retrieval: Retrieve relevant information from the knowledge base, can be used multiple times with different query terms

Always maintain professionalism and ensure the generated content is valuable and easy to understand."""

# Next step prompt for section agent workflow
NEXT_STEP_PROMPT = """
Please analyze the current progress of section generation, then decide on the next action:

1. If more relevant information is still needed, use the knowledge_retrieval tool to retrieve
2. If sufficient information is available, directly generate the content for this section

Please choose the appropriate tool to continue execution based on the current situation.
"""