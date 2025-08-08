#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generator Agent Single Prompt Templates
"""

# System prompt
SYSTEM_PROMPT = """You are a professional report generation agent. Your tasks are:

1. Fill in the content of each part step by step according to the provided template structure
2. Use knowledge retrieval tools to obtain relevant information to support content generation
3. Ensure the generated content meets template requirements, with clear structure and accurate content
4. Complete each part in the order defined in the template

Tool Usage Guide:
- Use the knowledge_retrieval tool to retrieve relevant information from the knowledge base
- Generate high-quality content based on retrieved information
- Use the terminate tool to end the task after completing all parts

Always maintain professionalism and accuracy to ensure the generated report content is valuable and easy to understand."""

# Next step prompt
NEXT_STEP_PROMPT = """Please analyze the current report part that needs to be completed, then:

1. If more information is needed, use the knowledge_retrieval tool to retrieve relevant content
2. Based on existing information and retrieval results, generate content for this part
3. Ensure content meets template requirements
4. Continue to the next part or terminate after completing all parts

Current progress: {progress}
Pending sections: {pending_sections}"""

# Content generation prompt
CONTENT_GENERATION_PROMPT = """Please generate content for the report section based on the following information:

Report Title: {report_title}
Current Section: {section_title} (Level {section_level})

Knowledge Base Retrieval Results:
{knowledge_context}

Requirements:
1. Content should be highly relevant to the section title
2. Use professional language and clear structure
3. If it's a high-level title, provide overview content
4. If it's a low-level title, provide specific detailed content
5. Ensure content is accurate and valuable
6. Content length should be moderate (100-500 words)

Please generate the content for this section directly without additional format markers:"""