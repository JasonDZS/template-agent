#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generator Agent Prompt Templates

This module contains prompt templates for the multi-agent report generator
that coordinates multiple section agents to create comprehensive structured reports.
It includes prompts for quality assessment, content polishing, consistency checking,
and final quality validation.
"""

# System prompt for the multi-agent report generator coordinator
SYSTEM_PROMPT = """You are a professional report generation coordinator agent. Your tasks are:

1. Coordinate multiple SectionAgents to generate report content by sections
2. Manage the order and concurrency control of section generation
3. Monitor the generation progress and quality of each section agent
4. Integrate all section content into a complete report
5. Ensure the report structure is complete and content is coherent

Working modes:
- Serial mode: Generate sections sequentially to ensure contextual coherence
- Parallel mode: Generate multiple sections simultaneously to improve efficiency

Always monitor section generation quality to ensure the final report is professional, accurate, and valuable."""

# Content quality assessment prompt for evaluating section content
QUALITY_ASSESSMENT_PROMPT = """Please assess the content quality of the following report section:

Section Title: {section_title}
Section Content:
{content}

Please evaluate quality from the following dimensions (1-5 points, 5 being the highest):
1. Content accuracy and professionalism
2. Clarity of language expression
3. Reasonableness of logical structure
4. Completeness of information
5. Relevance to section title

If obvious issues are found, please specify what the problems are. Please return the assessment results in JSON format:
{{
    "overall_score": score(1-5),
    "accuracy_score": accuracy score(1-5),
    "clarity_score": clarity score(1-5),
    "structure_score": structure score(1-5),
    "completeness_score": completeness score(1-5),
    "relevance_score": relevance score(1-5),
    "issues": ["issue1", "issue2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...]
}}"""

# Content polishing prompt for improving section quality
POLISH_CONTENT_PROMPT = """Please polish and optimize the following section content:

Section Title: {section_title}
Original Content:
{original_content}

Issues Found:
{issues}

Optimization Suggestions:
{suggestions}

Please provide polished content with the following requirements:
1. Maintain the original core information and viewpoints
2. Improve the professionalism and clarity of language expression
3. Optimize logical structure and paragraph organization
4. Ensure content completeness and accuracy
5. Enhance relevance to the section title

Please output the polished content directly without additional explanations:"""

# Consistency check prompt for ensuring report coherence
CONSISTENCY_CHECK_PROMPT = """Please check the content consistency of the following report:

Report Title: {report_title}

Section Summaries:
{section_summaries}

Full Content: {full_content}

Please check for consistency issues in the following aspects:
1. Whether terminology usage is consistent
2. Whether data references are consistent
3. Whether viewpoint expressions contradict each other
4. Whether logical relationships between sections are reasonable
5. Whether the overall discourse structure is coherent

If consistency issues are found, please return in JSON format:
{{
    "issues": [
        {{
            "type": "issue type",
            "description": "issue description",
            "affected_sections": ["related section1", "related section2"],
            "suggestion": "fix suggestion"
        }}
    ]
}}

If no issues are found, return: {{"issues": []}}"""

# Consistency issue fix prompt for resolving identified problems
FIX_CONSISTENCY_PROMPT = """Please fix the consistency issues in the following section content:

Section Title: {section_title}
Original Content:
{original_content}

Consistency Issue:
Type: {issue_type}
Description: {description}
Fix Suggestion: {suggestion}

Please provide the fixed content, ensuring:
1. Resolve the identified consistency issues
2. Maintain content accuracy and completeness
3. Keep consistency with other parts of the report

Please output the fixed content directly:"""

# Final quality check prompt for overall report validation
FINAL_QUALITY_CHECK_PROMPT = """Please conduct a final quality assessment of the following complete report:

{full_report}

Please evaluate the report quality from an overall perspective:
1. Structural completeness (whether all necessary content is covered)
2. Logical coherence (whether logic between sections is clear)
3. Content professionalism (whether it is professional and accurate)
4. Language quality (whether expression is clear)
5. Practical value (whether it is valuable to readers)

Please give an overall score (1-10 points) and a brief evaluation.
If the score is ≥8, answer "PASS"; otherwise answer "FAIL" and explain the main issues."""