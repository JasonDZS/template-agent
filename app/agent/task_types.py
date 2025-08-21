#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task types for different agents.

This module defines the task types that can be handled by different agents
in the system.
"""

from enum import Enum


class TaskType(str, Enum):
    """Task types for different agents."""
    GENERATION = "generation"  # Content generation tasks
    MERGE = "merge"           # Content merging tasks
    ANALYSIS = "analysis"     # Data analysis tasks
    SUMMARY = "summary"       # Content summarization tasks