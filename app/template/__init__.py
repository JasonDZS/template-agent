"""Template Agent Module.

This module provides a structured approach to document processing with clear
separation of concerns:

- document_types: Core data types for markdown documents
- task_models: Task scheduling and execution models
- task_executors: Task execution implementations
- task_scheduler: Task scheduling and concurrency management
- document_scheduler: Document-specific scheduling integration
- converters: Utilities for format conversion
"""

# Core document types
from .document_types import (
    NodeType, MarkdownNode, HeadingNode, ParagraphNode, 
    ListNode, ListItemNode, CodeBlockNode, MarkdownDocument
)

# Task system
from .task_models import (
    TaskStatus, TaskType, TaskResult, TaskInput, TaskOutput, Task
)

from .task_executors import (
    TaskExecutor, GenerationTaskExecutor, MergeTaskExecutor, ExecutorRegistry
)

from .task_scheduler import TaskScheduler

from .document_scheduler import MarkdownTaskSchedule, create_task_schedule

# Conversion utilities
from .converters import (
    parse_markdown_to_document_tree,
    parse_markdown_file_to_document_tree,
    parse_markdown_with_metadata,
    extract_document_info,
    batch_parse_markdown_files,
    convert_json_to_document_tree,
    convert_document_tree_to_json
)

# Conversion utilities
from .converters import parse_markdown_file_to_document_tree

__all__ = [
    # Document types
    'NodeType', 'MarkdownNode', 'HeadingNode', 'ParagraphNode',
    'ListNode', 'ListItemNode', 'CodeBlockNode', 'MarkdownDocument',
    
    # Task system
    'TaskStatus', 'TaskType', 'TaskResult', 'TaskInput', 'TaskOutput', 'Task',
    'TaskExecutor', 'GenerationTaskExecutor', 'MergeTaskExecutor', 'ExecutorRegistry',
    'TaskScheduler',
    
    # Document scheduling
    'MarkdownTaskSchedule', 'create_task_schedule',
    
    # Converters
    'parse_markdown_to_document_tree', 'parse_markdown_file_to_document_tree',
    'parse_markdown_with_metadata', 'extract_document_info',
    'batch_parse_markdown_files', 'convert_json_to_document_tree',
    'convert_document_tree_to_json'
]