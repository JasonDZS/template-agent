"""
Template Report Generation Service

A comprehensive service for generating reports from markdown templates with 
streaming output support and concurrent task execution.
"""

from .models import (
    ReportGenerationRequest, 
    ReportTask, 
    TaskStatus, 
    StreamMessage,
    TaskProgress,
    ReportGenerationResponse
)

from .report_service import ReportGenerationService
from .streaming_executors import StreamingSectionExecutor, StreamingMergeExecutor

__all__ = [
    'ReportGenerationRequest',
    'ReportTask', 
    'TaskStatus',
    'StreamMessage',
    'TaskProgress',
    'ReportGenerationResponse',
    'ReportGenerationService',
    'StreamingSectionExecutor',
    'StreamingMergeExecutor'
]