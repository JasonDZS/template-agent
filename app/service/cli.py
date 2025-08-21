#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI tool for managing the Report Generation Service.

This module provides command-line utilities for starting the service,
managing jobs, and monitoring system status.
"""

import asyncio
import click
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.service.api import run_dev_server
from app.service.report_service import get_report_service
from app.service.models import ReportGenerationRequest
from app.logger import logger


@click.group()
def cli():
    """Template Report Generation Service CLI."""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the report generation service API server."""
    click.echo(f"Starting Report Generation Service on {host}:{port}")
    if reload:
        click.echo("Auto-reload enabled")
    
    try:
        run_dev_server(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        click.echo("\nService stopped by user")
    except Exception as e:
        click.echo(f"Error starting service: {e}")
        sys.exit(1)


@cli.command()
@click.argument('template_path')
@click.option('--knowledge-base', default='workdir/documents', help='Knowledge base path')
@click.option('--concurrent', default=3, help='Max concurrent tasks')
@click.option('--streaming/--no-streaming', default=True, help='Enable streaming output')
@click.option('--model-merge/--no-model-merge', default=True, help='Enable model-based merging')
@click.option('--report-title', help='Custom report title')
@click.option('--output-path', help='Custom output path')
def generate(template_path: str, knowledge_base: str, concurrent: int, 
             streaming: bool, model_merge: bool, report_title: Optional[str],
             output_path: Optional[str]):
    """Generate a report from a template (standalone mode)."""
    click.echo(f"Generating report from template: {template_path}")
    
    async def run_generation():
        service = get_report_service()
        await service.start()
        
        try:
            # Create request
            request = ReportGenerationRequest(
                template_path=template_path,
                knowledge_base_path=knowledge_base,
                max_concurrent=concurrent,
                enable_streaming=streaming,
                enable_model_merge=model_merge,
                report_title=report_title,
                output_path=output_path
            )
            
            # Create job
            click.echo("Creating job...")
            response = await service.create_report_job(request)
            
            if response.status != "created":
                click.echo(f"Failed to create job: {response.message}")
                return
            
            job_id = response.job_id
            click.echo(f"Job created: {job_id}")
            
            # Start job
            click.echo("Starting job execution...")
            start_response = await service.start_job(job_id)
            
            if start_response.status != "started":
                click.echo(f"Failed to start job: {start_response.message}")
                return
            
            # Monitor progress
            click.echo("Monitoring progress...")
            while True:
                status_response = await service.get_job_status(job_id)
                
                if status_response is None:
                    click.echo("Job not found")
                    break
                
                if status_response.status in ["completed", "failed", "cancelled"]:
                    if status_response.status == "completed":
                        click.echo(f"✅ Job completed successfully!")
                        if status_response.report_path:
                            click.echo(f"📄 Report saved: {status_response.report_path}")
                    else:
                        click.echo(f"❌ Job {status_response.status}")
                    break
                
                # Show progress
                if status_response.progress:
                    progress = status_response.progress
                    click.echo(f"Progress: {progress.completed_tasks}/{progress.total_tasks} tasks completed "
                             f"({progress.overall_progress:.1f}%)")
                
                await asyncio.sleep(2)
                
        except Exception as e:
            click.echo(f"Error during generation: {e}")
        finally:
            await service.stop()
    
    try:
        asyncio.run(run_generation())
    except KeyboardInterrupt:
        click.echo("\nGeneration cancelled by user")


@cli.command()
def status():
    """Show service status and active jobs."""
    click.echo("Report Generation Service Status")
    click.echo("=" * 40)
    
    async def show_status():
        service = get_report_service()
        
        # Basic service info
        click.echo(f"Service initialized: {'Yes' if service else 'No'}")
        click.echo(f"Active jobs: {len(service.jobs)}")
        click.echo(f"Active connections: {len(service.connections)}")
        click.echo()
        
        if service.jobs:
            click.echo("Active Jobs:")
            click.echo("-" * 20)
            
            for job_id, job in service.jobs.items():
                click.echo(f"Job ID: {job_id[:8]}...")
                click.echo(f"  Status: {job.status}")
                click.echo(f"  Tasks: {len(job.tasks)}")
                click.echo(f"  Started: {job.start_time.strftime('%Y-%m-%d %H:%M:%S') if job.start_time else 'N/A'}")
                
                if job.status == "running":
                    progress = await service._get_job_progress(job_id)
                    if progress:
                        click.echo(f"  Progress: {progress.completed_tasks}/{progress.total_tasks} "
                                 f"({progress.overall_progress:.1f}%)")
                
                click.echo()
    
    try:
        asyncio.run(show_status())
    except Exception as e:
        click.echo(f"Error getting status: {e}")


@cli.command()
@click.argument('job_id')
def cancel(job_id: str):
    """Cancel a running job."""
    async def cancel_job():
        service = get_report_service()
        response = await service.cancel_job(job_id)
        
        if response.status == "not_found":
            click.echo("Job not found")
        elif response.status == "cancelled":
            click.echo("Job cancelled successfully")
        else:
            click.echo(f"Cancel response: {response.message}")
    
    try:
        asyncio.run(cancel_job())
    except Exception as e:
        click.echo(f"Error cancelling job: {e}")


@cli.command()
def cleanup():
    """Clean up old completed jobs and files."""
    click.echo("Cleaning up old files...")
    
    # Clean up old reports
    output_dir = Path("workdir/output")
    if output_dir.exists():
        old_files = []
        cutoff = datetime.now().timestamp() - (24 * 60 * 60 * 7)  # 7 days ago
        
        for file_path in output_dir.glob("*.md"):
            if file_path.stat().st_mtime < cutoff:
                old_files.append(file_path)
        
        if old_files:
            click.confirm(f"Delete {len(old_files)} old report files?", abort=True)
            for file_path in old_files:
                file_path.unlink()
                click.echo(f"Deleted: {file_path.name}")
        else:
            click.echo("No old files to clean up")
    
    click.echo("Cleanup completed")


@cli.command()
@click.argument('config_file')
def config(config_file: str):
    """Load and validate configuration file."""
    config_path = Path(config_file)
    
    if not config_path.exists():
        click.echo(f"Config file not found: {config_file}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        click.echo("Configuration loaded successfully:")
        click.echo(json.dumps(config_data, indent=2))
        
        # Validate required fields
        required_fields = ['template_path', 'knowledge_base_path']
        missing_fields = [field for field in required_fields if field not in config_data]
        
        if missing_fields:
            click.echo(f"Warning: Missing required fields: {', '.join(missing_fields)}")
        else:
            click.echo("✅ Configuration is valid")
            
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error reading config file: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()