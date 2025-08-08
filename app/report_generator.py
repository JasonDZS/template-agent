#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generator Main Module

This module integrates Agent, Tool, and Converter functionality to provide
a comprehensive report generation system with template-based automation.
"""

import asyncio
import argparse
from pathlib import Path
from typing import Optional

from app.agent.report_multi import ReportGeneratorAgent
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.converter import MarkdownConverter, ConversionRequest
from app.logger import logger
from app.config import settings


class ReportGenerationSystem:
    """Comprehensive report generation system.
    
    This class provides a high-level interface for generating reports using
    templates, knowledge bases, and AI agents. It handles the entire workflow
    from template processing to final report output.
    
    Attributes:
        knowledge_base_path: Path to the knowledge base directory
        template_base_path: Path to the template directory
        converter: Markdown converter instance
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "workdir/documents",
                 template_base_path: str = "workdir/template"):
        """Initialize the report generation system.
        
        Args:
            knowledge_base_path: Directory containing knowledge base documents
            template_base_path: Directory containing report templates
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.template_base_path = Path(template_base_path)
        self.converter = MarkdownConverter()
        
        # Ensure directories exist
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.template_base_path.mkdir(parents=True, exist_ok=True)
    
    async def generate_report(self, 
                            template_name: str,
                            output_name: Optional[str] = None,
                            max_steps: int = 20) -> str:
        """Generate a report using the specified template.
        
        Args:
            template_name: Name or path of the template to use
            output_name: Optional custom output filename
            max_steps: Maximum number of execution steps
            
        Returns:
            Success message with generation details
            
        Raises:
            FileNotFoundError: If template file is not found
            Exception: If report generation fails
        """
        try:
            # Find template file
            template_path = self._find_template(template_name)
            if not template_path:
                raise FileNotFoundError(f"Template file not found: {template_name}")
            
            # Set output path
            if not output_name:
                output_name = f"generated_report_{template_path.stem}"
            
            output_path = Path("workdir") / "output" / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create and configure Agent
            agent = ReportGeneratorAgent(
                name=f"report_generator_{template_path.stem}",
                description=f"Generate report based on template {template_path.name}",
                template_path=str(template_path),
                knowledge_base_path=str(self.knowledge_base_path),
                output_path=str(output_path),
                max_steps=max_steps,
                parallel_sections=False,
                max_concurrent=settings.max_concurrent,
            )
            
            logger.info(f"Starting report generation with template: {template_path.name}")
            logger.info(f"Knowledge base path: {self.knowledge_base_path}")
            logger.info(f"Output path: {output_path}")
            
            # Run Agent
            result = await agent.run_with_template(str(template_path), str(output_path))
            
            # Get progress information
            progress = agent.get_progress()
            
            success_message = f"""Report generation completed!

Template: {template_path.name}
Progress: {progress['completed_sections']}/{progress['total_sections']} 
          ({progress['progress_percentage']:.1f}%)
Output: {output_path}.json, {output_path}.md

Detailed execution result:
{result}"""
            
            logger.info("Report generation task completed")
            return success_message
            
        except Exception as e:
            error_message = f"Report generation failed: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)
    
    def _find_template(self, template_name: str) -> Optional[Path]:
        """Find template file by name or path.
        
        Args:
            template_name: Template name or path to search for
            
        Returns:
            Path to the template file if found, None otherwise
        """
        # Direct path
        if Path(template_name).exists():
            return Path(template_name)
        
        # Search in template directory
        possible_paths = [
            self.template_base_path / template_name,
            self.template_base_path / f"{template_name}.md",
            self.template_base_path / f"{template_name}.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    async def convert_template(self, 
                             template_path: str,
                             output_format: str = "json") -> str:
        """Convert template between formats.
        
        Args:
            template_path: Path to the template file
            output_format: Target format (json or markdown)
            
        Returns:
            Success message with conversion details
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            Exception: If conversion fails
        """
        try:
            template_file = Path(template_path)
            if not template_file.exists():
                raise FileNotFoundError(f"Template file does not exist: {template_path}")
            
            # Read template content
            template_content = template_file.read_text(encoding='utf-8')
            
            # Determine source format
            source_format = "markdown" if template_content.strip().startswith('#') else "json"
            
            # Execute conversion
            conversion_request = ConversionRequest(
                source_format=source_format,
                target_format=output_format,
                content=template_content
            )
            
            result = self.converter.convert(conversion_request)
            
            if result.success:
                # Save conversion result
                output_path = template_file.with_suffix(f'.{output_format}')
                if output_format == "json":
                    import json
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result.result, f, ensure_ascii=False, indent=2)
                else:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(result.result)
                
                return f"Template conversion successful: {output_path}"
            else:
                raise Exception(result.error)
                
        except Exception as e:
            error_message = f"Template conversion failed: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)
    
    async def test_knowledge_base(self, query: str = "智能鞋垫") -> str:
        """Test knowledge base retrieval functionality.
        
        Args:
            query: Query string to test with
            
        Returns:
            Test results with statistics and sample retrieval
        """
        try:
            knowledge_tool = KnowledgeRetrievalTool(str(self.knowledge_base_path))
            result = await knowledge_tool.execute(query=query, top_k=3)
            
            if result.error:
                return f"Knowledge base retrieval test failed: {result.error}"
            else:
                stats = knowledge_tool.get_statistics()
                return f"""Knowledge base retrieval test successful!

Statistics:
- Total documents: {stats['total_documents']}
- Total size: {stats['total_size']} characters
- Average size: {stats['average_size']} characters
- File types: {stats['file_types']}

Retrieval results (query: "{query}"):
{result.output}"""
                
        except Exception as e:
            error_message = f"Knowledge base test failed: {str(e)}"
            logger.error(error_message)
            return error_message
    
    def list_templates(self) -> str:
        """List available templates in the template directory.
        
        Returns:
            Formatted string listing all available templates
        """
        templates = []
        
        if self.template_base_path.exists():
            for file_path in self.template_base_path.iterdir():
                if file_path.is_file() and file_path.suffix in ['.md', '.json']:
                    templates.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "type": file_path.suffix[1:]
                    })
        
        if not templates:
            return "No available templates found"
        
        result = "Available templates:\n"
        for i, template in enumerate(templates, 1):
            result += f"{i}. {template['name']} ({template['type']}, {template['size']} bytes)\n"
        
        return result


async def main():
    """Command-line entry point for the report generation system.
    
    Provides a CLI interface for generating reports, converting templates,
    testing knowledge bases, and listing available templates.
    """
    parser = argparse.ArgumentParser(description="Intelligent Report Generator")
    parser.add_argument("action", choices=["generate", "convert", "test", "list"], 
                       help="Action to perform")
    parser.add_argument("--template", "-t", help="Template name or path")
    parser.add_argument("--output", "-o", help="Output filename")
    parser.add_argument("--format", "-f", default="json", 
                       choices=["json", "markdown"], help="Conversion target format")
    parser.add_argument("--query", "-q", default="智能鞋垫", help="Test query string")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum execution steps")
    parser.add_argument("--knowledge-base", default="workdir/documents", 
                       help="Knowledge base directory path")
    parser.add_argument("--template-base", default="workdir/template", 
                       help="Template base directory path")
    
    args = parser.parse_args()
    
    # Create report generation system
    system = ReportGenerationSystem(
        knowledge_base_path=args.knowledge_base,
        template_base_path=args.template_base
    )
    
    try:
        if args.action == "generate":
            if not args.template:
                print("Error: Report generation requires template specification (--template)")
                return
            
            result = await system.generate_report(
                template_name=args.template,
                output_name=args.output,
                max_steps=args.max_steps
            )
            print(result)
            
        elif args.action == "convert":
            if not args.template:
                print("Error: Template conversion requires template path specification (--template)")
                return
            
            result = await system.convert_template(
                template_path=args.template,
                output_format=args.format
            )
            print(result)
            
        elif args.action == "test":
            result = await system.test_knowledge_base(query=args.query)
            print(result)
            
        elif args.action == "list":
            result = system.list_templates()
            print(result)
            
    except Exception as e:
        print(f"Execution failed: {e}")
        logger.error(f"Command execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())