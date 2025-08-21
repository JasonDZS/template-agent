#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for Template Report Generation Service.

This script provides a convenient way to start the service with various options.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.service.cli import cli


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Template Report Generation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the API server
  python run_service.py serve --port 8000
  
  # Generate a report directly
  python run_service.py generate template.md --concurrent 3
  
  # Check service status
  python run_service.py status
  
  # Start with custom configuration
  python run_service.py serve --host 0.0.0.0 --port 8080 --reload

For more detailed help on any command, use:
  python run_service.py COMMAND --help
        """
    )
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        # Start with default serve command
        sys.argv.extend(['serve', '--help'])
    
    # Use click CLI
    try:
        cli()
    except KeyboardInterrupt:
        print("\nService stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()