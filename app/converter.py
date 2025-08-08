#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown and JSON Conversion Core Module

This module provides comprehensive functionality for converting between
Markdown and JSON formats, including parsing, rendering, and structured
document representation.
"""

import json
import re
from typing import List, Dict, Any, Optional, Union
from app.type import (
    MarkdownDocument, MarkdownElement, HeadingElement, CodeBlockElement,
    ListElement, LinkElement, ImageElement, TableElement,
    ElementType, ListType, TableAlignment, ConversionOptions,
    ConversionRequest, ConversionResponse
)
from app.config import settings


class MarkdownParser:
    """Parser for converting Markdown text to structured document objects.
    
    This class provides comprehensive parsing capabilities for Markdown syntax,
    including headings, code blocks, lists, blockquotes, images, and metadata.
    
    Attributes:
        current_pos: Current position in the line array during parsing
        lines: Array of text lines being parsed
    """
    
    def __init__(self):
        """Initialize the Markdown parser."""
        self.current_pos = 0
        self.lines = []
    
    def parse(self, markdown_text: str) -> MarkdownDocument:
        """Parse Markdown text into a document object.
        
        Args:
            markdown_text: The Markdown text to parse
            
        Returns:
            MarkdownDocument object containing the parsed structure
        """
        self.lines = markdown_text.split('\n')
        self.current_pos = 0
        
        # Extract metadata
        metadata = self._extract_metadata()
        
        # Parse content
        elements = []
        while self.current_pos < len(self.lines):
            element = self._parse_next_element()
            if element:
                elements.append(element)
        
        # Get title (if first element is h1)
        title = None
        if elements and isinstance(elements[0], HeadingElement) and elements[0].level == 1:
            title = elements[0].content
        
        return MarkdownDocument(
            title=title,
            metadata=metadata,
            content=elements
        )
    
    def _extract_metadata(self) -> Optional[Dict[str, Any]]:
        """Extract metadata from HTML comments.
        
        Returns:
            Dictionary of metadata key-value pairs, or None if no metadata found
        """
        metadata = {}
        
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos].strip()
            
            # Check if this is a metadata comment
            if line.startswith('<!-- Metadata:'):
                self.current_pos += 1
                
                # Parse metadata content
                while self.current_pos < len(self.lines):
                    line = self.lines[self.current_pos].strip()
                    if line == '-->':
                        self.current_pos += 1
                        break
                    
                    # Parse key-value pairs
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                    
                    self.current_pos += 1
                break
            elif line:
                # Encountered non-empty line that's not metadata, stop searching
                break
            
            self.current_pos += 1
        
        return metadata if metadata else None
    
    def _parse_next_element(self) -> Optional[MarkdownElement]:
        """Parse the next element in the document.
        
        Returns:
            The next parsed MarkdownElement, or None if end of document
        """
        if self.current_pos >= len(self.lines):
            return None
        
        line = self.lines[self.current_pos].strip()
        
        # Skip empty lines
        if not line:
            self.current_pos += 1
            return self._parse_next_element()
        
        # Parse heading
        if line.startswith('#'):
            return self._parse_heading()
        
        # Parse code block
        if line.startswith('```'):
            return self._parse_code_block()
        
        # Parse list
        if re.match(r'^[\-\*\+]\s', line) or re.match(r'^\d+\.\s', line):
            return self._parse_list()
        
        # Parse blockquote
        if line.startswith('>'):
            return self._parse_blockquote()
        
        # Parse horizontal rule
        if re.match(r'^[\-\*_]{3,}$', line):
            self.current_pos += 1
            return MarkdownElement(type=ElementType.HORIZONTAL_RULE)
        
        # Parse image
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if img_match:
            return self._parse_image(img_match)
        
        # Default to paragraph
        return self._parse_paragraph()
    
    def _parse_heading(self) -> HeadingElement:
        """Parse a heading element.
        
        Returns:
            HeadingElement with appropriate level and content
        """
        line = self.lines[self.current_pos]
        self.current_pos += 1
        
        # Calculate heading level
        level = 0
        for char in line:
            if char == '#':
                level += 1
            else:
                break
        
        content = line[level:].strip()
        return HeadingElement(level=level, content=content)
    
    def _parse_code_block(self) -> CodeBlockElement:
        """Parse a fenced code block.
        
        Returns:
            CodeBlockElement with language and content
        """
        start_line = self.lines[self.current_pos]
        self.current_pos += 1
        
        # Extract language
        language = start_line[3:].strip() if len(start_line) > 3 else None
        
        # Collect code content
        code_lines = []
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos]
            if line.strip() == '```':
                self.current_pos += 1
                break
            code_lines.append(line)
            self.current_pos += 1
        
        content = '\n'.join(code_lines)
        return CodeBlockElement(language=language, content=content)
    
    def _parse_list(self) -> ListElement:
        """Parse a list (ordered or unordered).
        
        Returns:
            ListElement containing all list items
        """
        items = []
        list_type = None
        start_num = 1
        
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos].strip()
            
            # Check if this is a list item
            ordered_match = re.match(r'^(\d+)\.\s+(.*)$', line)
            unordered_match = re.match(r'^[\-\*\+]\s+(.*)$', line)
            
            if ordered_match:
                if list_type is None:
                    list_type = ListType.ORDERED
                    start_num = int(ordered_match.group(1))
                elif list_type != ListType.ORDERED:
                    break
                
                content = ordered_match.group(2)
                items.append(MarkdownElement(type=ElementType.LIST_ITEM, content=content))
                self.current_pos += 1
                
            elif unordered_match:
                if list_type is None:
                    list_type = ListType.UNORDERED
                elif list_type != ListType.UNORDERED:
                    break
                
                content = unordered_match.group(1)
                items.append(MarkdownElement(type=ElementType.LIST_ITEM, content=content))
                self.current_pos += 1
                
            elif line == '':
                # Empty line, continue checking next line
                self.current_pos += 1
                continue
            else:
                # Not a list item, end list parsing
                break
        
        return ListElement(
            list_type=list_type or ListType.UNORDERED,
            start=start_num if list_type == ListType.ORDERED else None,
            children=items
        )
    
    def _parse_blockquote(self) -> MarkdownElement:
        """Parse a blockquote element.
        
        Returns:
            MarkdownElement representing the blockquote
        """
        quote_lines = []
        
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos]
            if line.strip().startswith('>'):
                quote_lines.append(line.strip()[1:].strip())
                self.current_pos += 1
            elif line.strip() == '':
                self.current_pos += 1
                continue
            else:
                break
        
        content = '\n'.join(quote_lines)
        return MarkdownElement(type=ElementType.BLOCKQUOTE, content=content)
    
    def _parse_image(self, match) -> ImageElement:
        """Parse an image element.
        
        Args:
            match: Regex match object containing image syntax
            
        Returns:
            ImageElement with source, alt text, and optional title
        """
        self.current_pos += 1
        alt = match.group(1)
        src_and_title = match.group(2)
        
        # Separate URL and title
        parts = src_and_title.split(' "', 1)
        src = parts[0]
        title = parts[1][:-1] if len(parts) > 1 else None
        
        return ImageElement(src=src, alt=alt, title=title)
    
    def _parse_paragraph(self) -> MarkdownElement:
        """Parse a paragraph element.
        
        Returns:
            MarkdownElement representing a paragraph with inline elements parsed
        """
        content_lines = []
        
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos]
            
            # If empty line or special markers encountered, end paragraph
            if (line.strip() == '' or 
                line.strip().startswith('#') or
                line.strip().startswith('```') or
                re.match(r'^[\-\*\+]\s', line.strip()) or
                re.match(r'^\d+\.\s', line.strip())):
                break
            
            content_lines.append(line)
            self.current_pos += 1
        
        content = ' '.join(line.strip() for line in content_lines)
        
        # Parse inline elements (links, bold, italic, etc.)
        content = self._parse_inline_elements(content)
        
        return MarkdownElement(type=ElementType.PARAGRAPH, content=content)
    
    def _parse_inline_elements(self, text: str) -> str:
        """Parse inline elements (simplified processing).
        
        This can be extended to support more complex inline element parsing.
        Currently only handles basic link parsing.
        
        Args:
            text: Text content to process
            
        Returns:
            Text with inline elements processed
        """
        # Parse links [text](url)
        def replace_links(match):
            link_text = match.group(1)
            url = match.group(2)
            return f"[{link_text}]({url})"
        
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_links, text)
        
        return text


class MarkdownRenderer:
    """Renderer for converting structured documents to Markdown text.
    
    This class takes MarkdownDocument objects and renders them back to
    properly formatted Markdown text with configurable options.
    
    Attributes:
        options: Conversion options that control rendering behavior
    """
    
    def __init__(self, options: Optional[ConversionOptions] = None):
        """Initialize the Markdown renderer.
        
        Args:
            options: Optional conversion options to customize rendering
        """
        self.options = options or ConversionOptions()
    
    def render(self, document: MarkdownDocument) -> str:
        """Render a document object to Markdown text.
        
        Args:
            document: The MarkdownDocument to render
            
        Returns:
            Formatted Markdown text string
        """
        result = []
        
        # Render title
        if document.title and not self._has_h1_in_content(document.content):
            result.append(f"# {document.title}\n")
        
        # Render metadata
        if document.metadata and self.options.include_metadata:
            result.append("<!-- Metadata:")
            for key, value in document.metadata.items():
                result.append(f"{key}: {value}")
            result.append("-->\n")
        
        # Render content
        for element in document.content:
            rendered = self._render_element(element)
            if rendered:
                result.append(rendered)
        
        return '\n'.join(result)
    
    def _has_h1_in_content(self, elements: List[MarkdownElement]) -> bool:
        """Check if content already contains an h1 heading.
        
        Args:
            elements: List of elements to check
            
        Returns:
            True if an h1 heading is found, False otherwise
        """
        for element in elements:
            if (isinstance(element, HeadingElement) and element.level == 1):
                return True
        return False
    
    def _render_element(self, element: MarkdownElement) -> str:
        """Render a single element to Markdown.
        
        Args:
            element: The element to render
            
        Returns:
            Markdown text representation of the element
        """
        if element.type == ElementType.HEADING:
            level = element.attributes.get("level", 1)
            return f"{'#' * level} {element.content}\n"
        
        elif element.type == ElementType.PARAGRAPH:
            return f"{element.content}\n"
        
        elif element.type == ElementType.CODE_BLOCK:
            language = element.attributes.get("language", "")
            return f"```{language}\n{element.content}\n```\n"
        
        elif element.type == ElementType.LIST:
            return self._render_list(element)
        
        elif element.type == ElementType.BLOCKQUOTE:
            lines = element.content.split('\n')
            quoted_lines = [f"> {line}" for line in lines]
            return '\n'.join(quoted_lines) + '\n'
        
        elif element.type == ElementType.LINK:
            url = element.attributes.get("url", "")
            title = element.attributes.get("title", "")
            title_attr = f' "{title}"' if title else ""
            return f"[{element.content}]({url}{title_attr})"
        
        elif element.type == ElementType.IMAGE:
            src = element.attributes.get("src", "")
            alt = element.attributes.get("alt", "")
            title = element.attributes.get("title", "")
            title_attr = f' "{title}"' if title else ""
            return f"![{alt}]({src}{title_attr})\n"
        
        elif element.type == ElementType.HORIZONTAL_RULE:
            return "---\n"
        
        return ""
    
    def _render_list(self, list_element: ListElement) -> str:
        """Render a list element to Markdown.
        
        Args:
            list_element: The list element to render
            
        Returns:
            Markdown text representation of the list
        """
        if not list_element.children:
            return ""
        
        result = []
        list_type = list_element.attributes.get("list_type", "unordered")
        start = list_element.attributes.get("start", 1)
        
        for i, item in enumerate(list_element.children):
            if list_type == "ordered":
                prefix = f"{start + i}. "
            else:
                prefix = "- "
            result.append(f"{prefix}{item.content}")
        
        return '\n'.join(result) + '\n'


class MarkdownConverter:
    """Main converter class for Markdown and JSON transformations.
    
    This class provides the primary interface for converting between Markdown
    and JSON formats, coordinating the parser and renderer components.
    
    Attributes:
        parser: MarkdownParser instance for parsing operations
        renderer: MarkdownRenderer instance for rendering operations
    """
    
    def __init__(self):
        """Initialize the converter with parser and renderer instances."""
        self.parser = MarkdownParser()
        self.renderer = MarkdownRenderer()
    
    def convert(self, request: ConversionRequest) -> ConversionResponse:
        """Execute format conversion based on the request.
        
        Args:
            request: ConversionRequest specifying source/target formats and content
            
        Returns:
            ConversionResponse containing the result or error information
        """
        try:
            if request.source_format.lower() == "markdown" and request.target_format.lower() == "json":
                return self._markdown_to_json(request)
            elif request.source_format.lower() == "json" and request.target_format.lower() == "markdown":
                return self._json_to_markdown(request)
            else:
                return ConversionResponse(
                    success=False,
                    error=f"Unsupported conversion type: {request.source_format} -> {request.target_format}"
                )
        
        except Exception as e:
            return ConversionResponse(
                success=False,
                error=f"Error occurred during conversion: {str(e)}"
            )
    
    def _markdown_to_json(self, request: ConversionRequest) -> ConversionResponse:
        """Convert Markdown to JSON format.
        
        Args:
            request: ConversionRequest with Markdown content
            
        Returns:
            ConversionResponse with JSON result or error
        """
        if not isinstance(request.content, str):
            return ConversionResponse(
                success=False,
                error="Markdown content must be a string"
            )
        
        # Parse Markdown
        document = self.parser.parse(request.content)
        
        # Convert to JSON
        json_result = document.model_dump()
        
        return ConversionResponse(
            success=True,
            result=json_result,
            metadata={
                "elements_count": len(document.content),
                "has_metadata": document.metadata is not None,
                "has_title": document.title is not None
            }
        )
    
    def _json_to_markdown(self, request: ConversionRequest) -> ConversionResponse:
        """Convert JSON to Markdown format.
        
        Args:
            request: ConversionRequest with JSON content
            
        Returns:
            ConversionResponse with Markdown result or error
        """
        if isinstance(request.content, str):
            try:
                json_data = json.loads(request.content)
            except json.JSONDecodeError as e:
                return ConversionResponse(
                    success=False,
                    error=f"JSON parsing error: {str(e)}"
                )
        elif isinstance(request.content, dict):
            json_data = request.content
        else:
            return ConversionResponse(
                success=False,
                error="JSON content must be a string or dictionary"
            )
        
        # Create document object from JSON
        try:
            document = MarkdownDocument(**json_data)
        except Exception as e:
            return ConversionResponse(
                success=False,
                error=f"JSON data format error: {str(e)}"
            )
        
        # Set rendering options
        if request.options:
            self.renderer = MarkdownRenderer(request.options)
        
        # Render to Markdown
        markdown_result = self.renderer.render(document)
        
        return ConversionResponse(
            success=True,
            result=markdown_result,
            metadata={
                "elements_count": len(document.content),
                "has_metadata": document.metadata is not None,
                "has_title": document.title is not None
            }
        )


# Convenience functions
def markdown_to_json(markdown_text: str, options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    """Convert Markdown text to JSON (convenience function).
    
    Args:
        markdown_text: The Markdown text to convert
        options: Optional conversion options
        
    Returns:
        Dictionary containing the JSON representation
        
    Raises:
        ValueError: If conversion fails
    """
    converter = MarkdownConverter()
    request = ConversionRequest(
        source_format="markdown",
        target_format="json",
        content=markdown_text,
        options=options
    )
    response = converter.convert(request)
    
    if response.success:
        return response.result
    else:
        raise ValueError(f"Conversion failed: {response.error}")


def json_to_markdown(json_data: Union[str, Dict[str, Any]], options: Optional[ConversionOptions] = None) -> str:
    """Convert JSON to Markdown text (convenience function).
    
    Args:
        json_data: JSON data as string or dictionary
        options: Optional conversion options
        
    Returns:
        Markdown text string
        
    Raises:
        ValueError: If conversion fails
    """
    converter = MarkdownConverter()
    request = ConversionRequest(
        source_format="json",
        target_format="markdown",
        content=json_data,
        options=options
    )
    response = converter.convert(request)
    
    if response.success:
        return response.result
    else:
        raise ValueError(f"Conversion failed: {response.error}")
