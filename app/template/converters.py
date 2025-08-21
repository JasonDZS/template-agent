"""Conversion utilities for Markdown documents.

This module provides utilities for converting between different formats:
- Markdown text to document tree
- JSON to document tree
- Document tree back to JSON
- Batch processing utilities
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from .document_types import (
    MarkdownDocument, MarkdownNode, HeadingNode, ParagraphNode, 
    ListNode, ListItemNode, CodeBlockNode, NodeType
)


def parse_markdown_to_document_tree(markdown_content: str, title: Optional[str] = None) -> MarkdownDocument:
    """Parse markdown content directly into MarkdownDocument tree.
    
    This function provides a direct way to convert markdown content into
    a document tree without going through the JSON conversion step.
    
    Args:
        markdown_content: Raw markdown content string
        title: Optional document title (if not provided, will be extracted from content)
        
    Returns:
        MarkdownDocument with tree structure
        
    Raises:
        Exception: If markdown parsing fails
    """
    from ..converter import MarkdownConverter, ConversionRequest
    
    # Use the existing converter to get JSON structure
    converter = MarkdownConverter()
    conversion_request = ConversionRequest(
        source_format="markdown",
        target_format="json",
        content=markdown_content,
        options=None
    )
    
    result = converter.convert(conversion_request)
    if not result.success:
        raise Exception(f"Markdown parsing failed: {result.error}")
    
    # Override title if provided
    json_data = result.result
    if title is not None:
        json_data["title"] = title
    
    # Convert to document tree
    return convert_json_to_document_tree(json_data)


def parse_markdown_file_to_document_tree(file_path: str, title: Optional[str] = None) -> MarkdownDocument:
    """Parse markdown file directly into MarkdownDocument tree.
    
    Convenience function to load and parse a markdown file into a document tree.
    
    Args:
        file_path: Path to markdown file
        title: Optional document title (if not provided, will be extracted from content)
        
    Returns:
        MarkdownDocument with tree structure
        
    Raises:
        FileNotFoundError: If file does not exist
        Exception: If file reading or parsing fails
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        # Read file content
        markdown_content = file_path_obj.read_text(encoding='utf-8')
        
        # Use filename as title if not provided
        if title is None:
            title = file_path_obj.stem
        
        return parse_markdown_to_document_tree(markdown_content, title)
        
    except Exception as e:
        raise Exception(f"Failed to parse markdown file {file_path}: {str(e)}")


def parse_markdown_with_metadata(markdown_content: str, metadata: Optional[Dict[str, Any]] = None) -> MarkdownDocument:
    """Parse markdown content into document tree with custom metadata.
    
    Args:
        markdown_content: Raw markdown content string
        metadata: Optional custom metadata dictionary
        
    Returns:
        MarkdownDocument with tree structure and metadata
    """
    doc = parse_markdown_to_document_tree(markdown_content)
    
    if metadata:
        doc.metadata.update(metadata)
    
    return doc


def extract_document_info(doc: MarkdownDocument) -> Dict[str, Any]:
    """Extract comprehensive document information from document tree.
    
    Args:
        doc: MarkdownDocument to analyze
        
    Returns:
        Dictionary containing document statistics and structure info
    """
    all_headings = doc.get_all_headings()
    leaf_headings = doc.get_leaf_headings()
    
    # Calculate heading level distribution
    level_distribution = {}
    for h in all_headings:
        level_distribution[h.level] = level_distribution.get(h.level, 0) + 1
    
    # Calculate depth statistics
    max_depth = max([h.depth for h in all_headings], default=0)
    depth_distribution = {}
    for h in all_headings:
        depth_distribution[h.depth] = depth_distribution.get(h.depth, 0) + 1
    
    # Get all node types
    all_nodes = list(doc.node_index.values())
    node_type_distribution = {}
    for node in all_nodes:
        node_type = node.type.value
        node_type_distribution[node_type] = node_type_distribution.get(node_type, 0) + 1
    
    return {
        "title": doc.title,
        "metadata": doc.metadata,
        "total_nodes": len(all_nodes),
        "total_headings": len(all_headings),
        "leaf_headings": len(leaf_headings),
        "max_heading_level": max([h.level for h in all_headings], default=0),
        "max_tree_depth": max_depth,
        "heading_level_distribution": level_distribution,
        "depth_distribution": depth_distribution,
        "node_type_distribution": node_type_distribution,
        "outline": doc.get_document_outline()
    }


def batch_parse_markdown_files(file_paths: List[str]) -> List[MarkdownDocument]:
    """Parse multiple markdown files into document trees.
    
    Args:
        file_paths: List of markdown file paths to parse
        
    Returns:
        List of MarkdownDocument instances
        
    Raises:
        Exception: If any file fails to parse
    """
    documents = []
    failed_files = []
    
    for file_path in file_paths:
        try:
            doc = parse_markdown_file_to_document_tree(file_path)
            documents.append(doc)
        except Exception as e:
            failed_files.append((file_path, str(e)))
    
    if failed_files:
        error_msg = "Failed to parse files:\n" + "\n".join([f"  - {path}: {error}" for path, error in failed_files])
        raise Exception(error_msg)
    
    return documents


def convert_json_to_document_tree(json_data: Dict[str, Any]) -> MarkdownDocument:
    """Convert JSON structure from converter to MarkdownDocument tree.
    
    Args:
        json_data: JSON data from markdown converter
        
    Returns:
        MarkdownDocument with tree structure
    """
    metadata = json_data.get("metadata")
    if metadata is None:
        metadata = {}
    
    doc = MarkdownDocument(
        title=json_data.get("title"),
        metadata=metadata
    )
    
    # Convert content elements to tree nodes
    content_list = json_data.get("content", [])
    _convert_content_to_tree(content_list, doc.root, doc)
    
    return doc


def convert_document_tree_to_json(doc: MarkdownDocument) -> Dict[str, Any]:
    """Convert MarkdownDocument tree back to JSON format.
    
    Args:
        doc: MarkdownDocument to convert
        
    Returns:
        JSON dictionary compatible with converter format
    """
    return {
        "title": doc.title,
        "metadata": doc.metadata,
        "content": [_convert_node_to_element(child) for child in doc.root.children]
    }


def _convert_content_to_tree(content_list: List[Dict[str, Any]], parent: MarkdownNode, doc: MarkdownDocument) -> None:
    """Convert content list to tree nodes recursively.
    
    Args:
        content_list: List of content elements
        parent: Parent node to attach to
        doc: Document instance for indexing
    """
    for element in content_list:
        node = _create_node_from_element(element)
        parent.add_child(node)
        doc.node_index[node.id] = node
        
        # Process children_content recursively
        attributes = element.get("attributes") or {}
        if attributes and "children_content" in attributes and attributes["children_content"]:
            _convert_content_to_tree(attributes["children_content"], node, doc)


def _create_node_from_element(element: Dict[str, Any]) -> MarkdownNode:
    """Create appropriate node type from element.
    
    Args:
        element: Element dictionary
        
    Returns:
        Created node
    """
    element_type = element.get("type", "text")
    content = element.get("content", "")
    attributes = element.get("attributes", {})
    
    if element_type == "heading":
        level = attributes.get("level", 1)
        return HeadingNode(
            content=content,
            attributes=attributes,
            level=level
        )
    elif element_type == "paragraph":
        return ParagraphNode(
            content=content,
            attributes=attributes
        )
    elif element_type == "list":
        ordered = attributes.get("list_type") == "ordered"
        start = attributes.get("start")
        return ListNode(
            content=content,
            attributes=attributes,
            ordered=ordered,
            start=start
        )
    elif element_type == "list_item":
        return ListItemNode(
            content=content,
            attributes=attributes
        )
    elif element_type == "code_block":
        language = attributes.get("language")
        return CodeBlockNode(
            content=content,
            attributes=attributes,
            language=language
        )
    else:
        # Map other types to generic nodes
        node_type_map = {
            "blockquote": NodeType.BLOCKQUOTE,
            "table": NodeType.TABLE,
            "image": NodeType.IMAGE,
            "link": NodeType.LINK,
            "text": NodeType.TEXT
        }
        node_type = node_type_map.get(element_type, NodeType.TEXT)
        
        return MarkdownNode(
            type=node_type,
            content=content,
            attributes=attributes
        )


def _convert_node_to_element(node: MarkdownNode) -> Dict[str, Any]:
    """Convert node back to element dictionary.
    
    Args:
        node: Node to convert
        
    Returns:
        Element dictionary
    """
    # Map node types back to element types
    type_map = {
        NodeType.HEADING: "heading",
        NodeType.PARAGRAPH: "paragraph",
        NodeType.LIST: "list",
        NodeType.LIST_ITEM: "list_item",
        NodeType.CODE_BLOCK: "code_block",
        NodeType.BLOCKQUOTE: "blockquote",
        NodeType.TABLE: "table",
        NodeType.IMAGE: "image",
        NodeType.LINK: "link",
        NodeType.TEXT: "text"
    }
    
    element = {
        "type": type_map.get(node.type, "text"),
        "content": node.content,
        "attributes": dict(node.attributes) if node.attributes else {}
    }
    
    # Add children_content if there are children
    if node.children:
        element["attributes"]["children_content"] = [
            _convert_node_to_element(child) for child in node.children
        ]
    
    return element