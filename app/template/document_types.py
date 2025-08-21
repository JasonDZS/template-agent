"""Markdown Document Data Types for Template Agent.

This module defines the core data types for markdown document representation:
- Base node types and enums
- Markdown document tree structure
- Specialized node types (heading, paragraph, list, etc.)
- Document analysis and navigation utilities
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class NodeType(str, Enum):
    """Document node types."""
    ROOT = "root"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    TABLE = "table"
    IMAGE = "image"
    LINK = "link"
    TEXT = "text"


class MarkdownNode(BaseModel):
    """Base node for markdown document tree structure.
    
    This is the foundational class for all document nodes that can
    form hierarchical tree structures with parent-child relationships.
    
    Attributes:
        id: Unique identifier for the node
        type: Type of the node (heading, paragraph, etc.)
        content: Text content of the node
        attributes: Additional attributes specific to node type
        children: Child nodes for hierarchical structure
        parent_id: ID of parent node (if any)
        depth: Depth level in the tree (0 for root)
        path: Path from root to current node
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique node identifier")
    type: NodeType = Field(..., description="Node type")
    content: Optional[str] = Field(None, description="Text content")
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Node attributes")
    children: List['MarkdownNode'] = Field(default_factory=list, description="Child nodes")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    depth: int = Field(0, description="Node depth in tree")
    path: List[str] = Field(default_factory=list, description="Path from root to current node")
    order: int = Field(0, description="Order among siblings (0-based index)")
    
    class Config:
        from_attributes = True
    
    def add_child(self, child: 'MarkdownNode') -> None:
        """Add a child node to this node.
        
        Args:
            child: Child node to add
        """
        child.parent_id = self.id
        child.depth = self.depth + 1
        child.path = self.path + [self.id]
        child.order = len(self.children)  # Set order based on current children count
        self.children.append(child)
    
    def remove_child(self, child_id: str) -> bool:
        """Remove a child node by ID.
        
        Args:
            child_id: ID of child to remove
            
        Returns:
            True if child was removed, False if not found
        """
        for i, child in enumerate(self.children):
            if child.id == child_id:
                del self.children[i]
                return True
        return False
    
    def find_child(self, child_id: str) -> Optional['MarkdownNode']:
        """Find child node by ID (recursive search).
        
        Args:
            child_id: ID of child to find
            
        Returns:
            Found child node or None
        """
        for child in self.children:
            if child.id == child_id:
                return child
            # Recursive search in grandchildren
            result = child.find_child(child_id)
            if result:
                return result
        return None
    
    def get_descendants(self) -> List['MarkdownNode']:
        """Get all descendant nodes (recursive).
        
        Returns:
            List of all descendant nodes
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def get_siblings(self, parent_node: Optional['MarkdownNode'] = None) -> List['MarkdownNode']:
        """Get sibling nodes.
        
        Args:
            parent_node: Parent node (if known)
            
        Returns:
            List of sibling nodes (excluding self)
        """
        if not parent_node:
            return []
        return [child for child in parent_node.children if child.id != self.id]
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (no children).
        
        Returns:
            True if leaf node, False otherwise
        """
        return len(self.children) == 0
    
    def is_heading_leaf(self) -> bool:
        """Check if this is a heading node with no heading children.
        
        Returns:
            True if heading with no heading children, False otherwise
        """
        if self.type != NodeType.HEADING:
            return False
        
        # Check if any child is a heading
        for child in self.children:
            if child.type == NodeType.HEADING:
                return False
        return True
    
    def children_content(self) -> str:
        """Get all children content converted to markdown format.
        
        This method reconstructs the original markdown content from all
        children nodes, maintaining the original structure and formatting.
        
        Returns:
            Markdown string of all children content
        """
        if not self.children:
            return ""
        
        return _nodes_to_markdown(self.children)
    
    def get_section_markdown(self) -> str:
        """Get the complete section content as markdown including the heading.
        
        For heading nodes, this includes the heading itself plus all children content.
        For other nodes, this returns the node content plus children content.
        
        Returns:
            Complete markdown content for this section
        """
        markdown_parts = []
        
        # Add the current node content
        if self.type == NodeType.HEADING:
            level_prefix = "#" * getattr(self, 'level', 1)
            markdown_parts.append(f"{level_prefix} {self.content or ''}")
        elif self.content:
            markdown_parts.append(self.content)
        
        # Add children content
        children_md = self.children_content()
        if children_md:
            markdown_parts.append(children_md)
        
        return "\n\n".join(part for part in markdown_parts if part.strip())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation.
        
        Returns:
            Dictionary representation of the node
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children],
            "parent_id": self.parent_id,
            "depth": self.depth,
            "path": self.path,
            "order": self.order
        }


class HeadingNode(MarkdownNode):
    """Heading node with level information.
    
    Attributes:
        level: Heading level (1-6)
    """
    type: NodeType = Field(NodeType.HEADING, description="Node type")
    level: int = Field(..., ge=1, le=6, description="Heading level (1-6)")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        self.attributes["level"] = self.level
    
    def get_child_headings(self) -> List['HeadingNode']:
        """Get direct child headings.
        
        Returns:
            List of direct child heading nodes
        """
        return [child for child in self.children if isinstance(child, HeadingNode)]
    
    def get_all_child_headings(self) -> List['HeadingNode']:
        """Get all descendant heading nodes.
        
        Returns:
            List of all descendant heading nodes
        """
        headings = []
        for child in self.children:
            if isinstance(child, HeadingNode):
                headings.append(child)
                headings.extend(child.get_all_child_headings())
        return headings
    
    def get_section_content(self) -> List[MarkdownNode]:
        """Get all content nodes in this section (excluding child headings).
        
        Returns:
            List of content nodes in this section
        """
        section_content = []
        for child in self.children:
            if child.type != NodeType.HEADING:
                section_content.append(child)
        return section_content
    
    def get_section_content_markdown(self) -> str:
        """Get section content (excluding child headings) as markdown.
        
        This returns only the content directly under this heading,
        without including any child headings or their content.
        
        Returns:
            Markdown string of section content only
        """
        section_content = self.get_section_content()
        if not section_content:
            return ""
        
        return _nodes_to_markdown(section_content)
    
    def get_full_section_markdown(self) -> str:
        """Get the complete section with heading and all content as markdown.
        
        This includes the heading itself plus ALL content (including child headings).
        This is useful for getting the complete section content as it appears 
        in the original markdown document.
        
        Returns:
            Complete markdown content including heading and all children
        """
        return self.get_section_markdown()
    
    def get_content_for_agent(self) -> str:
        """Get content suitable for passing to section agents.
        
        This method returns the children content in a format optimized
        for section agent processing, maintaining readability while
        preserving the essential structure and information.
        
        Returns:
            Clean content string for agent processing
        """
        children_md = self.children_content()
        if not children_md:
            return ""
        
        # Clean up formatting issues for better agent processing
        content = children_md
        
        # Fix common table formatting issues
        if '|' in content:
            # Ensure tables have proper spacing
            content = content.replace(' | ', ' | ')
            # Try to ensure table rows are on separate lines
            content = content.replace(' | |', ' |\n|')
        
        # Ensure proper paragraph separation
        content = content.replace('\n\n\n', '\n\n')
        
        return content.strip()


class ParagraphNode(MarkdownNode):
    """Paragraph text node."""
    type: NodeType = Field(NodeType.PARAGRAPH, description="Node type")


class ListNode(MarkdownNode):
    """List container node.
    
    Attributes:
        ordered: Whether this is an ordered list
        start: Starting number for ordered lists
    """
    type: NodeType = Field(NodeType.LIST, description="Node type")
    ordered: bool = Field(False, description="Whether this is an ordered list")
    start: Optional[int] = Field(None, description="Starting number for ordered lists")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        self.attributes["ordered"] = self.ordered
        if self.start is not None:
            self.attributes["start"] = self.start


class ListItemNode(MarkdownNode):
    """List item node."""
    type: NodeType = Field(NodeType.LIST_ITEM, description="Node type")


class CodeBlockNode(MarkdownNode):
    """Code block node.
    
    Attributes:
        language: Programming language for syntax highlighting
    """
    type: NodeType = Field(NodeType.CODE_BLOCK, description="Node type")
    language: Optional[str] = Field(None, description="Programming language")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        if self.language:
            self.attributes["language"] = self.language


class MarkdownDocument(BaseModel):
    """Complete markdown document with tree structure.
    
    This class represents a complete markdown document as a tree structure
    with a root node and hierarchical organization of content.
    
    Attributes:
        title: Document title
        metadata: Document metadata
        root: Root node of the document tree
        node_index: Index of all nodes by ID for fast lookup
    """
    title: Optional[str] = Field(None, description="Document title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    root: MarkdownNode = Field(default_factory=lambda: MarkdownNode(type=NodeType.ROOT), description="Root node")
    node_index: Dict[str, MarkdownNode] = Field(default_factory=dict, description="Node index by ID")
    
    class Config:
        from_attributes = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._build_index()
    
    def _build_index(self) -> None:
        """Build node index for fast lookup."""
        self.node_index = {}
        self._index_node(self.root)
    
    def _index_node(self, node: MarkdownNode) -> None:
        """Recursively index a node and its children.
        
        Args:
            node: Node to index
        """
        self.node_index[node.id] = node
        for child in node.children:
            self._index_node(child)
    
    def add_node(self, parent_id: str, node: MarkdownNode) -> bool:
        """Add a node to the document tree.
        
        Args:
            parent_id: ID of parent node
            node: Node to add
            
        Returns:
            True if node was added, False if parent not found
        """
        parent = self.node_index.get(parent_id)
        if parent:
            parent.add_child(node)
            self._index_node(node)
            return True
        return False
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the document tree.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            True if node was removed, False if not found
        """
        node = self.node_index.get(node_id)
        if not node or node.parent_id is None:
            return False
        
        parent = self.node_index.get(node.parent_id)
        if parent and parent.remove_child(node_id):
            # Remove from index (including descendants)
            self._remove_from_index(node)
            return True
        return False
    
    def _remove_from_index(self, node: MarkdownNode) -> None:
        """Remove node and its descendants from index.
        
        Args:
            node: Node to remove from index
        """
        if node.id in self.node_index:
            del self.node_index[node.id]
        for child in node.children:
            self._remove_from_index(child)
    
    def find_node(self, node_id: str) -> Optional[MarkdownNode]:
        """Find node by ID.
        
        Args:
            node_id: ID of node to find
            
        Returns:
            Found node or None
        """
        return self.node_index.get(node_id)
    
    def get_all_headings(self) -> List[HeadingNode]:
        """Get all heading nodes in the document.
        
        Returns:
            List of all heading nodes
        """
        headings = []
        for node in self.node_index.values():
            if isinstance(node, HeadingNode):
                headings.append(node)
        return sorted(headings, key=lambda h: (h.depth, h.path))
    
    def get_leaf_headings(self) -> List[HeadingNode]:
        """Get all leaf heading nodes (headings with no child headings).
        
        Returns:
            List of leaf heading nodes
        """
        return [h for h in self.get_all_headings() if h.is_heading_leaf()]
    
    def get_top_level_headings(self) -> List[HeadingNode]:
        """Get top-level heading nodes (direct children of root).
        
        Returns:
            List of top-level heading nodes
        """
        return [child for child in self.root.children if isinstance(child, HeadingNode)]
    
    def get_headings_by_level(self, level: int) -> List[HeadingNode]:
        """Get all headings of a specific level.
        
        Args:
            level: Heading level to filter by (1-6)
            
        Returns:
            List of headings at the specified level
        """
        return [h for h in self.get_all_headings() if h.level == level]
    
    def get_document_outline(self) -> List[Dict[str, Any]]:
        """Get document outline as a list of heading information.
        
        Returns:
            List of heading information dictionaries
        """
        outline = []
        for heading in self.get_all_headings():
            outline.append({
                "id": heading.id,
                "level": heading.level,
                "content": heading.content,
                "depth": heading.depth,
                "path": heading.path,
                "has_children": not heading.is_heading_leaf(),
                "child_count": len(heading.get_child_headings())
            })
        return outline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation.
        
        Returns:
            Dictionary representation of the document
        """
        return {
            "title": self.title,
            "metadata": self.metadata,
            "content": self.root.to_dict(),
            "outline": self.get_document_outline()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarkdownDocument':
        """Create document from dictionary representation.
        
        Args:
            data: Dictionary containing document data
            
        Returns:
            New MarkdownDocument instance
        """
        doc = cls(
            title=data.get("title"),
            metadata=data.get("metadata", {})
        )
        
        # Reconstruct tree structure
        if "content" in data:
            doc.root = cls._dict_to_node(data["content"])
            doc._build_index()
        
        return doc
    
    @staticmethod
    def _dict_to_node(data: Dict[str, Any]) -> MarkdownNode:
        """Convert dictionary to node (recursive).
        
        Args:
            data: Dictionary containing node data
            
        Returns:
            Reconstructed node
        """
        node_type = NodeType(data["type"])
        
        # Create appropriate node type
        if node_type == NodeType.HEADING:
            level = data.get("attributes", {}).get("level", 1)
            node = HeadingNode(
                id=data["id"],
                content=data.get("content"),
                attributes=data.get("attributes", {}),
                level=level,
                parent_id=data.get("parent_id"),
                depth=data.get("depth", 0),
                path=data.get("path", []),
                order=data.get("order", 0)
            )
        elif node_type == NodeType.PARAGRAPH:
            node = ParagraphNode(
                id=data["id"],
                content=data.get("content"),
                attributes=data.get("attributes", {}),
                parent_id=data.get("parent_id"),
                depth=data.get("depth", 0),
                path=data.get("path", []),
                order=data.get("order", 0)
            )
        elif node_type == NodeType.LIST:
            ordered = data.get("attributes", {}).get("ordered", False)
            start = data.get("attributes", {}).get("start")
            node = ListNode(
                id=data["id"],
                content=data.get("content"),
                attributes=data.get("attributes", {}),
                ordered=ordered,
                start=start,
                parent_id=data.get("parent_id"),
                depth=data.get("depth", 0),
                path=data.get("path", []),
                order=data.get("order", 0)
            )
        elif node_type == NodeType.CODE_BLOCK:
            language = data.get("attributes", {}).get("language")
            node = CodeBlockNode(
                id=data["id"],
                content=data.get("content"),
                attributes=data.get("attributes", {}),
                language=language,
                parent_id=data.get("parent_id"),
                depth=data.get("depth", 0),
                path=data.get("path", []),
                order=data.get("order", 0)
            )
        else:
            # Generic node
            node = MarkdownNode(
                id=data["id"],
                type=node_type,
                content=data.get("content"),
                attributes=data.get("attributes", {}),
                parent_id=data.get("parent_id"),
                depth=data.get("depth", 0),
                path=data.get("path", []),
                order=data.get("order", 0)
            )
        
        # Add children recursively
        for child_data in data.get("children", []):
            child_node = MarkdownDocument._dict_to_node(child_data)
            node.children.append(child_node)
        
        return node


def _nodes_to_markdown(nodes: List[MarkdownNode]) -> str:
    """Convert a list of nodes to markdown format.
    
    This is a helper function that handles the conversion of various node types
    back to their original markdown representation.
    
    Args:
        nodes: List of nodes to convert
        
    Returns:
        Markdown string representation
    """
    markdown_parts = []
    
    for node in nodes:
        if node.type == NodeType.HEADING:
            # Handle heading nodes
            level = getattr(node, 'level', 1)
            level_prefix = "#" * level
            heading_line = f"{level_prefix} {node.content or ''}"
            markdown_parts.append(heading_line)
            
            # Add children content recursively
            if node.children:
                children_md = _nodes_to_markdown(node.children)
                if children_md:
                    markdown_parts.append(children_md)
                    
        elif node.type == NodeType.PARAGRAPH:
            # Handle paragraph nodes
            if node.content:
                # Check if this might be a table (contains multiple | characters)
                content = node.content
                if '|' in content and content.count('|') > 6:  # Likely a table
                    # Simple approach: ensure proper spacing around pipes
                    content = content.replace('|', ' | ').replace('  |  ', ' | ')
                    # Add line breaks for better readability if it's a long single line
                    if len(content) > 200 and '\n' not in content:
                        # Try to detect table row boundaries (this is a simple heuristic)
                        parts = content.split(' | ')
                        if len(parts) > 8:  # Likely multiple rows concatenated
                            rows = []
                            current_row = []
                            for i, part in enumerate(parts):
                                current_row.append(part)
                                # Every 4th element likely ends a row (based on template structure)
                                if (i + 1) % 4 == 0:
                                    rows.append(' | '.join(current_row))
                                    current_row = []
                            if current_row:
                                rows.append(' | '.join(current_row))
                            content = '\n'.join(rows)
                
                markdown_parts.append(content)
            
            # Add children content if any (for nested structures)
            if node.children:
                children_md = _nodes_to_markdown(node.children)
                if children_md:
                    markdown_parts.append(children_md)
                    
        elif node.type == NodeType.LIST:
            # Handle list nodes
            list_items = []
            ordered = getattr(node, 'ordered', False)
            start_num = getattr(node, 'start', 1) if ordered else None
            
            for i, child in enumerate(node.children):
                if child.type == NodeType.LIST_ITEM:
                    if ordered:
                        item_prefix = f"{start_num + i}. "
                    else:
                        item_prefix = "- "
                    
                    item_content = child.content or ""
                    # Handle nested content in list items
                    if child.children:
                        nested_content = _nodes_to_markdown(child.children)
                        if nested_content:
                            # Indent nested content
                            indented_nested = "\n".join(["  " + line for line in nested_content.split("\n")])
                            item_content = f"{item_content}\n{indented_nested}" if item_content else indented_nested
                    
                    list_items.append(f"{item_prefix}{item_content}")
            
            if list_items:
                markdown_parts.append("\n".join(list_items))
                
        elif node.type == NodeType.CODE_BLOCK:
            # Handle code block nodes
            language = getattr(node, 'language', '') or node.attributes.get('language', '') if node.attributes else ''
            content = node.content or ""
            
            code_block = f"```{language}\n{content}\n```"
            markdown_parts.append(code_block)
            
        elif node.type == NodeType.BLOCKQUOTE:
            # Handle blockquote nodes
            if node.content:
                # Prefix each line with "> "
                quoted_lines = [f"> {line}" for line in node.content.split("\n")]
                markdown_parts.append("\n".join(quoted_lines))
            
            # Handle nested content in blockquotes
            if node.children:
                children_md = _nodes_to_markdown(node.children)
                if children_md:
                    # Prefix children content with "> "
                    quoted_children = [f"> {line}" for line in children_md.split("\n")]
                    markdown_parts.append("\n".join(quoted_children))
                    
        elif node.type == NodeType.TABLE:
            # Handle table nodes - preserve original table formatting
            if node.content:
                # Ensure proper line breaks are preserved in table content
                table_content = node.content.replace('|', ' |').replace(' |', ' |\n').replace('|\n |', '|\n|')
                markdown_parts.append(table_content)
            
        elif node.type in [NodeType.LINK, NodeType.IMAGE]:
            # Handle links and images
            if node.content:
                markdown_parts.append(node.content)
                
        else:
            # Handle other node types (text, etc.)
            if node.content:
                markdown_parts.append(node.content)
            
            # Add children content for any other nested structures
            if node.children:
                children_md = _nodes_to_markdown(node.children)
                if children_md:
                    markdown_parts.append(children_md)
    
    return "\n\n".join(part for part in markdown_parts if part.strip())


# Update forward references
MarkdownNode.model_rebuild()
HeadingNode.model_rebuild()
MarkdownDocument.model_rebuild()