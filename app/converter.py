#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown和JSON互转核心模块
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
    """Markdown解析器"""
    
    def __init__(self):
        self.current_pos = 0
        self.lines = []
    
    def parse(self, markdown_text: str) -> MarkdownDocument:
        """解析Markdown文本为文档对象"""
        self.lines = markdown_text.split('\n')
        self.current_pos = 0
        
        # 提取元数据
        metadata = self._extract_metadata()
        
        # 解析内容
        elements = []
        while self.current_pos < len(self.lines):
            element = self._parse_next_element()
            if element:
                elements.append(element)
        
        # 获取标题（如果第一个元素是一级标题）
        title = None
        if elements and isinstance(elements[0], HeadingElement) and elements[0].level == 1:
            title = elements[0].content
        
        return MarkdownDocument(
            title=title,
            metadata=metadata,
            content=elements
        )
    
    def _extract_metadata(self) -> Optional[Dict[str, Any]]:
        """提取HTML注释中的元数据"""
        metadata = {}
        
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos].strip()
            
            # 检查是否是元数据注释
            if line.startswith('<!-- Metadata:'):
                self.current_pos += 1
                
                # 解析元数据内容
                while self.current_pos < len(self.lines):
                    line = self.lines[self.current_pos].strip()
                    if line == '-->':
                        self.current_pos += 1
                        break
                    
                    # 解析键值对
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                    
                    self.current_pos += 1
                break
            elif line:
                # 遇到非空行且不是元数据，停止搜索
                break
            
            self.current_pos += 1
        
        return metadata if metadata else None
    
    def _parse_next_element(self) -> Optional[MarkdownElement]:
        """解析下一个元素"""
        if self.current_pos >= len(self.lines):
            return None
        
        line = self.lines[self.current_pos].strip()
        
        # 跳过空行
        if not line:
            self.current_pos += 1
            return self._parse_next_element()
        
        # 解析标题
        if line.startswith('#'):
            return self._parse_heading()
        
        # 解析代码块
        if line.startswith('```'):
            return self._parse_code_block()
        
        # 解析列表
        if re.match(r'^[\-\*\+]\s', line) or re.match(r'^\d+\.\s', line):
            return self._parse_list()
        
        # 解析引用
        if line.startswith('>'):
            return self._parse_blockquote()
        
        # 解析水平分割线
        if re.match(r'^[\-\*_]{3,}$', line):
            self.current_pos += 1
            return MarkdownElement(type=ElementType.HORIZONTAL_RULE)
        
        # 解析图片
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if img_match:
            return self._parse_image(img_match)
        
        # 默认作为段落处理
        return self._parse_paragraph()
    
    def _parse_heading(self) -> HeadingElement:
        """解析标题"""
        line = self.lines[self.current_pos]
        self.current_pos += 1
        
        # 计算标题级别
        level = 0
        for char in line:
            if char == '#':
                level += 1
            else:
                break
        
        content = line[level:].strip()
        return HeadingElement(level=level, content=content)
    
    def _parse_code_block(self) -> CodeBlockElement:
        """解析代码块"""
        start_line = self.lines[self.current_pos]
        self.current_pos += 1
        
        # 提取语言
        language = start_line[3:].strip() if len(start_line) > 3 else None
        
        # 收集代码内容
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
        """解析列表"""
        items = []
        list_type = None
        start_num = 1
        
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos].strip()
            
            # 检查是否是列表项
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
                # 空行，继续检查下一行
                self.current_pos += 1
                continue
            else:
                # 不是列表项，结束列表解析
                break
        
        return ListElement(
            list_type=list_type or ListType.UNORDERED,
            start=start_num if list_type == ListType.ORDERED else None,
            children=items
        )
    
    def _parse_blockquote(self) -> MarkdownElement:
        """解析引用"""
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
        """解析图片"""
        self.current_pos += 1
        alt = match.group(1)
        src_and_title = match.group(2)
        
        # 分离URL和标题
        parts = src_and_title.split(' "', 1)
        src = parts[0]
        title = parts[1][:-1] if len(parts) > 1 else None
        
        return ImageElement(src=src, alt=alt, title=title)
    
    def _parse_paragraph(self) -> MarkdownElement:
        """解析段落"""
        content_lines = []
        
        while self.current_pos < len(self.lines):
            line = self.lines[self.current_pos]
            
            # 如果遇到空行或特殊标记，结束段落
            if (line.strip() == '' or 
                line.strip().startswith('#') or
                line.strip().startswith('```') or
                re.match(r'^[\-\*\+]\s', line.strip()) or
                re.match(r'^\d+\.\s', line.strip())):
                break
            
            content_lines.append(line)
            self.current_pos += 1
        
        content = ' '.join(line.strip() for line in content_lines)
        
        # 解析内联元素（链接、粗体、斜体等）
        content = self._parse_inline_elements(content)
        
        return MarkdownElement(type=ElementType.PARAGRAPH, content=content)
    
    def _parse_inline_elements(self, text: str) -> str:
        """解析内联元素（简化处理）"""
        # 这里可以扩展以支持更复杂的内联元素解析
        # 当前只做基础的链接解析
        
        # 解析链接 [text](url)
        def replace_links(match):
            link_text = match.group(1)
            url = match.group(2)
            return f"[{link_text}]({url})"
        
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_links, text)
        
        return text


class MarkdownRenderer:
    """Markdown渲染器"""
    
    def __init__(self, options: Optional[ConversionOptions] = None):
        self.options = options or ConversionOptions()
    
    def render(self, document: MarkdownDocument) -> str:
        """渲染文档为Markdown文本"""
        result = []
        
        # 渲染标题
        if document.title and not self._has_h1_in_content(document.content):
            result.append(f"# {document.title}\n")
        
        # 渲染元数据
        if document.metadata and self.options.include_metadata:
            result.append("<!-- Metadata:")
            for key, value in document.metadata.items():
                result.append(f"{key}: {value}")
            result.append("-->\n")
        
        # 渲染内容
        for element in document.content:
            rendered = self._render_element(element)
            if rendered:
                result.append(rendered)
        
        return '\n'.join(result)
    
    def _has_h1_in_content(self, elements: List[MarkdownElement]) -> bool:
        """检查内容中是否已有一级标题"""
        for element in elements:
            if (isinstance(element, HeadingElement) and element.level == 1):
                return True
        return False
    
    def _render_element(self, element: MarkdownElement) -> str:
        """渲染单个元素"""
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
        """渲染列表"""
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
    """Markdown转换器主类"""
    
    def __init__(self):
        self.parser = MarkdownParser()
        self.renderer = MarkdownRenderer()
    
    def convert(self, request: ConversionRequest) -> ConversionResponse:
        """执行转换"""
        try:
            if request.source_format.lower() == "markdown" and request.target_format.lower() == "json":
                return self._markdown_to_json(request)
            elif request.source_format.lower() == "json" and request.target_format.lower() == "markdown":
                return self._json_to_markdown(request)
            else:
                return ConversionResponse(
                    success=False,
                    error=f"不支持的转换类型: {request.source_format} -> {request.target_format}"
                )
        
        except Exception as e:
            return ConversionResponse(
                success=False,
                error=f"转换过程中发生错误: {str(e)}"
            )
    
    def _markdown_to_json(self, request: ConversionRequest) -> ConversionResponse:
        """Markdown转JSON"""
        if not isinstance(request.content, str):
            return ConversionResponse(
                success=False,
                error="Markdown内容必须是字符串类型"
            )
        
        # 解析Markdown
        document = self.parser.parse(request.content)
        
        # 转换为JSON
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
        """JSON转Markdown"""
        if isinstance(request.content, str):
            try:
                json_data = json.loads(request.content)
            except json.JSONDecodeError as e:
                return ConversionResponse(
                    success=False,
                    error=f"JSON解析错误: {str(e)}"
                )
        elif isinstance(request.content, dict):
            json_data = request.content
        else:
            return ConversionResponse(
                success=False,
                error="JSON内容必须是字符串或字典类型"
            )
        
        # 从JSON创建文档对象
        try:
            document = MarkdownDocument(**json_data)
        except Exception as e:
            return ConversionResponse(
                success=False,
                error=f"JSON数据格式错误: {str(e)}"
            )
        
        # 设置渲染选项
        if request.options:
            self.renderer = MarkdownRenderer(request.options)
        
        # 渲染为Markdown
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


# 便捷函数
def markdown_to_json(markdown_text: str, options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    """Markdown文本转JSON（便捷函数）"""
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
        raise ValueError(f"转换失败: {response.error}")


def json_to_markdown(json_data: Union[str, Dict[str, Any]], options: Optional[ConversionOptions] = None) -> str:
    """JSON转Markdown文本（便捷函数）"""
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
        raise ValueError(f"转换失败: {response.error}")
