from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum


class ElementType(str, Enum):
    """Markdown元素类型枚举"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    LIST = "list"
    LIST_ITEM = "list_item"
    BLOCKQUOTE = "blockquote"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    LINK = "link"
    IMAGE = "image"
    BOLD = "bold"
    ITALIC = "italic"
    STRIKETHROUGH = "strikethrough"
    HORIZONTAL_RULE = "horizontal_rule"
    LINE_BREAK = "line_break"
    TEXT = "text"


class ListType(str, Enum):
    """列表类型枚举"""
    ORDERED = "ordered"
    UNORDERED = "unordered"


class TableAlignment(str, Enum):
    """表格对齐方式枚举"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    NONE = "none"


class MarkdownElement(BaseModel):
    """Markdown元素基础模型"""
    type: ElementType = Field(..., description="元素类型")
    content: Optional[str] = Field(None, description="文本内容")
    children: Optional[List['MarkdownElement']] = Field(None, description="子元素列表")
    attributes: Optional[Dict[str, Any]] = Field(None, description="元素属性")
    
    class Config:
        # 允许递归模型
        from_attributes = True


class HeadingElement(MarkdownElement):
    """标题元素"""
    type: Literal[ElementType.HEADING] = Field(ElementType.HEADING)
    level: int = Field(..., ge=1, le=6, description="标题级别 (1-6)")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        self.attributes["level"] = self.level


class CodeBlockElement(MarkdownElement):
    """代码块元素"""
    type: Literal[ElementType.CODE_BLOCK] = Field(ElementType.CODE_BLOCK)
    language: Optional[str] = Field(None, description="编程语言")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        if self.language:
            self.attributes["language"] = self.language


class ListElement(MarkdownElement):
    """列表元素"""
    type: Literal[ElementType.LIST] = Field(ElementType.LIST)
    list_type: ListType = Field(..., description="列表类型")
    start: Optional[int] = Field(None, description="有序列表起始数字")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        self.attributes["list_type"] = self.list_type
        if self.start is not None:
            self.attributes["start"] = self.start


class LinkElement(MarkdownElement):
    """链接元素"""
    type: Literal[ElementType.LINK] = Field(ElementType.LINK)
    url: str = Field(..., description="链接URL")
    title: Optional[str] = Field(None, description="链接标题")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        self.attributes["url"] = self.url
        if self.title:
            self.attributes["title"] = self.title


class ImageElement(MarkdownElement):
    """图片元素"""
    type: Literal[ElementType.IMAGE] = Field(ElementType.IMAGE)
    src: str = Field(..., description="图片源URL")
    alt: Optional[str] = Field(None, description="替代文本")
    title: Optional[str] = Field(None, description="图片标题")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        self.attributes["src"] = self.src
        if self.alt:
            self.attributes["alt"] = self.alt
        if self.title:
            self.attributes["title"] = self.title


class TableElement(MarkdownElement):
    """表格元素（精简版：直接 rows）"""
    type: Literal[ElementType.TABLE] = Field(ElementType.TABLE)
    headers: List[str] = Field(default_factory=list, description="表头")
    alignments: List[TableAlignment] = Field(default_factory=list, description="列对齐方式")
    rows: List[List[str]] = Field(default_factory=list, description="数据行 (二维数组)")

    def __init__(self, **data):
        super().__init__(**data)
        if self.attributes is None:
            self.attributes = {}
        if self.headers:
            self.attributes["headers"] = self.headers
        if self.alignments:
            self.attributes["alignments"] = [a.value for a in self.alignments]



class MarkdownDocument(BaseModel):
    """Markdown文档模型"""
    title: Optional[str] = Field(None, description="文档标题")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")
    content: List[MarkdownElement] = Field(default_factory=list, description="文档内容元素列表")
    
    class Config:
        from_attributes = True


class ConversionOptions(BaseModel):
    """转换选项配置"""
    preserve_html: bool = Field(True, description="是否保留HTML标签")
    include_metadata: bool = Field(True, description="是否包含元数据")
    flatten_structure: bool = Field(False, description="是否扁平化结构")
    custom_renderers: Optional[Dict[str, str]] = Field(None, description="自定义渲染器")


class ConversionRequest(BaseModel):
    """转换请求模型"""
    source_format: str = Field(..., description="源格式 (markdown/json)")
    target_format: str = Field(..., description="目标格式 (json/markdown)")
    content: Union[str, Dict[str, Any]] = Field(..., description="待转换内容")
    options: Optional[ConversionOptions] = Field(None, description="转换选项")


class ConversionResponse(BaseModel):
    """转换响应模型"""
    success: bool = Field(..., description="转换是否成功")
    result: Optional[Union[str, Dict[str, Any]]] = Field(None, description="转换结果")
    error: Optional[str] = Field(None, description="错误信息")
    metadata: Optional[Dict[str, Any]] = Field(None, description="转换元数据")


# 更新递归模型引用
MarkdownElement.model_rebuild()