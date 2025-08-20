#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown <-> JSON 转换（表格精简版）
精简点：
  - TableElement 直接使用 rows: List[List[str]]
  - 不再通过 children 构造 TABLE_ROW / TABLE_CELL
  - 兼容旧三种格式并自动升级：
      1) children = [table_row(children=[table_cell...])]
      2) content = '{"headers":[...],"rows":[...]}'
      3) attributes['rows'] = [...]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

from app.type import (
    MarkdownDocument,
    MarkdownElement,
    HeadingElement,
    CodeBlockElement,
    ListElement,
    LinkElement,
    ImageElement,
    TableElement,
    ElementType,
    ListType,
    TableAlignment,
    ConversionOptions,
    ConversionRequest,
    ConversionResponse,
)
from app.config import settings

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown <-> JSON 转换（重构版）
改动要点：
  - 递归块解析：引入 _parse_block_sequence
  - 列表项内部再次块级解析（支持表格 / 标题 / HR / 代码块等）
  - 支持 Setext 标题（= 与 - 下划线）
  - 表格解析：对齐、列数填空（不足补空，超出截断）
  - 任务列表：attributes['task']=True, attributes['checked']=bool
  - 单一段落 list item 折叠为 content
  - 水平线 HR 元素 content=None
  - 保留 InlineParser（strong/em/del/code span/link）行为
  - 兼容脚注占位
"""

# ========== 正则 ==========
_RE_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)(\s+#+\s*)?$")
_RE_SETEXT_H1 = re.compile(r"^=+$")
_RE_SETEXT_H2 = re.compile(r"^-+$")
_RE_CODE_FENCE = re.compile(r"^```(\w+)?\s*$")
_RE_ORDERED_ITEM = re.compile(r"^(\d+)\.\s+(.*)$")
_RE_UNORDERED_ITEM = re.compile(r"^[-*+]\s+(.*)$")
_RE_TASK_ITEM = re.compile(r"^[-*+]\s+\[( |x|X)\]\s+(.*)$")
_RE_HRULE = re.compile(r"^(?:\s*)(?:[*-_]\s*){3,}$")
_RE_IMAGE_FULL = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
_RE_IMAGE_INLINE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_BOLD = re.compile(r"(?<!\*)\*\*([^\*]+)\*\*(?!\*)")
_RE_ITALIC = re.compile(r"(?<!\*)\*([^\*]+)\*(?!\*)")
_RE_STRIKE = re.compile(r"~~([^~]+)~~")
_RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_BLOCKQUOTE = re.compile(r"^>\s?(.*)$")
_RE_TABLE_ALIGN_CELL = re.compile(r"^:?-{3,}:?$")
_RE_FOOTNOTE_DEF = re.compile(r"^\[\^([^\]]+)\]:\s+(.*)$")
_RE_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")

# ========== 内联解析 ==========
class InlineParser:
    def __init__(self, enable_links: bool = True, enable_format: bool = True):
        self.enable_links = enable_links
        self.enable_format = enable_format

    def parse(self, text: str) -> str:
        if not text:
            return ""
        text = _RE_ESCAPE.sub(lambda m: m.group(1), text)
        code_spans: List[str] = []

        def code_repl(m):
            code_spans.append(m.group(1))
            return f"[[CODE_SPAN_{len(code_spans)-1}]]"

        text = _RE_INLINE_CODE.sub(code_repl, text)

        if self.enable_format:
            text = _RE_BOLD.sub(r"<strong>\1</strong>", text)
            text = _RE_ITALIC.sub(r"<em>\1</em>", text)
            text = _RE_STRIKE.sub(r"<del>\1</del>", text)

        if self.enable_links:
            text = _RE_LINK.sub(r'<a href="\2">\1</a>', text)

        for i, raw in enumerate(code_spans):
            esc = (
                raw.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;")
            )
            text = text.replace(f"[[CODE_SPAN_{i}]]", f"<code>{esc}</code>")
        return text

@dataclass
class _ListItemAcc:
    marker_line: str
    lines: List[str]
    ordered: bool
    start_number: Optional[int]
    task: bool
    checked: Optional[bool]

class MarkdownParser:
    def __init__(self, options: Optional[ConversionOptions] = None):
        self.options = options or ConversionOptions()
        self.lines: List[str] = []
        self.pos: int = 0
        self.inline = InlineParser()
        self._footnotes: Dict[str, str] = {}

    # ---------- Public API ----------
    def parse(self, markdown_text: str) -> MarkdownDocument:
        self.lines = markdown_text.splitlines()
        self.pos = 0
        self._footnotes.clear()
        metadata = self._extract_metadata()

        elements = self._parse_block_sequence(self.lines)

        title = None
        if elements and isinstance(elements[0], HeadingElement) and elements[0].level == 1:
            title = elements[0].content

        return MarkdownDocument(
            title=title,
            metadata=metadata,
            content=elements,
        )

    # ---------- 基础工具 ----------
    def _eof(self) -> bool:
        return self.pos >= len(self.lines)

    def _cur(self) -> str:
        return self.lines[self.pos] if self.pos < len(self.lines) else ""

    def _peek(self, k=1) -> str:
          i = self.pos + k
          return self.lines[i] if i < len(self.lines) else ""

    @staticmethod
    def _indent(line: str) -> int:
        expanded = line.replace("\t", "    ")
        return len(expanded) - len(expanded.lstrip(" "))

    # ---------- 元数据采集 ----------
    def _extract_metadata(self) -> Optional[Dict[str, Any]]:
        saved = self.pos
        while not self._eof() and not self._cur().strip():
            self.pos += 1
        if self._eof():
            return None
        line = self._cur().strip()
        if line.startswith("<!--") and "Metadata:" in line:
            self.pos += 1
            meta: Dict[str, Any] = {}
            while not self._eof():
                t = self._cur().strip()
                if t == "-->":
                    self.pos += 1
                    break
                if ":" in t:
                    k, v = t.split(":", 1)
                    meta[k.strip()] = v.strip()
                self.pos += 1
            return meta or None
        self.pos = saved
        return None

    # ---------- 顶层（复用）块序列解析 ----------
    def _parse_block_sequence(self, lines: List[str]) -> List[MarkdownElement]:
        """
        对传入的行数组执行完整块级解析。使用局部游标以支持递归。
        """
        saved_lines, saved_pos = self.lines, self.pos
        self.lines, self.pos = lines, 0
        out: List[MarkdownElement] = []
        try:
            while not self._eof():
                # 跳过空行
                if not self._cur().strip():
                    self.pos += 1
                    continue
                el = self._parse_block()
                if el:
                    out.append(el)
        finally:
            self.lines, self.pos = saved_lines, saved_pos
        return out

    # ---------- Block 分发 ----------
    def _parse_block(self) -> Optional[MarkdownElement]:
        # 跳过连续空
        while not self._eof() and not self._cur().strip():
            self.pos += 1
        if self._eof():
            return None

        line = self._cur()
        stripped = line.strip()

        # 代码块
        m_code = _RE_CODE_FENCE.match(stripped)
        if m_code:
            return self._parse_code_block(m_code.group(1))

        # ATX Heading
        m_h = _RE_ATX_HEADING.match(stripped)
        if m_h:
            return self._parse_atx_heading(m_h)

        # Setext Heading (需要当前行为纯文本且下一行为 === 或 ---)
        if self._is_setext_heading():
            return self._parse_setext_heading()

        # Horizontal Rule
        if _RE_HRULE.match(stripped):
            self.pos += 1
            return MarkdownElement(type=ElementType.HORIZONTAL_RULE, content=None)

        # Blockquote
        if stripped.startswith(">"):
            return self._parse_blockquote()

        # 表格（需要当前行为表头 & 下一行分隔）
        if self._looks_like_table_header():
            tbl = self._parse_table()
            if tbl:
                return tbl

        # 脚注定义
        m_fn = _RE_FOOTNOTE_DEF.match(stripped)
        if m_fn:
            self._footnotes[m_fn.group(1)] = m_fn.group(2)
            self.pos += 1
            # 不输出可见元素
            return MarkdownElement(type=ElementType.PARAGRAPH, content="")

        # 列表
        if self._is_list_line(stripped):
            return self._parse_list()

        # 段落（可能退化也可能只有一行）
        return self._parse_paragraph()

    # ---------- 辅助判定 ----------
    def _is_list_line(self, stripped: str) -> bool:
        return (
            _RE_TASK_ITEM.match(stripped)
            or _RE_ORDERED_ITEM.match(stripped)
            or _RE_UNORDERED_ITEM.match(stripped)
        )

    def _is_setext_heading(self) -> bool:
        """
        当前行非空，下一行是全 '=' 或 '-'，且当前行不是 ATX 头/列表/HR/代码/表格。
        """
        if self.pos + 1 >= len(self.lines):
            return False
        cur = self._cur().rstrip()
        next_line = self._peek().strip()
        if not cur.strip():
            return False
        # 不与 ATX 冲突
        if _RE_ATX_HEADING.match(cur.strip()):
            return False
        # 不将表格第一行误判为 setext
        if self._looks_like_table_header():
            return False
        if _RE_SETEXT_H1.match(next_line) or _RE_SETEXT_H2.match(next_line):
            # 排除只含 --- 但应该是 HR 的情况：HR 单独行，上一行非文本不构成 setext
            return True
        return False

    def _looks_like_table_header(self) -> bool:
        """
        条件：当前行含 '|'，下一行是对齐分隔行（含至少一个管道并由 - / : 组成）。
        """
        if self.pos + 1 >= len(self.lines):
            return False
          # 允许前导空格
        header = self._cur().rstrip()
        divider = self._peek().strip()
        if "|" not in header or "|" not in divider:
            return False
        # 校验分隔行各 cell 合法
        cells = [c.strip() for c in divider.strip().strip("|").split("|")]
        if not cells:
            return False
        # 所有 cell 满足对齐模式
        if all(_RE_TABLE_ALIGN_CELL.match(c.replace(" ", "")) for c in cells):
            # 确认 header 里至少有一个非空 cell
            header_cells = [c.strip() for c in header.strip().strip("|").split("|")]
            if any(hc for hc in header_cells):
                return True
        return False

    # ---------- 具体解析 ----------
    def _parse_atx_heading(self, m) -> HeadingElement:
        hashes, text, _tail = m.group(1), m.group(2), m.group(3)
        level = len(hashes)
        self.pos += 1
        return HeadingElement(level=level, content=self.inline.parse(text.strip()))

    def _parse_setext_heading(self) -> HeadingElement:
        cur_line = self._cur().rstrip()
        underline = self._peek().strip()
        self.pos += 2
        level = 1 if _RE_SETEXT_H1.match(underline) else 2
        return HeadingElement(level=level, content=self.inline.parse(cur_line.strip()))

    def _parse_code_block(self, lang: Optional[str]) -> CodeBlockElement:
        self.pos += 1
        buf: List[str] = []
        while not self._eof():
            line = self._cur()
            if _RE_CODE_FENCE.match(line.strip()):
                self.pos += 1
                break
            buf.append(line.rstrip("\n"))
            self.pos += 1
        return CodeBlockElement(language=lang or None, content="\n".join(buf))

    def _parse_blockquote(self) -> MarkdownElement:
        buf: List[str] = []
        while not self._eof():
            raw = self._cur()
            stripped = raw.strip()
            m = _RE_BLOCKQUOTE.match(stripped)
            if m:
                buf.append(m.group(1))
                self.pos += 1
                continue
            if not stripped:
                buf.append("")
                self.pos += 1
                continue
            break
        # 递归解析内部（可选：当前实现合并为一个段落或二次解析）
        inner = self._parse_block_sequence(buf)
        # 如果 inner 只有一个 paragraph，可折叠
        if len(inner) == 1 and inner[0].type == ElementType.PARAGRAPH:
            return MarkdownElement(type=ElementType.BLOCKQUOTE, content=inner[0].content)
        return MarkdownElement(type=ElementType.BLOCKQUOTE, children=inner)

    def _parse_image_full_line(self, m) -> ImageElement:
        self.pos += 1
        alt = m.group(1)
        src_title = m.group(2)
        title = None
        if " \"" in src_title:
            p = src_title.split(" \"", 1)
            src = p[0]
            title = p[1].rstrip('"')
        else:
            src = src_title
        return ImageElement(src=src, alt=alt, title=title)

    def _parse_paragraph(self) -> MarkdownElement:
        start = self.pos
        lines: List[str] = []
        while not self._eof():
            raw = self._cur()
            stripped = raw.strip()
            # Lookahead 终止条件
            if (
                not stripped
                or _RE_ATX_HEADING.match(stripped)
                or _RE_CODE_FENCE.match(stripped)
                or _RE_HRULE.match(stripped)
                or self._is_list_line(stripped)
                or self._is_setext_heading()
                or self._looks_like_table_header()
                or stripped.startswith(">")
                or _RE_FOOTNOTE_DEF.match(stripped)
            ):
                break
            lines.append(raw.rstrip())
            self.pos += 1

        text = "\n".join(lines).strip()
        return MarkdownElement(type=ElementType.PARAGRAPH, content=self.inline.parse(text))

    def _parse_table(self) -> Optional[TableElement]:
        """
        解析表格：当前行为 header，下一行为对齐分隔。
        行读取直到遇到空行或不是表格格式的行。
        不同列数：不足补空，多余截断。
        """
        header_line = self._cur().strip()
        divider_line = self._peek().strip()
        header_cells = [c.strip() for c in header_line.strip().strip("|").split("|")]
        align_cells = [c.strip() for c in divider_line.strip().strip("|").split("|")]
        self.pos += 2

        # 解析对齐
        alignments: List[TableAlignment] = []
          # 参考 Markdown 表格对齐判断
        for ac in align_cells:
            raw = ac.replace(" ", "")
            if raw.startswith(":") and raw.endswith(":"):
                alignments.append(TableAlignment.CENTER)
            elif raw.startswith(":"):
                alignments.append(TableAlignment.LEFT)
            elif raw.endswith(":"):
                alignments.append(TableAlignment.RIGHT)
            else:
                alignments.append(TableAlignment.LEFT)
        # 对齐长度与 header 对齐
        if len(alignments) < len(header_cells):
            alignments.extend([TableAlignment.LEFT] * (len(header_cells) - len(alignments)))
        elif len(alignments) > len(header_cells):
            alignments = alignments[: len(header_cells)]

        rows: List[List[str]] = []
        # 后续数据行
        while not self._eof():
            raw = self._cur()
            stripped = raw.strip()
            if not stripped:
                break
            # 允许前导空格
            if "|" not in stripped:
                break
            # 简单判定：符合表格行（有至少一个管道）
            cells = [c.strip() for c in stripped.strip().strip("|").split("|")]
            # 填空/截断
            if len(cells) < len(header_cells):
                cells.extend([""] * (len(header_cells) - len(cells)))
            elif len(cells) > len(header_cells):
                cells = cells[: len(header_cells)]
            rows.append(cells)
            self.pos += 1

        return TableElement(
            headers=header_cells,
            rows=rows,
            alignments=alignments,
        )

    # ---------- 列表解析（递归） ----------
    def _parse_list(self) -> ListElement:
        """
        收集整个同级列表，然后拆分每个列表项，再对每项内容递归 _parse_block_sequence。
        """
        list_indent = self._indent(self._cur())
        collected: List[str] = []
        start_pos = self.pos
        # 首轮：收集所有属于此列表的行
        while not self._eof():
            raw = self._cur()
            stripped = raw.strip()
            if not stripped:
                collected.append(raw)
                self.pos += 1
                continue
            cur_indent = self._indent(raw)
            # 同级新项
            if self._is_list_line(stripped) and cur_indent == list_indent:
                collected.append(raw)
                self.pos += 1
                continue
            # 继续行 / 子块（更深缩进）
            if cur_indent > list_indent:
                collected.append(raw)
                self.pos += 1
                continue
            # 其他：列表结束
            break

        # 拆分列表项
        items: List[_ListItemAcc] = []
        current: Optional[_ListItemAcc] = None

        def push_current():
            nonlocal current
            if current is None:
                return
            items.append(current)
            current = None

        for raw in collected:
            stripped = raw.strip()
            if not stripped:
                if current:
                    current.lines.append("")
                continue
            # 新项
            if self._is_list_line(stripped) and self._indent(raw) == list_indent:
                push_current()
                # 判定类型
                m_task = _RE_TASK_ITEM.match(stripped)
                if m_task:
                    checked = (m_task.group(1).lower() == "x")
                    text_after = m_task.group(2)
                    current = _ListItemAcc(
                        marker_line=raw,
                        lines=[text_after],
                        ordered=False,
                        start_number=None,
                        task=True,
                        checked=checked,
                    )
                    continue
                m_ord = _RE_ORDERED_ITEM.match(stripped)
                if m_ord:
                    start_num = int(m_ord.group(1))
                    after = m_ord.group(2)
                    current = _ListItemAcc(
                        marker_line=raw,
                        lines=[after],
                        ordered=True,
                        start_number=start_num,
                        task=False,
                        checked=None,
                    )
                    continue
                m_un = _RE_UNORDERED_ITEM.match(stripped)
                if m_un:
                    current = _ListItemAcc(
                        marker_line=raw,
                        lines=[m_un.group(1)],
                        ordered=False,
                        start_number=None,
                        task=False,
                        checked=None,
                    )
                    continue
            else:
                # 继续行：去除 list_indent+2 左右的基础缩进
                if current:
                    # 计算最小可裁剪缩进（list_indent + 2）
                    logical = raw
                    cut_base = list_indent + 2
                    if self._indent(raw) >= cut_base:
                        # 裁剪至相对内容
                        rel = raw[cut_base:]
                        logical = rel
                    current.lines.append(logical)
        push_current()

        # 确定列表类型与起始
        ordered = any(it.ordered for it in items)
        # 若混合（不建议）按出现第一个有序判断
        first_ordered = next((it for it in items if it.ordered), None)
        start_number = first_ordered.start_number if first_ordered else None

        list_children: List[MarkdownElement] = []
        for acc in items:
            # 递归解析
            children = self._parse_block_sequence(acc.lines)
            # 折叠策略：单 paragraph
            if (
                len(children) == 1
                and children[0].type == ElementType.PARAGRAPH
            ):
                attrs = {}
                if acc.task:
                    attrs["task"] = True
                    attrs["checked"] = bool(acc.checked)
                list_children.append(
                    MarkdownElement(
                        type=ElementType.LIST_ITEM,
                        content=children[0].content,
                        attributes=attrs or None,
                    )
                )
            else:
                attrs = {}
                if acc.task:
                    attrs["task"] = True
                    attrs["checked"] = bool(acc.checked)
                list_children.append(
                    MarkdownElement(
                        type=ElementType.LIST_ITEM,
                        children=children,
                        attributes=attrs or None,
                    )
                )

        list_type = ListType.ORDERED if ordered else ListType.UNORDERED
        if list_type == ListType.ORDERED and start_number is None:
            start_number = 1

        return ListElement(
            list_type=list_type,
            start=start_number if list_type == ListType.ORDERED else None,
            children=list_children,
        )

    # ========== 调试 / 可选工具 ==========
    def debug_dump(self, doc: MarkdownDocument) -> str:
        """
        简单调试：返回 JSON。
        """
        def to_dict(el: MarkdownElement) -> Any:
            base = {
                "type": el.type.name if hasattr(el.type, "name") else str(el.type),
            }
            if getattr(el, "level", None):
                base["level"] = getattr(el, "level")
            if getattr(el, "language", None):
                base["language"] = getattr(el, "language")
            if getattr(el, "list_type", None):
                base["list_type"] = getattr(el, "list_type").name
            if getattr(el, "start", None):
                base["start"] = getattr(el, "start")
            if isinstance(el, TableElement):
                base["headers"] = el.headers
                base["rows"] = el.rows
                base["alignments"] = [a.name for a in el.alignments] if el.alignments else None
            if el.content is not None:
                base["content"] = el.content
            if el.children:
                base["children"] = [to_dict(c) for c in el.children]
            if el.attributes:
                base["attributes"] = el.attributes
            return base
        return json.dumps([to_dict(c) for c in doc.content], ensure_ascii=False, indent=2)


# ========== 如果需要：对外使用示例函数 ==========
def convert_markdown(markdown_text: str, options: Optional[ConversionOptions] = None) -> MarkdownDocument:
    parser = MarkdownParser(options=options)
    return parser.parse(markdown_text)

# ========== 渲染器 ==========
class MarkdownRenderer:
    def __init__(self, options: Optional[ConversionOptions] = None):
        self.options = options or ConversionOptions()

    def render(self, doc: MarkdownDocument) -> str:
        out: List[str] = []
        if doc.title and not self._has_h1(doc.content):
            out.append(f"# {doc.title}\n")
        if doc.metadata and self.options.include_metadata:
            out.append("<!-- Metadata:")
            for k, v in doc.metadata.items():
                out.append(f"{k}: {v}")
            out.append("-->\n")
        for el in doc.content:
            out.append(self._render_element(el))
        return "\n".join(filter(None, out)).rstrip() + "\n"

    def _has_h1(self, elements: List[MarkdownElement]) -> bool:
        return any(isinstance(e, HeadingElement) and e.level == 1 for e in elements)

    def _render_element(self, el: MarkdownElement) -> str:
        etype = el.type
        if etype == ElementType.HEADING:
            level = el.attributes.get("level") if el.attributes else getattr(el, "level", 1)
            return f"{'#' * int(level)} {el.content}\n"
        if etype == ElementType.PARAGRAPH:
            return f"{el.content}\n"
        if etype == ElementType.CODE_BLOCK:
            lang = el.attributes.get("language", "") if el.attributes else ""
            return f"```{lang}\n{el.content or ''}\n```\n"
        if etype == ElementType.BLOCKQUOTE:
            return "\n".join(f"> {l}" if l else ">" for l in (el.content or "").splitlines()) + "\n"
        if etype == ElementType.HORIZONTAL_RULE:
            return "---\n"
        if etype == ElementType.IMAGE:
            src = el.attributes.get("src","") if el.attributes else ""
            alt = el.attributes.get("alt","") if el.attributes else ""
            title = el.attributes.get("title","") if el.attributes else ""
            t = f' "{title}"' if title else ""
            return f"![{alt}]({src}{t})\n"
        if etype == ElementType.LIST:
            return self._render_list(el)
        if etype == ElementType.TABLE:
            return self._render_table(el)
        return el.content or ""

    def _render_list(self, el: MarkdownElement, indent=0) -> str:
        children = el.children or []
        list_type = el.attributes.get("list_type") if el.attributes else ListType.UNORDERED
        start = el.attributes.get("start", 1) if el.attributes else 1
        lines: List[str] = []
        num = start
        for item in children:
            if item.type != ElementType.LIST_ITEM:
                if item.type == ElementType.LIST:
                    lines.append(self._render_list(item, indent))
                continue
            marker = f"{num}." if list_type == ListType.ORDERED else "-"
            if list_type == ListType.ORDERED:
                num += 1
            pad = "  " * indent
            lines.append(f"{pad}{marker} {item.content}".rstrip())
            if item.children:
                for sub in item.children:
                    if sub.type == ElementType.LIST:
                        lines.append(self._render_list(sub, indent + 1))
                    else:
                        if sub.content:
                            lines.append("  " * (indent + 1) + sub.content)
        return "\n".join(lines) + "\n"

    def _render_table(self, el: TableElement) -> str:
        # 升级旧格式（若存在）
        self._upgrade_table_element(el)
        headers = el.headers
        rows = el.rows
        aligns = el.alignments or [TableAlignment.NONE] * len(headers)
        align_map = {
            TableAlignment.LEFT: ":---",
            TableAlignment.CENTER: ":---:",
            TableAlignment.RIGHT: "---:",
            TableAlignment.NONE: "---"
        }
        header_line = "|" + "|".join(h or "" for h in headers) + "|"
        align_line = "|" + "|".join(align_map.get(a, "---") for a in aligns) + "|"
        body_lines = []
        for r in rows:
            if len(r) < len(headers):
                r = r + [""] * (len(headers) - len(r))
            elif len(r) > len(headers):
                r = r[:len(headers)]
            body_lines.append("|" + "|".join(r) + "|")
        return "\n".join([header_line, align_line] + body_lines) + "\n" + "\n"

    # 升级旧格式 => rows
    def _upgrade_table_element(self, el: TableElement):
        if el.rows:  # 已是新格式
            return
        # 1) content JSON
        if el.content:
            try:
                data = json.loads(el.content)
                if isinstance(data, dict) and "rows" in data:
                    el.rows = [[str(c) if c is not None else "" for c in row] for row in data.get("rows", []) if isinstance(row, list)]
                    el.content = None
            except Exception:
                pass
        # 2) attributes['rows']
        if not el.rows and el.attributes and "rows" in el.attributes:
            raw_rows = el.attributes.pop("rows")
            if isinstance(raw_rows, list):
                el.rows = [[str(c) if c is not None else "" for c in row] for row in raw_rows if isinstance(row, list)]
        # 3) children (table_row/table_cell)
        if not el.rows and el.children:
            new_rows: List[List[str]] = []
            for r in el.children:
                if r.type == ElementType.TABLE_ROW:
                    cells: List[str] = []
                    if r.children:
                        for c in r.children:
                            if c.type == ElementType.TABLE_CELL:
                                cells.append(c.content or "")
                    new_rows.append(cells)
            if new_rows:
                el.rows = new_rows
            el.children = None  # 清理冗余

# ========== 转换器 ==========
class MarkdownConverter:
    def __init__(self):
        self.parser = MarkdownParser()
        self.renderer = MarkdownRenderer()

    def convert(self, request: ConversionRequest) -> ConversionResponse:
        try:
            src = request.source_format.lower()
            tgt = request.target_format.lower()
            if src == tgt:
                return ConversionResponse(success=True, result=request.content)
            if src == "markdown" and tgt == "json":
                return self._markdown_to_json(request)
            if src == "json" and tgt == "markdown":
                return self._json_to_markdown(request)
            return ConversionResponse(success=False, error=f"不支持的转换类型: {request.source_format} -> {request.target_format}")
        except Exception as e:
            return ConversionResponse(success=False, error=f"转换过程中发生错误: {e}")

    def _markdown_to_json(self, req: ConversionRequest) -> ConversionResponse:
        if not isinstance(req.content, str):
            return ConversionResponse(success=False, error="Markdown内容必须是字符串")
        doc = self.parser.parse(req.content)
        # 序列化
        payload = doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
        return ConversionResponse(
            success=True,
            result=payload,
            metadata={
                "elements_count": len(doc.content),
                "has_metadata": doc.metadata is not None,
                "has_title": doc.title is not None
            }
        )

    def _json_to_markdown(self, req: ConversionRequest) -> ConversionResponse:
        # 解析 JSON
        if isinstance(req.content, str):
            try:
                data = json.loads(req.content)
            except json.JSONDecodeError as e:
                return ConversionResponse(success=False, error=f"JSON解析错误: {e}")
        else:
            data = req.content
        try:
            doc = MarkdownDocument(**data)
        except Exception as e:
            return ConversionResponse(success=False, error=f"JSON数据格式错误: {e}")

        # 升级所有表格
        self._upgrade_tables_in_doc(doc)

        if req.options:
            self.renderer = MarkdownRenderer(req.options)
        md = self.renderer.render(doc)
        return ConversionResponse(
            success=True,
            result=md,
            metadata={
                "elements_count": len(doc.content),
                "has_metadata": doc.metadata is not None,
                "has_title": doc.title is not None
            }
        )

    def _upgrade_tables_in_doc(self, doc: MarkdownDocument):
        def visit(el: MarkdownElement):
            if el.type == ElementType.TABLE and isinstance(el, TableElement):
                self.renderer._upgrade_table_element(el)
            if el.children:
                for c in el.children:
                    visit(c)
        for e in doc.content:
            visit(e)

# ========== 便捷函数 ==========
def markdown_to_json(markdown_text: str, options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    conv = MarkdownConverter()
    req = ConversionRequest(source_format="markdown", target_format="json", content=markdown_text, options=options)
    resp = conv.convert(req)
    if resp.success:
        return resp.result  # type: ignore
    raise ValueError(resp.error or "转换失败")

def json_to_markdown(json_data: Union[str, Dict[str, Any]], options: Optional[ConversionOptions] = None) -> str:
    conv = MarkdownConverter()
    req = ConversionRequest(source_format="json", target_format="markdown", content=json_data, options=options)
    resp = conv.convert(req)
    if resp.success:
        return resp.result  # type: ignore
    raise ValueError(resp.error or "转换失败")