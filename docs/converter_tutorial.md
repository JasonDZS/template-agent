# Markdown & JSON 转换器使用教程

## 概述

本转换器提供了 Markdown 和 JSON 格式之间的高质量双向转换功能，支持复杂的文档结构、表格、代码块等元素，并实现了近乎无损的转换效果。

## 核心特性

- ✅ **双向转换**：Markdown ↔ JSON 互相转换
- ✅ **结构化解析**：支持标题层次、列表、表格、代码块等
- ✅ **表格支持**：完整支持 Markdown 表格及对齐方式
- ✅ **元数据处理**：支持文档元数据的提取和重构
- ✅ **高保真度**：近乎无损的转换质量
- ✅ **回转稳定**：多次转换结果一致

## 快速开始

### 基本导入

```python
from app.converter import (
    markdown_to_json,
    json_to_markdown,
    MarkdownConverter,
    ConversionRequest,
    ConversionOptions
)
```

### 简单转换示例

```python
# Markdown 转 JSON
markdown_text = """# 标题

这是一个段落。

## 子标题

- 列表项 1
- 列表项 2
"""

# 转换为 JSON
json_result = markdown_to_json(markdown_text)
print(json_result)

# JSON 转回 Markdown
regenerated_markdown = json_to_markdown(json_result)
print(regenerated_markdown)
```

## 详细用法

### 1. 便捷函数

#### `markdown_to_json()`

```python
def markdown_to_json(markdown_text: str, options: Optional[ConversionOptions] = None) -> Dict[str, Any]:
    \"\"\"将 Markdown 文本转换为 JSON 格式\"\"\"
```

**参数**：
- `markdown_text` (str): 要转换的 Markdown 文本
- `options` (ConversionOptions, 可选): 转换选项配置

**返回**：JSON 格式的字典

**示例**：
```python
markdown = """# 企业报告

## 财务数据

| 指标 | 2023年 | 2024年 |
|------|--------|--------|
| 收入 | 100万 | 120万 |
| 利润 | 20万  | 25万  |
"""

json_data = markdown_to_json(markdown)
```

#### `json_to_markdown()`

```python
def json_to_markdown(json_data: Union[str, Dict[str, Any]], options: Optional[ConversionOptions] = None) -> str:
    \"\"\"将 JSON 数据转换为 Markdown 文本\"\"\"
```

**参数**：
- `json_data`: JSON 数据（字符串或字典格式）
- `options` (ConversionOptions, 可选): 转换选项配置

**返回**：Markdown 格式的文本

### 2. 高级用法 - MarkdownConverter

```python
from app.converter import MarkdownConverter, ConversionRequest, ConversionOptions

# 创建转换器实例
converter = MarkdownConverter()

# 配置转换选项
options = ConversionOptions(
    preserve_html=True,          # 保留 HTML 标签
    include_metadata=True,       # 包含元数据
    flatten_structure=False      # 保持层级结构
)

# 创建转换请求
request = ConversionRequest(
    source_format=\"markdown\",
    target_format=\"json\",
    content=markdown_text,
    options=options
)

# 执行转换
response = converter.convert(request)

if response.success:
    print(\"转换成功！\")
    print(response.result)
    print(f\"元素数量: {response.metadata['elements_count']}\")
else:
    print(f\"转换失败: {response.error}\")
```

### 3. 支持的 Markdown 元素

#### 标题 (Headings)
```markdown
# 一级标题
## 二级标题
### 三级标题
```

#### 段落 (Paragraphs)
```markdown
这是一个普通段落。

这是另一个段落，包含 **粗体** 和 *斜体* 文本。
```

#### 列表 (Lists)
```markdown
# 无序列表
- 项目 1
- 项目 2
- 项目 3

# 有序列表
1. 第一项
2. 第二项
3. 第三项
```

#### 表格 (Tables)
```markdown
| 列标题1 | 列标题2 | 列标题3 |
|:--------|:-------:|--------:|
| 左对齐  | 居中    | 右对齐  |
| 数据1   | 数据2   | 数据3   |
```

支持的对齐方式：
- `:---` 左对齐
- `:---:` 居中对齐  
- `---:` 右对齐
- `---` 默认对齐

#### 代码块 (Code Blocks)
```markdown
\`\`\`python
def hello_world():
    print(\"Hello, World!\")
\`\`\`
```

#### 引用 (Blockquotes)
```markdown
> 这是一个引用文本。
> 可以跨越多行。
```

#### 水平分割线 (Horizontal Rules)
```markdown
---
```

#### 图片 (Images)
```markdown
![图片描述](图片路径 \"可选标题\")
```

#### 链接 (Links)
```markdown
[链接文本](链接地址 \"可选标题\")
```

### 4. 转换选项配置

```python
from app.type import ConversionOptions

options = ConversionOptions(
    preserve_html=True,          # 是否保留 HTML 标签
    include_metadata=True,       # 是否包含文档元数据
    flatten_structure=False,     # 是否扁平化结构
    custom_renderers=None        # 自定义渲染器配置
)
```

## JSON 输出格式

### 文档结构
```json
{
  \"title\": \"文档标题\",
  \"metadata\": {
    \"key\": \"value\"
  },
  \"content\": [
    {
      \"type\": \"heading\",
      \"content\": \"标题内容\",
      \"children\": [/* JSON格式的子元素 */],
      \"attributes\": {
        \"level\": 1,
        \"children_json\": [/* 详细的JSON结构数据 */],
        \"children_content\": \"原始Markdown文本内容\"
      }
    }
  ]
}
```

### 重要说明：children vs children_content

- **`children`**: 存放子元素的 JSON 对象数组，用于程序化处理
- **`children_json`**: 完整的 JSON 结构数据，保留所有细节
- **`children_content`**: 原始 Markdown 格式文本，可直接使用

### 元素类型

| 类型 | 说明 | 示例属性 |
|------|------|----------|
| `heading` | 标题 | `level`: 1-6 |
| `paragraph` | 段落 | - |
| `list` | 列表 | `list_type`: \"ordered\"/\"unordered\" |
| `table` | 表格 | `headers`, `alignments` |
| `code_block` | 代码块 | `language` |
| `blockquote` | 引用 | - |
| `image` | 图片 | `src`, `alt`, `title` |
| `link` | 链接 | `url`, `title` |

### 表格特殊结构
```json
{
  \"type\": \"table\",
  \"content\": null,
  \"children\": [
    {
      \"type\": \"table_row\",
      \"children\": [
        {
          \"type\": \"table_cell\",
          \"content\": \"单元格内容\"
        }
      ]
    }
  ],
  \"attributes\": {
    \"headers\": [\"列1\", \"列2\", \"列3\"],
    \"alignments\": [\"left\", \"center\", \"right\"]
  }
}
```

## 实际应用示例

### 示例 1：文档模板处理

```python
# 读取企业评估模板
with open('企业信贷评估模板.md', 'r', encoding='utf-8') as f:
    template_content = f.read()

# 转换为 JSON 进行结构化处理
json_data = markdown_to_json(template_content)

# 可以对 JSON 数据进行各种处理
# 例如：提取所有表格
tables = []
def extract_tables(elements):
    for element in elements:
        if element.get('type') == 'table':
            tables.append(element)
        # 递归处理嵌套内容
        if element.get('attributes', {}).get('children_content'):
            extract_tables(element['attributes']['children_content'])

extract_tables(json_data['content'])
print(f\"找到 {len(tables)} 个表格\")

# 转换回 Markdown
processed_markdown = json_to_markdown(json_data)
```

### 示例 2：内容重构

```python
def reorganize_document(markdown_text):
    \"\"\"重新组织文档结构\"\"\"
    
    # 转换为 JSON
    json_data = markdown_to_json(markdown_text)
    
    # 提取所有段落
    paragraphs = []
    # 提取所有表格  
    tables = []
    
    def extract_content(elements):
        for element in elements:
            if element.get('type') == 'paragraph':
                paragraphs.append(element['content'])
            elif element.get('type') == 'table':
                tables.append(element)
    
    extract_content(json_data['content'])
    
    # 重新构建文档
    new_content = []
    
    # 添加概述
    new_content.append({
        \"type\": \"heading\",
        \"content\": \"文档概述\",
        \"attributes\": {\"level\": 2}
    })
    
    new_content.append({
        \"type\": \"paragraph\",
        \"content\": f\"本文档包含 {len(paragraphs)} 个段落和 {len(tables)} 个表格。\"
    })
    
    # 添加原始内容
    new_content.extend(json_data['content'])
    
    # 更新 JSON 数据
    json_data['content'] = new_content
    
    # 转换回 Markdown
    return json_to_markdown(json_data)

# 使用示例
reorganized = reorganize_document(original_markdown)
```

### 示例 3：错误处理

```python
def safe_convert(markdown_text):
    \"\"\"安全的转换函数，包含错误处理\"\"\"
    
    try:
        # 转换为 JSON
        json_result = markdown_to_json(markdown_text)
        
        # 验证转换质量
        regenerated = json_to_markdown(json_result)
        
        # 计算相似度（简单字符数比较）
        similarity = len(regenerated) / len(markdown_text)
        
        if similarity < 0.8:
            print(f\"警告：转换质量较低，相似度: {similarity:.2%}\")
        
        return {
            'success': True,
            'json_data': json_result,
            'markdown_data': regenerated,
            'similarity': similarity
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'json_data': None,
            'markdown_data': None
        }

# 使用示例
result = safe_convert(markdown_content)
if result['success']:
    print(\"转换成功\")
    print(f\"质量评分: {result['similarity']:.2%}\")
else:
    print(f\"转换失败: {result['error']}\")
```

## 性能和限制

### 性能建议
- **大文档**：对于超过 10MB 的文档，建议分块处理
- **复杂表格**：包含大量行的表格可能需要更多处理时间
- **嵌套结构**：深度嵌套的标题结构会增加处理复杂度

### 当前限制
- **表格格式**：对齐符号格式可能略有标准化（功能不受影响）
- **HTML 标签**：部分复杂 HTML 可能无法完全保留
- **特殊字符**：某些 Unicode 字符在转换过程中可能需要特殊处理

### 质量保证
- **回转稳定性**：✅ 多次转换结果一致
- **内容完整性**：✅ 不丢失核心内容
- **结构保持**：✅ 保持文档层次结构
- **格式近似**：⚠️ 格式可能有轻微标准化

## 常见问题

### Q: 转换后的 Markdown 格式略有不同？
A: 这是正常的。转换器会对格式进行标准化，但不影响内容和功能。例如表格对齐符号从 `:-----` 变为 `:---`。

### Q: 如何处理大型文档？
A: 建议按章节分割处理，或使用流式处理方式。

### Q: 是否支持自定义 Markdown 语法？
A: 目前支持标准 Markdown 语法。自定义语法需要扩展解析器。

### Q: 转换是否真的无损？
A: 在内容和结构层面是无损的，在格式层面会有轻微标准化。回转稳定性已得到保证。

## 更新日志

### v1.0 (当前版本)
- ✅ 基础 Markdown 解析和渲染
- ✅ 表格支持及对齐处理  
- ✅ 修复重复标题问题
- ✅ 移除不必要缩进
- ✅ 实现回转稳定性
- ✅ 支持复杂文档结构

## 技术支持

如遇到问题，请检查：
1. 输入的 Markdown 语法是否标准
2. 是否有特殊字符或编码问题
3. 文档大小是否在合理范围内

更多技术细节请参考源代码中的类型定义和注释。