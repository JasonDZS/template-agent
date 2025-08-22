# SectionAgent使用教程

## 概述

SectionAgent是一个专门用于生成报告章节内容的智能代理，它基于BaseAgent构建，通过结合知识库检索和LLM生成来创建高质量、结构化的章节内容。该代理特别适用于需要基于知识库信息生成专业报告章节的场景。

## 核心功能

- **知识库检索**：自动从指定知识库中检索相关信息
- **智能内容生成**：使用LLM模型生成高质量的章节内容
- **格式化输出**：支持指定输出格式，确保结果符合预期结构
- **上下文感知**：基于章节信息和报告上下文进行内容生成
- **异步执行**：支持异步操作，提高处理效率
- **错误处理**：内置完善的错误处理和日志记录机制

## 基本使用方法

### 1. 导入必要的模块

```python
from app.agent.section_agent import SectionAgent
import asyncio
```

### 2. 准备配置数据

#### 章节信息配置
```python
section_info = {
    "content": "（一）行业地位评估",          # 章节标题
    "level": 3,                  # 章节级别
    "id": 101                   # 章节ID
}
```

#### 报告上下文配置
```python
report_context = {
    "title": "某公司2024年度信贷评估报告",    # 报告标题
}
```

#### 输出格式定义（可选）
```python
output_format = """
1. 市场份额： [ ] % （数据来源： [ ] ）  
2. 竞争优势（勾选或补充）：  
   □ 技术领先	□ 成本优势	□ 品牌优势	□ 渠道优势	□ 政策支持  
3. 行业风险预警（列出 3 项主要风险及应对措施）：

| 序号 | 风险点描述 | 影响说明 | 应对/缓释措施 |
|------|------------|----------|---------------|
| 1 | [ ] | [ ] | [ ] |
| 2 | [ ] | [ ] | [ ] |
| 3 | [ ] | [ ] | [ ] |
"""
```

### 3. 创建和运行SectionAgent

```python
async def generate_section():
    # 创建SectionAgent实例
    agent = SectionAgent(
        section_info=section_info,
        report_context=report_context,
        knowledge_base_path="workdir/finance/documents",  # 知识库路径
        output_format=output_format  # 可选：输出格式
    )
    
    # 运行代理
    result = await agent.run()
    
    # 获取生成的内容
    if agent.is_finished():
        content = agent.get_content()
        print("生成的章节内容：")
        print(content)
        return content
    else:
        print("章节生成失败")
        return None

# 执行生成
content = asyncio.run(generate_section())
```

## 参数详细说明

### 构造函数参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `section_info` | Dict[str, Any] | 是 | - | 章节信息，包含标题、级别、ID等 |
| `report_context` | Dict[str, Any] | 是 | - | 报告上下文信息 |
| `knowledge_base_path` | str | 否 | "workdir/documents" | 知识库路径 |
| `output_format` | Optional[str] | 否 | None | 期望的输出格式模板 |
| `**kwargs` | - | 否 | - | 传递给BaseAgent的额外参数 |

### section_info详细字段

```python
section_info = {
    "content": "章节标题",      # 必需：章节的主要标题
    "level": 1,              # 必需：章节级别（1为一级标题，2为二级标题等）
    "id": 101,              # 必需：唯一标识符
    "description": "章节描述", # 可选：章节的详细描述
    "keywords": ["关键词1", "关键词2"]  # 可选：相关关键词列表
}
```

### report_context详细字段

```python
report_context = {
    "title": "报告标题",        # 必需：整体报告的标题
    "type": "信贷评估报告",     # 可选：报告类型
    "company": "公司名称",      # 可选：目标公司
    "industry": "行业分类",     # 可选：行业信息
    "year": "2024",           # 可选：报告年份
    "author": "分析师姓名"      # 可选：作者信息
}
```

## 完整使用示例

### 示例1：基础用法

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from app.agent.section_agent import SectionAgent
import asyncio

async def basic_example():
    """基础使用示例"""
    
    # 定义章节信息
    section_info = {
        "content": "行业分析",
        "level": 2,
        "id": 201
    }
    
    # 定义报告上下文
    report_context = {
        "title": "制造业企业信贷评估报告"
    }
    
    # 创建代理
    agent = SectionAgent(
        section_info=section_info,
        report_context=report_context
    )
    
    # 执行生成
    await agent.run()
    
    # 获取结果
    if agent.is_finished():
        content = agent.get_content()
        print("生成成功！")
        print("-" * 50)
        print(content)
        print("-" * 50)
    else:
        print("生成失败")

if __name__ == "__main__":
    asyncio.run(basic_example())
```

### 示例2：高级配置

```python
async def advanced_example():
    """高级配置示例"""
    
    # 详细的章节信息
    section_info = {
        "content": "财务状况分析",
        "level": 2,
        "id": 301,
        "description": "分析企业的资产负债情况、盈利能力和现金流状况",
        "keywords": ["资产负债表", "利润表", "现金流量表", "财务比率"]
    }
    
    # 详细的报告上下文
    report_context = {
        "title": "ABC制造公司2024年信贷评估报告",
        "type": "企业信贷评估",
        "company": "ABC制造有限公司",
        "industry": "精密制造业",
        "year": "2024",
        "author": "高级分析师"
    }
    
    # 自定义输出格式
    output_format = """
## {section_title}

### 财务概况
[企业整体财务状况概述]

### 资产负债分析
- **总资产规模**：[数据和分析]
- **负债结构**：[数据和分析] 
- **所有者权益**：[数据和分析]

### 盈利能力分析
- **营业收入**：[数据和分析]
- **净利润**：[数据和分析]
- **盈利能力指标**：[ROE、ROA等指标分析]

### 现金流分析
- **经营性现金流**：[数据和分析]
- **投资性现金流**：[数据和分析] 
- **筹资性现金流**：[数据和分析]

### 财务风险评估
[财务风险点识别和评估]

### 结论与建议
[基于分析得出的结论和建议]
"""
    
    # 创建代理，指定自定义知识库路径
    agent = SectionAgent(
        section_info=section_info,
        report_context=report_context,
        knowledge_base_path="data/financial_documents",  # 自定义知识库路径
        output_format=output_format
    )
    
    # 执行生成
    print("开始生成章节内容...")
    await agent.run()
    
    # 检查生成状态
    if agent.is_finished():
        content = agent.get_content()
        print(f"章节'{section_info['content']}'生成完成！")
        print(f"内容长度：{len(content)}字符")
        print("\n生成内容预览：")
        print(content[:500] + "..." if len(content) > 500 else content)
    else:
        print("章节生成未完成或失败")

if __name__ == "__main__":
    asyncio.run(advanced_example())
```

### 示例3：批量生成多个章节

```python
async def batch_generation_example():
    """批量生成多个章节的示例"""
    
    # 定义多个章节
    sections = [
        {
            "content": "公司概况",
            "level": 2,
            "id": 401
        },
        {
            "content": "行业分析", 
            "level": 2,
            "id": 402
        },
        {
            "content": "财务分析",
            "level": 2, 
            "id": 403
        },
        {
            "content": "风险评估",
            "level": 2,
            "id": 404
        }
    ]
    
    report_context = {
        "title": "综合企业评估报告",
        "company": "目标企业",
        "year": "2024"
    }
    
    generated_contents = {}
    
    for section_info in sections:
        print(f"\n正在生成章节：{section_info['content']}")
        
        # 创建代理
        agent = SectionAgent(
            section_info=section_info,
            report_context=report_context
        )
        
        # 生成内容
        await agent.run()
        
        if agent.is_finished():
            content = agent.get_content()
            generated_contents[section_info['content']] = content
            print(f"✓ 章节'{section_info['content']}'生成完成")
        else:
            print(f"✗ 章节'{section_info['content']}'生成失败")
    
    print(f"\n批量生成完成，成功生成{len(generated_contents)}个章节")
    return generated_contents

if __name__ == "__main__":
    contents = asyncio.run(batch_generation_example())
```

## 工作流程详解

### 1. 初始化阶段
```python
agent = SectionAgent(section_info, report_context, ...)
```
- 解析章节信息和报告上下文
- 初始化知识检索工具
- 构建系统提示词
- 设置代理基本属性

### 2. 执行阶段
```python
await agent.run()
```
- **查询构建**：基于章节标题和报告标题构建检索查询
- **知识检索**：从知识库中检索相关信息
- **提示构建**：将检索结果整合到生成提示中
- **内容生成**：调用LLM生成章节内容
- **后处理**：清理和格式化输出内容

### 3. 结果获取
```python
content = agent.get_content()
```
- 检查生成状态
- 获取最终生成的内容

## 核心方法说明

### `async step() -> str`
执行单个生成步骤，包括：
- 知识库检索
- LLM内容生成
- 状态更新

### `get_content() -> str`
获取生成的章节内容

### `is_finished() -> bool`
检查章节生成是否完成

## 配置和调优

### 知识检索参数调优
知识检索行为受以下配置参数影响（在`app/config.py`中设置）：

```python
# 设置检索参数
settings.top_k = 5        # 检索返回的文档数量
settings.distance = 0.7   # 相似度阈值
```

### LLM模型兼容性
代理对不同LLM模型进行了优化：

```python
# Qwen模型特殊处理
if self.llm.model in ["Qwen/Qwen3-4B", "Qwen/Qwen3-32B"]:
    # 禁用思考标签
    knowledge_prompt += "\no_think<think>\n\n</think>"
```

### 自定义知识库路径
```python
agent = SectionAgent(
    section_info=section_info,
    report_context=report_context,
    knowledge_base_path="custom/knowledge/path"  # 自定义路径
)
```

## 错误处理和调试

### 常见错误类型

1. **知识库检索失败**
   - 原因：知识库路径不存在或权限不足
   - 处理：代理会使用默认知识生成内容

2. **LLM调用失败**
   - 原因：网络问题或API限制
   - 处理：检查网络连接和API配置

3. **内容生成为空**
   - 原因：提示词不当或模型响应异常
   - 处理：检查提示词格式和模型状态

### 调试技巧

1. **启用详细日志**：
```python
import logging
logging.basicConfig(level=logging.INFO)
```

2. **检查代理状态**：
```python
print(f"代理状态: {agent.state}")
print(f"生成完成: {agent.is_finished()}")
```

3. **查看内存消息**：
```python
for msg in agent.memory.messages:
    print(f"{msg.role}: {msg.content[:100]}...")
```

## 最佳实践

### 1. 章节信息设计
- 使用清晰、具体的章节标题
- 合理设置章节级别
- 提供相关的关键词和描述

### 2. 知识库管理
- 保持知识库文档的更新和相关性
- 使用结构化的文档格式
- 定期清理过时信息

### 3. 输出格式设计
- 使用清晰的章节结构
- 包含必要的子标题和格式化元素
- 考虑后续处理的需求

### 4. 性能优化
- 合理设置检索参数`top_k`和`distance`
- 对于批量生成，考虑并发控制
- 监控内存使用情况

## 扩展和定制

### 继承SectionAgent
```python
class CustomSectionAgent(SectionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 自定义初始化
    
    async def step(self) -> str:
        # 自定义生成逻辑
        return await super().step()
```

### 添加自定义工具
```python
from app.tool.custom_tool import CustomTool

class EnhancedSectionAgent(SectionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_tool = CustomTool()
    
    async def step(self) -> str:
        # 使用自定义工具
        custom_result = await self.custom_tool.execute(query)
        # 整合到生成流程中
        return await super().step()
```

## 完整工作示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from app.agent.section_agent import SectionAgent

async def complete_workflow_example():
    """完整的工作流程示例"""
    
    # 第一步：准备数据
    section_info = {
        "content": "市场竞争分析",
        "level": 2,
        "id": 501,
        "description": "分析目标企业在行业中的竞争地位和优劣势",
        "keywords": ["市场份额", "竞争对手", "SWOT分析", "行业地位"]
    }
    
    report_context = {
        "title": "制造业龙头企业投资价值分析报告",
        "type": "投资分析报告",
        "company": "某制造业龙头企业",
        "industry": "高端制造",
        "year": "2024"
    }
    
    custom_format = """
## {section_title}

### 市场环境概述
[行业整体市场环境分析]

### 竞争格局分析
#### 主要竞争对手
[竞争对手分析]

#### 市场份额对比
[市场份额数据和分析]

### 企业竞争优势
[目标企业的核心竞争优势]

### 企业竞争劣势
[需要改善的方面]

### SWOT分析
- **优势(Strengths)**：[列出主要优势]
- **劣势(Weaknesses)**：[列出主要劣势] 
- **机会(Opportunities)**：[列出市场机会]
- **威胁(Threats)**：[列出潜在威胁]

### 竞争策略建议
[基于分析提出的策略建议]
"""
    
    try:
        # 第二步：创建代理
        print("创建SectionAgent实例...")
        agent = SectionAgent(
            section_info=section_info,
            report_context=report_context,
            knowledge_base_path="workdir/documents",
            output_format=custom_format
        )
        
        # 第三步：执行生成
        print("开始执行章节生成...")
        start_time = asyncio.get_event_loop().time()
        
        await agent.run()
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        # 第四步：获取和处理结果
        if agent.is_finished():
            content = agent.get_content()
            
            print(f"✓ 章节生成成功！")
            print(f"执行时间: {execution_time:.2f}秒")
            print(f"内容长度: {len(content)}字符")
            print(f"内容字数: {len(content.replace(' ', '').replace('\n', ''))}字")
            
            # 保存结果到文件
            output_file = f"output_{section_info['id']}_{section_info['content']}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ 结果已保存到: {output_file}")
            
            # 显示内容预览
            print("\n" + "="*60)
            print("内容预览:")
            print("="*60)
            print(content[:1000] + "..." if len(content) > 1000 else content)
            print("="*60)
            
            return content
            
        else:
            print("✗ 章节生成失败或未完成")
            return None
            
    except Exception as e:
        print(f"✗ 生成过程中出现异常: {e}")
        return None

if __name__ == "__main__":
    result = asyncio.run(complete_workflow_example())
```

通过这个详细的教程，你应该能够熟练使用SectionAgent进行各种章节内容生成任务。记住要根据具体需求调整参数配置，并充分利用知识库检索功能来提高生成内容的质量和相关性。