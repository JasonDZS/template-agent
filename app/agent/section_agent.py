#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
章节生成Agent
"""

from typing import Dict, Any, Optional
from app.agent.base import BaseAgent
from app.config import settings
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.schema import AgentState
from app.logger import logger


class SectionAgent(BaseAgent):
    """单个章节生成Agent"""
    
    def __init__(self, 
                 section_info: Dict[str, Any],
                 report_context: Dict[str, Any],
                 knowledge_base_path: str = "workdir/documents",
                 **kwargs):
        super().__init__(**kwargs)
        
        self.section_info = section_info
        self.report_context = report_context
        self.knowledge_base_path = knowledge_base_path
        
        # 初始化工具
        self.knowledge_tool = KnowledgeRetrievalTool(knowledge_base_path)
        
        # 生成的内容
        self.generated_content = ""
        self.is_completed = False
        
        # 设置Agent默认参数
        section_title = section_info.get("content", "未命名章节")
        if not self.description:
            self.description = f"生成章节'{section_title}'的专用Agent"
        
        # 设置系统提示
        self.system_prompt = f"""你是专门负责生成报告章节"{section_title}"的Agent。

报告背景信息：
- 报告标题：{report_context.get('title', '')}
- 当前章节：{section_title}（级别 {section_info.get('level', 1)}）
- 章节ID：{section_info.get('id', 0)}

你的任务：
1. 深入理解该章节在整个报告中的作用和定位
2. 使用knowledge_retrieval工具获取相关信息
3. 生成高质量、结构化的章节内容
4. 确保内容与章节标题高度相关且符合报告整体风格

内容要求：
- 如果是高级别标题（1-2级），提供概括性、战略性内容
- 如果是低级别标题（3级以上），提供具体详细的实施内容
- 内容长度适中（200-800字）
- 使用专业语言，确保准确性和可读性
- 必要时包含数据、案例或具体建议

完成条件：
- 生成完整的章节内容后使用terminate工具结束任务
"""
    
    async def step(self) -> str:
        """执行单步操作"""
        try:
            if self.is_completed:
                self.state = AgentState.FINISHED
                return f"章节'{self.section_info.get('content', '')}'已完成生成"
            
            # 构建检索查询
            section_title = self.section_info.get("content", "")
            report_title = self.report_context.get("title", "")
            query = f"{section_title} {report_title}"
            
            logger.info(f"开始生成章节: {section_title}")
            
            # 从知识库检索相关信息
            try:
                retrieval_result = await self.knowledge_tool.execute(
                    query=query,
                    top_k=settings.top_k,
                    threshold=settings.distance
                )
                
                if retrieval_result.error:
                    logger.warning(f"知识库检索失败: {retrieval_result.error}")
                    knowledge_context = "未获取到相关知识库信息，请基于常识和专业知识生成内容。"
                else:
                    knowledge_context = retrieval_result.output
            except Exception as e:
                logger.error(f"知识检索异常: {e}")
                knowledge_context = "知识库检索遇到问题，请基于专业判断生成内容。"
            
            # 构建内容生成提示
            content_prompt = f"""请为以下章节生成专业的内容：

章节标题：{section_title}
章节级别：{self.section_info.get('level', 1)}
报告标题：{report_title}

知识库相关信息：
{knowledge_context}

请基于以上信息，为该章节生成内容。要求：

1. 内容结构：
   - 如果是1-2级标题：提供该部分的概述、重要性和主要观点
   - 如果是3级以上标题：提供具体的实施细节、方法或案例

2. 内容质量：
   - 确保与章节标题高度相关
   - 内容准确、专业、有价值
   - 语言清晰易懂，逻辑性强
   - 长度适中（200-800字）

3. 格式要求：
   - 直接输出章节正文内容
   - 不要包含章节标题（会自动添加）
   - 可以使用子标题、列表等格式
   - 必要时可以包含具体的数据或建议

请开始生成该章节的内容："""
            
            self.update_memory("user", content_prompt)
            
            # 调用LLM生成内容
            response = await self.llm.ask(self.memory.messages)
            
            if response and response.strip():
                self.generated_content = response.strip()
                self.is_completed = True
                self.update_memory("assistant", self.generated_content)
                
                logger.info(f"章节'{section_title}'内容生成完成，长度: {len(self.generated_content)}")
                return f"章节'{section_title}'生成完成"
            else:
                logger.warning(f"LLM未返回有效内容for章节: {section_title}")
                return f"正在生成章节'{section_title}'的内容..."
                
        except Exception as e:
            logger.error(f"章节生成步骤执行失败: {e}")
            return f"章节生成失败: {str(e)}"
    
    def get_content(self) -> str:
        """获取生成的内容"""
        return self.generated_content
    
    def is_finished(self) -> bool:
        """检查是否完成"""
        return self.is_completed and self.state == AgentState.FINISHED
    
    async def run_section_generation(self) -> str:
        """运行章节生成任务"""
        try:
            # 运行Agent直到完成
            while not self.is_finished():
                result = await self.step()
                if self.state == AgentState.ERROR:
                    break
            
            return self.generated_content
            
        except Exception as e:
            logger.error(f"章节生成运行失败: {e}")
            self.state = AgentState.ERROR
            raise