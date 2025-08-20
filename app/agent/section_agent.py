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
from app.prompt import report_multi as prompts

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
        section_title = section_info.get("heading", "未命名章节")
        if not self.description:
            self.description = f"生成章节'{section_title}'的专用Agent"
        
        # 设置系统提示
        self.system_prompt = prompts.FULLFILL_SYSTEM_PROMPT
        self.update_memory("system", self.system_prompt)
    
    async def step(self) -> str:
        """执行单步操作"""
        try:
            if self.is_completed:
                self.state = AgentState.FINISHED
                return f"章节'{self.section_info.get('heading', '')}'已完成生成"
            
                
            
            # 构建检索查询
            section_title = self.section_info.get("heading", "")
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
                    knowledge_context = retrieval_result.system
            except Exception as e:
                logger.error(f"知识检索异常: {e}")
                knowledge_context = "知识库检索遇到问题，请基于专业判断生成内容。"
            
            # 构建内容生成提示
            content_prompt = prompts.FULLFILL_CONTENT_PROMPT.format(section_content=self.section_info.get("content"))
            knowledge_prompt = prompts.KNOWLEDGE_CONTEXT_PROMPT.format(knowledge_context=knowledge_context)
            
            self.update_memory("user", content_prompt)
            self.update_memory("user", knowledge_prompt)
            
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
    
    def remove_fences(text: str) -> str:
        lines = text.splitlines()
        kept = [ln for ln in lines if not ln.strip().startswith("```")]
        return "\n".join(kept)    
    
    async def run_section_generation(self) -> str:
        """运行章节生成任务"""

        try:
            # 运行Agent直到完成
            while not self.is_finished():
                
                if self.section_info.get("content") != "":
                    result = await self.step()
                else:
                    self.generated_content = "\n"
                    self.is_completed = True
                    self.state = AgentState.FINISHED
                
                if self.state == AgentState.ERROR:
                    break
            return self.generated_content
            
        except Exception as e:
            logger.error(f"章节生成运行失败: {e}")
            self.state = AgentState.ERROR
            raise