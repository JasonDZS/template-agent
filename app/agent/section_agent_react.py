#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ToolCallAgent的章节生成Agent
支持多次知识库检索直到完成章节内容
"""

from typing import Dict, Any, Optional
from pydantic import Field
from app.agent.toolcall import ToolCallAgent
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.tool import ToolCollection, Terminate
from app.prompt.section_agent import get_system_prompt, NEXT_STEP_PROMPT
from app.schema import AgentState
from app.logger import logger


class SectionAgentReAct(ToolCallAgent):
    """基于ToolCall的章节生成Agent，支持多次知识库检索"""
    
    # 添加Pydantic字段
    section_info: Dict[str, Any] = Field(default_factory=dict)
    report_context: Dict[str, Any] = Field(default_factory=dict)
    knowledge_base_path: str = Field(default="workdir/documents")
    generated_content: str = Field(default="")
    is_content_complete: bool = Field(default=False)
    
    def __init__(self, 
                 section_info: Dict[str, Any],
                 report_context: Dict[str, Any],
                 knowledge_base_path: str = "workdir/documents",
                 **kwargs):
        
        section_title = section_info.get("content", "未命名章节")
        section_level = section_info.get("level", 1)
        section_id = section_info.get("id", 0)
        report_title = report_context.get("title", "")
        
        # 设置基本信息
        name = f"section_{section_id}_{section_title[:10]}"
        description = f"生成章节'{section_title}'的专用Agent"
        
        # 初始化父类
        super().__init__(
            name=name,
            description=description,
            **kwargs
        )
        
        # 设置字段值
        self.section_info = section_info
        self.report_context = report_context  
        self.knowledge_base_path = knowledge_base_path
        self.generated_content = ""
        self.is_content_complete = False
        
        # 设置提示词
        self.system_prompt = get_system_prompt(
            section_title=section_title,
            report_title=report_title,
            section_level=section_level,
            section_id=section_id
        )
        self.next_step_prompt = NEXT_STEP_PROMPT
        
        # 初始化工具
        knowledge_tool = KnowledgeRetrievalTool(knowledge_base_path)
        terminate_tool = Terminate()
        
        self.available_tools = ToolCollection(
            knowledge_tool,
            terminate_tool
        )
        
        logger.info(f"初始化SectionAgent: {section_title}")
    
    async def execute_tool(self, command) -> str:
        """重写工具执行方法，处理内容生成逻辑"""
        result = await super().execute_tool(command)
        
        # 如果是知识检索工具的结果，记录检索信息
        if command.function.name == "knowledge_retrieval":
            logger.info(f"章节'{self.section_info.get('content', '')}'完成知识检索")
        
        # 如果是terminate工具，处理内容完成
        elif command.function.name == "terminate":
            self.is_content_complete = True
            logger.info(f"章节'{self.section_info.get('content', '')}'生成完成")
        
        return result
    
    async def think(self) -> bool:
        """重写思考方法，加入内容生成判断"""
        # 如果还没有生成最终内容，继续思考
        if not self.is_content_complete and self.state != AgentState.FINISHED:
            return await super().think()
        
        return False
    
    def get_final_content(self) -> str:
        """从对话历史中提取最终生成的内容"""
        if not self.memory or not self.memory.messages:
            return ""
        
        # 查找最后一个assistant消息中的内容
        for message in reversed(self.memory.messages):
            if message.role == "assistant" and message.content:
                content = message.content.strip()
                # 过滤掉工具调用相关的内容
                if (not content.startswith("我需要") and 
                    not content.startswith("让我") and
                    not content.startswith("现在我") and
                    len(content) > 50):  # 确保是实际的章节内容而不是简短的工具使用说明
                    return content
        
        return ""
    
    async def run_section_generation(self) -> str:
        """运行章节生成任务"""
        try:
            logger.info(f"开始生成章节: {self.section_info.get('content', '')}")
            
            # 运行Agent直到完成
            result = await self.run()
            
            # 获取生成的内容
            final_content = self.get_final_content()
            
            if final_content:
                self.generated_content = final_content
                logger.info(f"章节生成成功，内容长度: {len(final_content)}")
                return final_content
            else:
                logger.warning(f"章节生成完成但未找到有效内容")
                return f"[章节生成未完成] {self.section_info.get('content', '')}"
            
        except Exception as e:
            logger.error(f"章节生成失败: {e}")
            self.state = AgentState.ERROR
            raise
    
    def get_content(self) -> str:
        """获取生成的内容"""
        return self.generated_content
    
    def is_finished(self) -> bool:
        """检查是否完成"""
        return self.is_content_complete and self.state == AgentState.FINISHED