#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构化报告生成Agent
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from app.agent.base import BaseAgent
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.converter import MarkdownConverter, ConversionRequest
from app.schema import AgentState
from app.logger import logger
from app.prompt import report_single as prompts


class ReportGeneratorAgentSingle(BaseAgent):
    """结构化报告生成Agent，基于单智能体实现"""

    def __init__(self,
                 template_path: Optional[str] = None,
                 knowledge_base_path: str = "workdir/documents",
                 output_path: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.template_path = Path(template_path) if template_path else None
        self.output_path = Path(output_path) if output_path else None
        self.knowledge_base_path = knowledge_base_path

        # 初始化工具
        self.knowledge_tool = KnowledgeRetrievalTool(knowledge_base_path)
        self.converter = MarkdownConverter()

        # 报告模板和内容
        self.template_structure = None
        self.report_content = {}
        self.current_section = None
        self.completed_sections = set()

        # 设置Agent默认参数
        if not self.name:
            self.name = "report_generator"
        if not self.description:
            self.description = "生成结构化报告的智能Agent"

        # 设置系统提示
        self.system_prompt = prompts.SYSTEM_PROMPT

        self.next_step_prompt = prompts.NEXT_STEP_PROMPT

    async def initialize_from_template(self, template_content: str):
        """从模板内容初始化Agent"""
        try:
            # 如果是Markdown模板，转换为JSON结构
            if template_content.strip().startswith('#'):
                conversion_request = ConversionRequest(
                    source_format = "markdown",
                    target_format = "json",
                    content = template_content
                )
                conversion_result = self.converter.convert(conversion_request)

                if conversion_result.success:
                    self.template_structure = conversion_result.result
                else:
                    raise ValueError(f"模板转换失败: {conversion_result.error}")
            else:
                # 假设是JSON格式
                self.template_structure = json.loads(template_content)

            # 初始化报告内容结构
            self._initialize_report_structure()

            logger.info(f"模板初始化完成，共 {len(self.template_structure.get('content', []))} 个部分")

        except Exception as e:
            logger.error(f"模板初始化失败: {e}")
            raise

    def _initialize_report_structure(self):
        """初始化报告内容结构"""
        self.report_content = {
            "title": self.template_structure.get("title", "报告"),
            "metadata": self.template_structure.get("metadata", {}),
            "content": []
        }

        # 为每个模板部分创建空的内容结构
        for element in self.template_structure.get("content", []):
            if element.get("type") == "heading":
                section = {
                    "type": "heading",
                    "level": element.get("level", 1),
                    "content": element.get("content", ""),
                    "id": len(self.report_content["content"]),
                    "completed": False,
                    "generated_content": ""
                }
                self.report_content["content"].append(section)

    async def step(self) -> str:
        """执行单步操作"""
        try:
            # 检查是否有待处理的部分
            pending_sections = [
                section for section in self.report_content["content"]
                if not section.get("completed", False)
            ]

            if not pending_sections:
                # 所有部分已完成，保存报告并终止
                await self._save_report()
                logger.info("所有部分已完成，保存报告")
                self.state = AgentState.FINISHED
                return "报告生成完成"

            # 选择下一个要处理的部分
            current_section = pending_sections[0]
            self.current_section = current_section

            section_title = current_section.get("content", f"第{current_section['id'] + 1}部分")
            logger.info(f"开始处理部分: {section_title}")

            # 构建当前步骤的提示
            progress_info = f"{len(self.completed_sections)}/{len(self.report_content['content'])}"
            pending_info = [s.get("content", f"部分{s['id'] + 1}") for s in pending_sections[:3]]

            step_prompt = self.next_step_prompt.format(
                progress = progress_info,
                pending_sections = ", ".join(pending_info)
            )

            # 更新系统消息
            self.update_memory("system", f"{self.system_prompt}\n\n{step_prompt}")

            # 为当前部分生成内容
            content_result = await self._generate_section_content(current_section)

            if content_result:
                current_section["generated_content"] = content_result
                current_section["completed"] = True
                self.completed_sections.add(current_section["id"])

                return f"完成部分 '{section_title}': {content_result[:100]}..."
            else:
                return f"正在处理部分 '{section_title}'"

        except Exception as e:
            logger.error(f"步骤执行失败: {e}")
            return f"执行失败: {str(e)}"

    async def _generate_section_content(self, section: Dict[str, Any]) -> str:
        """为指定部分生成内容"""
        section_title = section.get("content", "")
        section_level = section.get("level", 1)

        # 构建检索查询
        query = f"{section_title} {self.report_content.get('title', '')}"

        try:
            # 从知识库检索相关信息
            retrieval_result = await self.knowledge_tool.execute(
                query = query,
                top_k = 3,
                threshold = 0.01
            )

            if retrieval_result.error:
                logger.warning(f"知识库检索失败: {retrieval_result.error}")
                knowledge_context = "未获取到相关知识库信息"
            else:
                knowledge_context = retrieval_result.output

            # 构建内容生成提示
            content_prompt = prompts.CONTENT_GENERATION_PROMPT.format(
                report_title=self.report_content.get('title', ''),
                section_title=section_title,
                section_level=section_level,
                knowledge_context=knowledge_context
            )

            self.update_memory("user", content_prompt)

            # 调用LLM生成内容
            response = await self.llm.ask(self.memory.messages)

            if response:
                generated_content = response.strip()
                self.update_memory("assistant", generated_content)
                return generated_content
            else:
                logger.warning(f"LLM未返回有效内容")
                return f"[待完善] {section_title}相关内容"

        except Exception as e:
            logger.error(f"内容生成失败: {e}")
            return f"[生成失败] {section_title}: {str(e)}"

    async def _save_report(self):
        """保存生成的报告"""
        try:
            # 构建完整的报告结构
            report_document = {
                "title": self.report_content["title"],
                "metadata": self.report_content["metadata"],
                "content": []
            }

            # 添加生成的内容
            for section in self.report_content["content"]:
                # 添加标题
                report_document["content"].append({
                    "type": "heading",
                    "level": section["level"],
                    "content": section["content"],
                    "attributes": {"level": section["level"]}
                })

                # 添加生成的内容作为段落
                if section.get("generated_content"):
                    report_document["content"].append({
                        "type": "paragraph",
                        "content": section["generated_content"],
                        "attributes": {}
                    })

            # 保存JSON格式
            if self.output_path:
                json_path = self.output_path.with_suffix('.json')
                with open(json_path, 'w', encoding = 'utf-8') as f:
                    json.dump(report_document, f, ensure_ascii = False, indent = 2)

                # 转换并保存Markdown格式
                conversion_request = ConversionRequest(
                    source_format = "json",
                    target_format = "markdown",
                    content = report_document
                )
                conversion_result = self.converter.convert(conversion_request)

                if conversion_result.success:
                    md_path = self.output_path.with_suffix('.md')
                    with open(md_path, 'w', encoding = 'utf-8') as f:
                        f.write(conversion_result.result)

                    logger.info(f"报告已保存: {json_path}, {md_path}")
                else:
                    logger.warning(f"Markdown转换失败: {conversion_result.error}")
            else:
                logger.info("未指定输出路径，报告未保存到文件")

        except Exception as e:
            logger.error(f"报告保存失败: {e}")

    def get_progress(self) -> Dict[str, Any]:
        """获取生成进度"""
        total_sections = len(self.report_content.get("content", []))
        completed_sections = len(self.completed_sections)

        return {
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "progress_percentage": (completed_sections / total_sections * 100) if total_sections > 0 else 0,
            "current_section": self.current_section.get("content", "") if self.current_section else None,
            "remaining_sections": total_sections - completed_sections
        }

    async def run_with_template(self, template_path: str, output_path: Optional[str] = None) -> str:
        """使用指定模板运行Agent"""
        try:
            # 加载模板
            template_content = Path(template_path).read_text(encoding = 'utf-8')
            await self.initialize_from_template(template_content)

            # 设置输出路径
            if output_path:
                self.output_path = Path(output_path)

            # 运行Agent
            result = await self.run()

            return result

        except Exception as e:
            logger.error(f"模板运行失败: {e}")
            raise