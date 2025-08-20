#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构化报告生成Agent
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import BaseAgent
from .section_agent_react import SectionAgentReAct
from .section_agent import SectionAgent
from ..config import settings
from ..tool.knowledge_retrieval import KnowledgeRetrievalTool
from ..converter import MarkdownConverter, ConversionRequest, MarkdownRenderer, MarkdownParser
from ..schema import AgentState
from ..logger import logger
from ..prompt import report_multi as prompts


class ReportGeneratorAgent(BaseAgent):
    """结构化报告生成协调器Agent"""
    
    def __init__(self, 
                 template_path: Optional[str] = None,
                 knowledge_base_path: str = "workdir/documents",
                 output_path: Optional[str] = None,
                 parallel_sections: bool = False,
                 max_concurrent: int = 3,
                 enable_polishing: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.template_path = Path(template_path) if template_path else None
        self.output_path = Path(output_path) if output_path else None
        self.knowledge_base_path = knowledge_base_path
        self.parallel_sections = parallel_sections
        self.max_concurrent = max_concurrent
        
        # 初始化工具
        self.knowledge_tool = KnowledgeRetrievalTool(knowledge_base_path)
        self.converter = MarkdownConverter()
        self.parser = MarkdownParser()
        self.render = MarkdownRenderer()
        
        # 报告模板和内容
        self.template_structure = None
        self.report_content = {}
        
        # 章节Agent管理
        self.section_agents: List[SectionAgent | SectionAgentReAct] = []
        self.completed_sections = set()
        self.active_agents: Dict[int, SectionAgentReAct] = {}
        
        # 内容质量控制
        self.enable_polishing = enable_polishing
        self.quality_check_passed = False
        self.polishing_completed = False
        
        # 设置Agent默认参数
        if not self.name:
            self.name = "report_coordinator"
        if not self.description:
            self.description = "协调多个SectionAgent生成结构化报告的智能Agent"
        
        # 设置系统提示
        self.system_prompt = prompts.SYSTEM_PROMPT
    
    async def initialize_from_template(self, template_content: str):
        """从模板内容初始化Agent"""
        try:
            # 如果是Markdown模板，转换为MD数据结构
            if template_content.strip().startswith('#'):
                # conversion_request = ConversionRequest(
                #     source_format="markdown",
                #     target_format="json",
                #     content=template_content,
                #     options=None
                # )
                conversion_result = self.parser.parse(template_content)
                self.template_structure = conversion_result
                # if conversion_result.success:
                #     self.template_structure = conversion_result.result
                # else:
                #     raise ValueError(f"模板转换失败: {conversion_result.error}")
            else:
                # 假设是JSON格式
                self.template_structure = json.loads(template_content)
            
            # 初始化报告内容结构
            self._initialize_report_structure()
            
            logger.info(f"模板初始化完成，共 {len(self.template_structure.content)} 个部分")
            
        except Exception as e:
            logger.error(f"模板初始化失败: {e}")
            raise
    
    def _initialize_report_structure(self):
        """初始化报告内容结构和章节Agents"""
        self.report_content = {
            "title": self.template_structure.title,
            "metadata": self.template_structure.metadata,
            "content": []
        }
        
        # 为每个模板部分创建章节结构和对应的Agent
        report_context = {
            "title": self.report_content["title"],
            "metadata": self.report_content["metadata"]
        }
        
        structure = ["table", "paragraph", "list"]
        
        section = {
            "heading": "",
            "level": 0,
            "id": 0,
            "content": "",
            "completed": False,
            "generated_content": ""
        }

        for element in self.template_structure.content:
            if element.type == 'heading':
                    
                self.report_content["content"].append(section)  
                
                if settings.section_agent_react:
                    section_agent = SectionAgentReAct(
                        # name=f"section_agent_{section['id']}",
                        section_info=section,
                        report_context=report_context,
                        knowledge_base_path=self.knowledge_base_path,
                        llm=self.llm
                    )
                else:
                    section_agent = SectionAgent(
                        name=f"section_agent_{section['id']}",
                        section_info=section,
                        report_context=report_context,
                        knowledge_base_path=self.knowledge_base_path,
                    )
                self.section_agents.append(section_agent)
                
                section = {
                    "heading": element.content,
                    "level": getattr(element, "level", 1),
                    "id": len(self.report_content["content"]),
                    "content": "",
                    "completed": False,
                    "generated_content": ""
                }
            
            elif element.type in structure:
                section["content"]+=self.render._render_element(element)

    async def step(self) -> str:
        """执行单步操作 - 协调章节生成"""
        try:
            # 检查是否有待处理的章节
            pending_sections = [
                section for section in self.report_content["content"]
                if not section.get("completed", False)
            ]
            
            if not pending_sections:
                # 所有章节已完成，进行内容润色和质量检查
                if self.enable_polishing and not self.polishing_completed:
                    return await self._polish_and_check_content()
                else:
                    # 润色完成，保存报告并终止
                    print(self.report_content)
                    await self._save_report()
                    logger.info("所有章节已完成，保存报告")
                    self.state = AgentState.FINISHED
                    return "报告生成完成"
            
            if self.parallel_sections:
                return await self._handle_parallel_generation()
            else:
                return await self._handle_sequential_generation()
                
        except Exception as e:
            logger.error(f"协调步骤执行失败: {e}")
            return f"执行失败: {str(e)}"
    
    async def _handle_sequential_generation(self) -> str:
        """处理串行章节生成"""
        def remove_fences(text: str) -> str:
            lines = text.splitlines()
            kept = [ln for ln in lines if not ln.strip().startswith("```")]
            return "\n".join(kept) 
        
        # 找到下一个待处理的章节
        for i, section in enumerate(self.report_content["content"]):
            if not section.get("completed", False):
                section_agent = self.section_agents[i]
                section_title = section.get("heading", f"第{i+1}章节")
                
                logger.info(f"开始生成章节: {section_title}")
                
                try:
                    # 运行章节Agent
                    generated_content = await section_agent.run_section_generation()
                    
                    if generated_content:
                        section["generated_content"] = remove_fences(generated_content)
                        section["completed"] = True
                        self.completed_sections.add(section["id"])
                        
                        progress = f"{len(self.completed_sections)}/{len(self.report_content['content'])}"
                        logger.info(f"章节'{section_title}'生成完成，进度: {progress}")
                        
                        return f"完成章节 '{section_title}' ({progress})"
                    else:
                        return f"正在生成章节 '{section_title}'"
                        
                except Exception as e:
                    logger.error(f"章节'{section_title}'生成失败: {e}")
                    section["generated_content"] = f"[生成失败] {str(e)}"
                    section["completed"] = True
                    return f"章节 '{section_title}' 生成失败: {str(e)}"
        
        return "所有章节处理完成"
    
    async def _handle_parallel_generation(self) -> str:
        """处理并行章节生成"""
        # 获取所有未完成的章节
        pending_sections = [
            (i, section) for i, section in enumerate(self.report_content["content"])
            if not section.get("completed", False)
        ]
        
        if not pending_sections:
            return "所有章节已完成"
        
        # 限制并发数量
        batch_size = min(len(pending_sections), self.max_concurrent)
        current_batch = pending_sections[:batch_size]
        
        logger.info(f"开始并行生成 {len(current_batch)} 个章节")
        
        # 创建并发任务
        tasks = []
        for section_idx, section in current_batch:
            section_agent = self.section_agents[section_idx]
            task = self._generate_section_with_agent(section_idx, section, section_agent)
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        completed_count = 0
        for i, result in enumerate(results):
            section_idx, section = current_batch[i]
            if isinstance(result, Exception):
                logger.error(f"章节并行生成失败: {result}")
                section["generated_content"] = f"[生成失败] {str(result)}"
            else:
                section["generated_content"] = result
            
            section["completed"] = True
            self.completed_sections.add(section["id"])
            completed_count += 1
        
        progress = f"{len(self.completed_sections)}/{len(self.report_content['content'])}"
        logger.info(f"并行生成完成 {completed_count} 个章节，总进度: {progress}")
        
        return f"并行完成 {completed_count} 个章节 ({progress})"
    
    async def _generate_section_with_agent(self, section_idx: int, section: Dict[str, Any], agent: SectionAgentReAct) -> str:
        """使用指定Agent生成章节内容"""
        try:
            section_title = section.get("content", f"章节{section_idx + 1}")
            logger.info(f"开始使用Agent生成章节: {section_title}")
            
            content = await agent.run_section_generation()
            return content if content else f"[空内容] {section_title}"
        except Exception as e:
            logger.error(f"Agent生成章节{section_idx + 1}失败: {e}")
            raise
    
    async def _polish_and_check_content(self) -> str:
        """润色和检查报告内容质量"""
        try:
            logger.info("开始对整体报告内容进行润色和质量检查")
            
            # 1. 内容质量评估
            quality_issues = await self._assess_content_quality()
            
            # 2. 内容润色
            if quality_issues:
                logger.info(f"发现 {len(quality_issues)} 个质量问题，开始润色")
                await self._polish_content(quality_issues)
            
            # 3. 一致性检查
            consistency_issues = await self._check_content_consistency()
            
            # 4. 修复一致性问题
            if consistency_issues:
                logger.info(f"发现 {len(consistency_issues)} 个一致性问题，开始修复")
                await self._fix_consistency_issues(consistency_issues)
            
            # 5. 最终质量验证
            final_quality = await self._final_quality_check()
            
            self.polishing_completed = True
            self.quality_check_passed = final_quality
            
            if final_quality:
                logger.info("✅ 内容润色和质量检查完成，报告质量良好")
                return "内容润色和质量检查完成"
            else:
                logger.warning("⚠️ 内容润色完成，但仍存在一些质量问题")
                return "内容润色完成（存在轻微质量问题）"
                
        except Exception as e:
            logger.error(f"内容润色和检查失败: {e}")
            self.polishing_completed = True
            return f"内容润色失败: {str(e)}"
    
    async def _assess_content_quality(self) -> List[Dict[str, Any]]:
        """评估内容质量"""
        quality_issues = []
        
        # 检查每个章节的内容质量
        for i, section in enumerate(self.report_content["content"]):
            if not section.get("generated_content"):
                continue
                
            section_title = section.get("content", f"章节{i+1}")
            content = section["generated_content"]
            
            # 构建质量评估提示
            quality_prompt = prompts.QUALITY_ASSESSMENT_PROMPT.format(
                section_title=section_title,
                content=content
            )
            
            self.update_memory("user", quality_prompt)
            
            try:
                response = await self.llm.ask(self.memory.messages)
                if response:
                    # 解析评估结果
                    import re
                    json_match = re.search(r'\{[^{}]*}', response, re.DOTALL)
                    if json_match:
                        evaluation = json.loads(json_match.group())
                        
                        # 如果质量分数低于阈值，记录为问题
                        if evaluation.get("overall_score", 5) < 4:
                            quality_issues.append({
                                "section_idx": i,
                                "section_title": section_title,
                                "evaluation": evaluation,
                                "content": content
                            })
                            
                self.update_memory("assistant", response)
                
            except Exception as e:
                logger.error(f"章节'{section_title}'质量评估失败: {e}")
        
        return quality_issues
    
    async def _polish_content(self, quality_issues: List[Dict[str, Any]]) -> None:
        """根据质量问题润色内容"""
        for issue_info in quality_issues:
            section_idx = issue_info["section_idx"]
            section_title = issue_info["section_title"]
            evaluation = issue_info["evaluation"]
            original_content = issue_info["content"]
            
            logger.info(f"正在润色章节: {section_title}")
            
            # 构建润色提示
            polish_prompt = prompts.POLISH_CONTENT_PROMPT.format(
                section_title=section_title,
                original_content=original_content,
                issues=', '.join(evaluation.get('issues', [])),
                suggestions=', '.join(evaluation.get('suggestions', []))
            )
            
            self.update_memory("user", polish_prompt)
            
            try:
                response = await self.llm.ask(self.memory.messages)
                if response and response.strip():
                    # 更新章节内容
                    polished_content = response.strip()
                    self.report_content["content"][section_idx]["generated_content"] = polished_content
                    
                    logger.info(f"章节'{section_title}'润色完成")
                    self.update_memory("assistant", polished_content)
                else:
                    logger.warning(f"章节'{section_title}'润色未获得有效响应")
                    
            except Exception as e:
                logger.error(f"章节'{section_title}'润色失败: {e}")
    
    async def _check_content_consistency(self) -> List[Dict[str, str]]:
        """检查内容一致性"""
        consistency_issues = []
        
        # 收集所有章节内容
        all_content = ""
        section_summaries = []
        
        for section in self.report_content["content"]:
            if section.get("generated_content"):
                section_title = section.get("content", "")
                content = section["generated_content"]
                all_content += f"\n\n## {section_title}\n{content}"
                section_summaries.append(f"- {section_title}: {content[:100]}...")
        
        # 构建一致性检查提示
        consistency_prompt = prompts.CONSISTENCY_CHECK_PROMPT.format(
            report_title=self.report_content.get('title', ''),
            section_summaries=chr(10).join(section_summaries),
            full_content=all_content
        )
        
        self.update_memory("user", consistency_prompt)
        
        try:
            response = await self.llm.ask(self.memory.messages)
            if response:
                # 解析一致性检查结果
                import re
                json_match = re.search(r'\{.*}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    consistency_issues = result.get("issues", [])
                    
                self.update_memory("assistant", response)
                
        except Exception as e:
            logger.error(f"一致性检查失败: {e}")
        
        return consistency_issues
    
    async def _fix_consistency_issues(self, consistency_issues: List[Dict[str, str]]) -> None:
        """修复一致性问题"""
        for issue in consistency_issues:
            issue_type = issue.get("type", "未知问题")
            description = issue.get("description", "")
            affected_sections = issue.get("affected_sections", [])
            suggestion = issue.get("suggestion", "")
            
            logger.info(f"正在修复一致性问题: {issue_type}")
            
            # 找到受影响的章节
            for section in self.report_content["content"]:
                section_title = section.get("content", "")
                if any(affected_section in section_title for affected_section in affected_sections):
                    
                    original_content = section.get("generated_content", "")
                    if not original_content:
                        continue
                    
                    # 构建修复提示
                    fix_prompt = prompts.FIX_CONSISTENCY_PROMPT.format(
                        section_title=section_title,
                        original_content=original_content,
                        issue_type=issue_type,
                        description=description,
                        suggestion=suggestion
                    )
                    
                    self.update_memory("user", fix_prompt)
                    
                    try:
                        response = await self.llm.ask(self.memory.messages)
                        if response and response.strip():
                            fixed_content = response.strip()
                            section["generated_content"] = fixed_content
                            logger.info(f"章节'{section_title}'一致性问题修复完成")
                            self.update_memory("assistant", fixed_content)
                        
                    except Exception as e:
                        logger.error(f"章节'{section_title}'一致性问题修复失败: {e}")
    
    async def _final_quality_check(self) -> bool:
        """最终质量检查"""
        try:
            # 构建最终报告内容
            full_report = f"# {self.report_content.get('title', '')}\\n"
            
            for section in self.report_content["content"]:
                section_title = section.get("content", "")
                content = section.get("generated_content", "")
                level_prefix = "#" * section.get("level", 1)
                full_report += f"\n{level_prefix} {section_title}\n\n{content}\n"
            
            # 最终质量评估提示
            final_check_prompt = prompts.FINAL_QUALITY_CHECK_PROMPT.format(
                full_report=full_report
            )
            
            self.update_memory("user", final_check_prompt)
            response = await self.llm.ask(self.memory.messages)
            
            if response:
                self.update_memory("assistant", response)
                # 简单判断是否通过
                return "PASS" in response.upper()
            else:
                return False
                
        except Exception as e:
            logger.error(f"最终质量检查失败: {e}")
            return False
    
    
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
                    "content": section["heading"],
                    "attributes": {"level": section.get("level", 1)},
                })
                
                #添加生成的表格
                '''
                if section.get("type") == "table" and section.get("generated_content"):
                    report_document["content"].append({
                        "type": "table",
                        "content": section["generated_content"],
                        "attributes": {"cols":len(section.get("headers",[])),
                                       "rows":len(section.get("rows",[]))}
                    })
                '''
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
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(report_document, f, ensure_ascii=False, indent=2)
                
                # 转换并保存Markdown格式
                conversion_request = ConversionRequest(
                    source_format="json",
                    target_format="markdown",
                    content=report_document,
                    options=None
                )
                conversion_result = self.converter.convert(conversion_request)
                
                if conversion_result.success:
                    md_path = self.output_path.with_suffix('.md')
                    with open(md_path, 'w', encoding='utf-8') as f:
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
        
        # 获取正在处理的章节
        active_sections = []
        for i, agent in enumerate(self.section_agents):
            if agent.state == AgentState.RUNNING and not self.report_content["content"][i].get("completed", False):
                active_sections.append(self.report_content["content"][i].get("content", f"章节{i+1}"))
        
        # 计算整体进度（包括润色阶段）
        base_progress = (completed_sections / total_sections * 80) if total_sections > 0 else 0
        if self.polishing_completed:
            base_progress += 20  # 润色完成额外加20%
        elif completed_sections == total_sections and self.enable_polishing:
            base_progress += 10  # 润色进行中加10%
        
        return {
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "progress_percentage": min(base_progress, 100),
            "active_sections": active_sections,
            "remaining_sections": total_sections - completed_sections,
            "mode": "parallel" if self.parallel_sections else "sequential",
            "polishing_enabled": self.enable_polishing,
            "polishing_completed": self.polishing_completed,
            "quality_check_passed": self.quality_check_passed,
            "current_phase": "polishing" if (completed_sections == total_sections and not self.polishing_completed and self.enable_polishing) else "generation"
        }
    
    async def run_with_template(self, template_path: str, output_path: Optional[str] = None) -> str:
        """使用指定模板运行Agent"""
        try:
            # 加载模板
            template_content = Path(template_path).read_text(encoding='utf-8')
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