#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器主模块
整合Agent、工具和转换器功能
"""

import asyncio
import argparse
from pathlib import Path
from typing import Optional

from app.agent.report_multi import ReportGeneratorAgent
from app.tool.knowledge_retrieval import KnowledgeRetrievalTool
from app.converter import MarkdownConverter, ConversionRequest
from app.logger import logger
from app.config import settings


class ReportGenerationSystem:
    """报告生成系统"""

    def __init__(self,
                 knowledge_base_path: str = "workdir/documents",
                 template_base_path: str = "workdir/template"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.template_base_path = Path(template_base_path)
        self.converter = MarkdownConverter()

        # 确保目录存在
        self.knowledge_base_path.mkdir(parents = True, exist_ok = True)
        self.template_base_path.mkdir(parents = True, exist_ok = True)

    async def generate_report(self,
                              template_name: str,
                              output_name: Optional[str] = None,
                              max_steps: int = 20) -> str:
        """生成报告"""
        try:
            # 查找模板文件
            template_path = self._find_template(template_name)
            if not template_path:
                raise FileNotFoundError(f"未找到模板文件: {template_name}")

            # 设置输出路径
            if not output_name:
                output_name = f"generated_report_{template_path.stem}"

            output_path = Path("workdir") / "output" / output_name
            output_path.parent.mkdir(parents = True, exist_ok = True)

            # 创建并配置Agent
            agent = ReportGeneratorAgent(
                name = f"report_generator_{template_path.stem}",
                description = f"基于模板 {template_path.name} 生成报告",
                template_path = str(template_path),
                knowledge_base_path = str(self.knowledge_base_path),
                output_path = str(output_path),
                max_steps = max_steps,
                parallel_sections = settings.parallel_sections,
                max_concurrent = settings.max_concurrent,
            )

            logger.info(f"开始生成报告，模板: {template_path.name}")
            logger.info(f"知识库路径: {self.knowledge_base_path}")
            logger.info(f"输出路径: {output_path}")

            # 运行Agent
            result = await agent.run_with_template(str(template_path), str(output_path))

            # 获取进度信息
            progress = agent.get_progress()

            success_message = f"""报告生成完成！

模板: {template_path.name}
进度: {progress['completed_sections']}/{progress['total_sections']} 
      ({progress['progress_percentage']:.1f}%)
输出: {output_path}.json, {output_path}.md

详细执行结果:
{result}"""

            logger.info("报告生成任务完成")
            return success_message

        except Exception as e:
            error_message = f"报告生成失败: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)

    def _find_template(self, template_name: str) -> Optional[Path]:
        """查找模板文件"""
        # 直接路径
        if Path(template_name).exists():
            return Path(template_name)

        # 在模板目录中查找
        possible_paths = [
            self.template_base_path / template_name,
            self.template_base_path / f"{template_name}.md",
            self.template_base_path / f"{template_name}.json"
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    async def convert_template(self,
                               template_path: str,
                               output_format: str = "json") -> str:
        """转换模板格式"""
        try:
            template_file = Path(template_path)
            if not template_file.exists():
                raise FileNotFoundError(f"模板文件不存在: {template_path}")

            # 读取模板内容
            template_content = template_file.read_text(encoding = 'utf-8')

            # 确定源格式
            source_format = "markdown" if template_content.strip().startswith('#') else "json"

            # 执行转换
            conversion_request = ConversionRequest(
                source_format = source_format,
                target_format = output_format,
                content = template_content
            )

            result = self.converter.convert(conversion_request)

            if result.success:
                # 保存转换结果
                output_path = template_file.with_suffix(f'.{output_format}')
                if output_format == "json":
                    import json
                    with open(output_path, 'w', encoding = 'utf-8') as f:
                        json.dump(result.result, f, ensure_ascii = False, indent = 2)
                else:
                    with open(output_path, 'w', encoding = 'utf-8') as f:
                        f.write(result.result)

                return f"模板转换成功: {output_path}"
            else:
                raise Exception(result.error)

        except Exception as e:
            error_message = f"模板转换失败: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)

    async def test_knowledge_base(self, query: str = "智能鞋垫") -> str:
        """测试知识库检索"""
        try:
            knowledge_tool = KnowledgeRetrievalTool(str(self.knowledge_base_path))
            result = await knowledge_tool.execute(query = query, threshold=settings.distance, top_k = settings.top_k)

            if result.error:
                return f"知识库检索测试失败: {result.error}"
            else:
                stats = knowledge_tool.get_statistics()
                return f"""知识库检索测试成功！

统计信息:
- 文档总数: {stats['total_documents']}
- 总大小: {stats['total_size']} 字符
- 平均大小: {stats['average_size']} 字符
- 文件类型: {stats['file_types']}

检索结果 (查询: "{query}"):
{result.output}"""

        except Exception as e:
            error_message = f"知识库测试失败: {str(e)}"
            logger.error(error_message)
            return error_message

    def list_templates(self) -> str:
        """列出可用模板"""
        templates = []

        if self.template_base_path.exists():
            for file_path in self.template_base_path.iterdir():
                if file_path.is_file() and file_path.suffix in ['.md', '.json']:
                    templates.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "type": file_path.suffix[1:]
                    })

        if not templates:
            return "未找到可用模板"

        result = "可用模板:\n"
        for i, template in enumerate(templates, 1):
            result += f"{i}. {template['name']} ({template['type']}, {template['size']} bytes)\n"

        return result


async def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description = "智能报告生成器")
    parser.add_argument("action", choices = ["generate", "convert", "test", "list"],
                        help = "执行的操作")
    parser.add_argument("--template", "-t", help = "模板名称或路径")
    parser.add_argument("--output", "-o", help = "输出文件名")
    parser.add_argument("--format", "-f", default = "json",
                        choices = ["json", "markdown"], help = "转换目标格式")
    parser.add_argument("--query", "-q", default = "智能鞋垫", help = "测试查询")
    parser.add_argument("--max-steps", type = int, default = 20, help = "最大执行步数")
    parser.add_argument("--knowledge-base", default = "workdir/documents",
                        help = "知识库路径")
    parser.add_argument("--template-base", default = "workdir/template",
                        help = "模板基础路径")

    args = parser.parse_args()

    # 创建报告生成系统
    system = ReportGenerationSystem(
        knowledge_base_path = args.knowledge_base,
        template_base_path = args.template_base
    )

    try:
        if args.action == "generate":
            if not args.template:
                print("错误: 生成报告需要指定模板 (--template)")
                return

            result = await system.generate_report(
                template_name = args.template,
                output_name = args.output,
                max_steps = args.max_steps
            )
            print(result)

        elif args.action == "convert":
            if not args.template:
                print("错误: 转换模板需要指定模板路径 (--template)")
                return

            result = await system.convert_template(
                template_path = args.template,
                output_format = args.format
            )
            print(result)

        elif args.action == "test":
            result = await system.test_knowledge_base(query = args.query)
            print(result)

        elif args.action == "list":
            result = system.list_templates()
            print(result)

    except Exception as e:
        print(f"执行失败: {e}")
        logger.error(f"命令执行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())