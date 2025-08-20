#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库检索工具 - 使用ChromaDB和OpenAI Compatible API
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.types import Collection
from openai import OpenAI, embeddings

from .base_tool import BaseTool, ToolResult
from ..logger import logger
from ..config import settings


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """使用OpenAI Compatible API的embedding函数"""
    
    client = OpenAI(
        api_key=settings.llm_settings.api_key,
        base_url=settings.llm_settings.base_url
    )
    model = settings.EMBEDDING_MODEL
    
    def __call__(self, input: Documents) -> Embeddings:
        """生成文本嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=input,
                dimensions=settings.EMBEDDING_DIM
            )
            logger.info(f"Embedding Shape: {len(response.data[0].embedding)}")
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            # 返回零向量作为fallback
            return [[0.0] * settings.EMBEDDING_DIM for _ in input]


class KnowledgeRetrievalTool(BaseTool):
    """知识库检索工具，基于ChromaDB和OpenAI Compatible API的向量化检索"""
    
    name: str = "knowledge_retrieval"
    description: str = "从知识库中检索相关文档和信息"
    knowledge_base_path: str = "workdir/documents"
    collection_name: str = "documents"
    
    # Optional fields that will be set during initialization
    client: Optional[Any] = None
    collection: Optional[Any] = None
    openai_client: Optional[Any] = None
    embedding_function: Optional[Any] = None
    documents: List[Dict[str, Any]] = []
    
    def __init__(self, knowledge_base_path: str = "workdir/documents", collection_name: str = "documents"):
        super().__init__(knowledge_base_path=knowledge_base_path, collection_name=collection_name)
        self.knowledge_base_path = Path(knowledge_base_path)
        self.collection_name = collection_name
        self.client = None
        self.collection: Optional[Collection] = None
        self.openai_client = None
        self.documents = []
        
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询的文本"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回最相关的文档数量",
                    "default": 3
                },
                "threshold": {
                    "type": "number",
                    "description": "相似度阈值（0-1之间）",
                    "default": 0.3
                }
            },
            "required": ["query"]
        }
        
        # 初始化OpenAI客户端
        self.openai_client = OpenAI(
            api_key=settings.llm_settings.api_key,
            base_url=settings.llm_settings.base_url
        )
        
        # 创建embedding函数
        self.embedding_function = OpenAIEmbeddingFunction(
            self.openai_client, 
            settings.EMBEDDING_MODEL
        )
        
        # 初始化知识库
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """初始化知识库"""
        try:
            logger.info(f"初始化知识库，路径: {self.knowledge_base_path}")
            
            # 初始化ChromaDB客户端
            chroma_db_path = Path("workdir") / "chroma_db"
            chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(chroma_db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 获取或创建集合
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"加载已存在的集合: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "文档知识库"},
                    embedding_function=self.embedding_function
                )
                logger.info(f"创建新集合: {self.collection_name}")
            
            # 加载文档
            self._load_documents()
            
            # 构建向量索引
            if self.documents:
                self._build_index()
                logger.info(f"知识库初始化完成，共加载 {len(self.documents)} 个文档")
            else:
                logger.warning("未找到任何文档")
                
        except Exception as e:
            logger.error(f"知识库初始化失败: {e}")
            raise
    
    def _load_documents(self):
        """从文档文件夹加载文档"""
        self.documents = []
        
        if not self.knowledge_base_path.exists():
            logger.warning(f"知识库路径不存在: {self.knowledge_base_path}")
            return
        
        # 支持的文档格式
        supported_extensions = {'.md', '.txt', '.json'}
        
        for file_path in self.knowledge_base_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    content = self._load_file_content(file_path)
                    if content.strip():  # 只添加非空内容
                        doc_id = self._generate_doc_id(file_path)
                        doc = {
                            "id": doc_id,
                            "path": str(file_path),
                            "filename": file_path.name,
                            "content": content,
                            "type": file_path.suffix.lower(),
                            "size": len(content)
                        }
                        self.documents.append(doc)
                        logger.debug(f"加载文档: {file_path.name}")
                        
                except Exception as e:
                    logger.warning(f"加载文档失败 {file_path}: {e}")
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """生成文档ID"""
        # 使用文件路径和修改时间生成唯一ID
        stat = file_path.stat()
        content = f"{file_path}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_file_content(self, file_path: Path) -> str:
        """加载文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                    # 将JSON转换为可搜索的文本
                    return self._json_to_searchable_text(data)
                else:
                    return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gb2312') as f:
                    if file_path.suffix.lower() == '.json':
                        data = json.load(f)
                        return self._json_to_searchable_text(data)
                    else:
                        return f.read()
            except Exception:
                logger.warning(f"无法读取文件 {file_path}，跳过")
                return ""
    
    def _json_to_searchable_text(self, data: Any) -> str:
        """将JSON数据转换为可搜索的文本"""
        if isinstance(data, dict):
            texts = []
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    texts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    texts.append(f"{key}: {self._json_to_searchable_text(value)}")
            return " ".join(texts)
        elif isinstance(data, list):
            return " ".join(self._json_to_searchable_text(item) for item in data)
        else:
            return str(data)
    
    def _build_index(self):
        """构建ChromaDB向量索引"""
        try:
            logger.info("构建ChromaDB向量索引...")
            
            # 获取当前集合中的文档数量
            existing_count = self.collection.count()
            
            # 准备要添加的文档
            new_documents = []
            ids_to_add = []
            contents_to_add = []
            metadatas_to_add = []
            
            for doc in self.documents:
                doc_id = doc["id"]
                
                # 检查文档是否已存在
                try:
                    existing_docs = self.collection.get(ids=[doc_id])
                    if existing_docs["ids"]:
                        continue  # 文档已存在，跳过
                except Exception:
                    pass  # 文档不存在，需要添加
                
                ids_to_add.append(doc_id)
                contents_to_add.append(doc["content"])
                metadatas_to_add.append({
                    "filename": doc["filename"],
                    "path": doc["path"],
                    "type": doc["type"],
                    "size": doc["size"]
                })
                new_documents.append(doc)
            
            if ids_to_add:
                logger.info(f"添加 {len(ids_to_add)} 个新文档到ChromaDB...")
                
                # 分批处理以避免内存问题
                batch_size = 64
                for i in range(0, len(ids_to_add), batch_size):
                    batch_ids = ids_to_add[i:i+batch_size]
                    batch_contents = contents_to_add[i:i+batch_size]
                    batch_metadatas = metadatas_to_add[i:i+batch_size]
                    
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_contents,
                        metadatas=batch_metadatas
                    )
                
                logger.info(f"成功添加 {len(ids_to_add)} 个文档")
            else:
                logger.info("所有文档都已存在于ChromaDB中")
            
            total_count = self.collection.count()
            logger.info(f"ChromaDB索引构建完成，总文档数: {total_count}")
            
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            raise
    
    async def execute(self, query: str, top_k: int = 3, threshold: float = 0.01, **kwargs) -> ToolResult:
        """执行知识库检索"""
        try:
            if not self.collection:
                return ToolResult(
                    error="知识库未初始化",
                    output="请确保知识库正确初始化"
                )
            
            logger.info(f"检索查询: {query}")
            
            # 执行向量检索
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            logger.info(f"检索到 {len(results)} 个结果")
#            logger.info(results)

            if not results["ids"] or not results["ids"][0]:
                return ToolResult(
                    output="未找到相关文档，请尝试调整查询内容",
                    error=None
                )
            
            # 处理结果
            formatted_results = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0], 
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # ChromaDB使用距离，距离越小相似度越高
                # 转换为相似度分数 (0-1)

                if distance <= threshold:
                    result = {
                        "rank": i + 1,
                        "score": float(distance),
                        "filename": metadata.get("filename", "unknown"),
                        "path": metadata.get("path", ""),
                        "content_preview": document[:300] + "..." if len(document) > 300 else document,
                        "content": document,
                        "type": metadata.get("type", ""),
                        "size": metadata.get("size", 0)
                    }
                    formatted_results.append(result)
            
            if not formatted_results:
                return ToolResult(
                    output=f"未找到相似度超过 {threshold} 的相关文档，请尝试降低相似度阈值",
                    error=None
                )
            
            # 格式化输出
            output_text = f"找到 {len(formatted_results)} 个相关文档:\n\n"
            for result in formatted_results:
                output_text += f"**{result['rank']}. {result['filename']}** (相似度: {result['score']:.3f})\n"
                output_text += f"类型: {result['type']}, 大小: {result['size']} 字符\n"
                output_text += f"预览: {result['content_preview']}\n"
                output_text += f"路径: {result['path']}\n\n"
            
            return ToolResult(
                output=output_text,
                system=json.dumps(formatted_results, ensure_ascii=False, indent=2)
            )
            
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return ToolResult(
                error=f"检索失败: {str(e)}",
                output="请检查知识库是否正确初始化"
            )
    
    def refresh_knowledge_base(self):
        """刷新知识库"""
        logger.info("刷新知识库...")
        try:
            # 重新加载文档
            self._load_documents()
            
            if self.documents:
                # 重建索引
                self._build_index()
                logger.info(f"知识库刷新完成，共 {len(self.documents)} 个文档")
            else:
                logger.warning("刷新后未找到任何文档")
                
        except Exception as e:
            logger.error(f"刷新知识库失败: {e}")
    
    def get_document_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """根据路径获取文档内容"""
        for doc in self.documents:
            if doc["path"] == path or doc["filename"] == path:
                return doc
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        if not self.collection:
            return {"total_documents": 0}
        
        try:
            total_count = self.collection.count()
            
            stats = {
                "total_documents": total_count,
                "collection_name": self.collection_name,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_provider": settings.EMBEDDING_PROVIDER
            }
            
            if self.documents:
                stats.update({
                    "total_size": sum(doc["size"] for doc in self.documents),
                    "file_types": {},
                    "average_size": 0
                })
                
                # 统计文件类型
                for doc in self.documents:
                    file_type = doc["type"]
                    if file_type not in stats["file_types"]:
                        stats["file_types"][file_type] = 0
                    stats["file_types"][file_type] += 1
                
                if stats["total_size"] > 0:
                    stats["average_size"] = stats["total_size"] // len(self.documents)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """删除集合（用于重置知识库）"""
        try:
            if self.client and self.collection_name:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"已删除集合: {self.collection_name}")
        except Exception as e:
            logger.warning(f"删除集合失败: {e}")