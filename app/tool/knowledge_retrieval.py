#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base Retrieval Tool - Using ChromaDB and OpenAI Compatible API.

This module provides a knowledge retrieval tool that uses ChromaDB for vector storage
and OpenAI Compatible API for text embeddings to perform semantic search across
document collections.
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
    """
    Embedding function using OpenAI Compatible API.
    
    This class implements ChromaDB's EmbeddingFunction interface to generate
    text embeddings using OpenAI Compatible API endpoints.
    
    Attributes:
        client: OpenAI client instance for API calls.
        model: The embedding model to use.
    """
    
    client = OpenAI(
        api_key=settings.llm_settings.api_key,
        base_url=settings.llm_settings.base_url
    )
    model = settings.EMBEDDING_MODEL
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate text embedding vectors.
        
        Args:
            input (Documents): List of text documents to embed.
            
        Returns:
            Embeddings: List of embedding vectors for the input documents.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=input,
                dimensions=settings.EMBEDDING_DIM
            )
            logger.info(f"Embedding Shape: {len(response.data[0].embedding)}")
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Failed to generate embedding vectors: {e}")
            # Return zero vectors as fallback
            return [[0.0] * settings.EMBEDDING_DIM for _ in input]


class KnowledgeRetrievalTool(BaseTool):
    """
    Knowledge base retrieval tool based on ChromaDB and OpenAI Compatible API vectorized search.
    
    This tool provides semantic search capabilities over a document collection using
    vector embeddings. It supports multiple document formats and maintains a persistent
    vector database for efficient retrieval.
    
    Implements singleton pattern to avoid multiple initializations.
    
    Attributes:
        name (str): Tool identifier name.
        description (str): Tool description for AI agents.
        knowledge_base_path (str): Path to the document collection.
        collection_name (str): Name of the ChromaDB collection.
        client: ChromaDB client instance.
        collection: ChromaDB collection instance.
        openai_client: OpenAI client for embeddings.
        embedding_function: Function to generate embeddings.
        documents (List[Dict[str, Any]]): Loaded document metadata.
    """
    
    name: str = "knowledge_retrieval"
    description: str = "Retrieve relevant documents and information from knowledge base"
    knowledge_base_path: str = "workdir/documents"
    collection_name: str = "documents"
    
    # Singleton pattern attributes
    _instance = None
    _initialized = False
    
    # Optional fields that will be set during initialization
    client: Optional[Any] = None
    collection: Optional[Any] = None
    openai_client: Optional[Any] = None
    embedding_function: Optional[Any] = None
    documents: List[Dict[str, Any]] = []
    
    def __new__(cls, knowledge_base_path: str = "workdir/documents", collection_name: str = "documents"):
        """
        Ensure only one instance exists (singleton pattern).
        
        Args:
            knowledge_base_path (str, optional): Path to the document collection directory.
            collection_name (str, optional): Name for the ChromaDB collection.
            
        Returns:
            KnowledgeRetrievalTool: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, knowledge_base_path: str = "workdir/documents", collection_name: str = "documents"):
        """
        Initialize the KnowledgeRetrievalTool (only once due to singleton pattern).
        
        Args:
            knowledge_base_path (str, optional): Path to the document collection directory.
                Defaults to "workdir/documents".
            collection_name (str, optional): Name for the ChromaDB collection.
                Defaults to "documents".
        """
        # Skip initialization if already initialized
        if self._initialized:
            return
            
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
                    "description": "Text query for retrieval"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant documents to return",
                    "default": 3
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold (between 0-1)",
                    "default": 0.3
                }
            },
            "required": ["query"]
        }
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=settings.llm_settings.api_key,
            base_url=settings.llm_settings.base_url
        )
        
        # Create embedding function
        self.embedding_function = OpenAIEmbeddingFunction()
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
        
        # Mark as initialized
        self._initialized = True
    
    def _initialize_knowledge_base(self):
        """
        Initialize the knowledge base.
        
        This method sets up the ChromaDB client, creates or loads the document collection,
        loads documents from the file system, and builds the vector index.
        
        Raises:
            Exception: If knowledge base initialization fails.
        """
        try:
            logger.info(f"Initializing knowledge base, path: {self.knowledge_base_path}")
            
            # Initialize ChromaDB client
            chroma_db_path = Path("workdir") / "chroma_db"
            chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(chroma_db_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document knowledge base"},
                    embedding_function=self.embedding_function
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Load documents
            self._load_documents()
            
            # Build vector index
            if self.documents:
                self._build_index()
                logger.info(f"Knowledge base initialization completed, loaded {len(self.documents)} documents")
            else:
                logger.warning("No documents found")
                
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
            raise
    
    def _load_documents(self):
        """
        Load documents from the document folder.
        
        This method scans the knowledge base directory for supported file formats
        and loads their content into memory for indexing.
        """
        self.documents = []
        
        if not self.knowledge_base_path.exists():
            logger.warning(f"Knowledge base path does not exist: {self.knowledge_base_path}")
            return
        
        # Supported document formats
        supported_extensions = {'.md', '.txt', '.json'}
        
        for file_path in self.knowledge_base_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    content = self._load_file_content(file_path)
                    if content.strip():  # Only add non-empty content
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
                        logger.debug(f"Loaded document: {file_path.name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load document {file_path}: {e}")
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """
        Generate document ID.
        
        Args:
            file_path (Path): Path to the document file.
            
        Returns:
            str: Unique document identifier.
        """
        # Generate unique ID using file path and modification time
        stat = file_path.stat()
        content = f"{file_path}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_file_content(self, file_path: Path) -> str:
        """
        Load file content.
        
        Args:
            file_path (Path): Path to the file to load.
            
        Returns:
            str: The content of the file as text.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                    # Convert JSON to searchable text
                    return self._json_to_searchable_text(data)
                else:
                    return f.read()
        except UnicodeDecodeError:
            # Try other encodings
            try:
                with open(file_path, 'r', encoding='gb2312') as f:
                    if file_path.suffix.lower() == '.json':
                        data = json.load(f)
                        return self._json_to_searchable_text(data)
                    else:
                        return f.read()
            except Exception:
                logger.warning(f"Cannot read file {file_path}, skipping")
                return ""
    
    def _json_to_searchable_text(self, data: Any) -> str:
        """
        Convert JSON data to searchable text.
        
        Args:
            data (Any): JSON data to convert.
            
        Returns:
            str: Searchable text representation of the JSON data.
        """
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
        """
        Build ChromaDB vector index.
        
        This method adds documents to the ChromaDB collection, generating embeddings
        and building the vector index for efficient similarity search.
        
        Raises:
            Exception: If index building fails.
        """
        try:
            logger.info("Building ChromaDB vector index...")
            
            # Get current document count in collection
            existing_count = self.collection.count()
            
            # Prepare documents to add
            new_documents = []
            ids_to_add = []
            contents_to_add = []
            metadatas_to_add = []
            
            for doc in self.documents:
                doc_id = doc["id"]
                
                # Check if document already exists
                try:
                    existing_docs = self.collection.get(ids=[doc_id])
                    if existing_docs["ids"]:
                        continue  # Document exists, skip
                except Exception:
                    pass  # Document doesn't exist, need to add
                
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
                logger.info(f"Adding {len(ids_to_add)} new documents to ChromaDB...")
                
                # Process in batches to avoid memory issues
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
                
                logger.info(f"Successfully added {len(ids_to_add)} documents")
            else:
                logger.info("All documents already exist in ChromaDB")
            
            total_count = self.collection.count()
            logger.info(f"ChromaDB index building completed, total documents: {total_count}")
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            raise
    
    async def execute(self, query: str, top_k: int = 3, threshold: float = 0.01, **kwargs) -> ToolResult:
        """
        Execute knowledge base retrieval.
        
        Args:
            query (str): Search query text.
            top_k (int, optional): Number of top results to return. Defaults to 3.
            threshold (float, optional): Similarity threshold for filtering results. Defaults to 0.01.
            **kwargs: Additional keyword arguments.
            
        Returns:
            ToolResult: Retrieval results containing found documents and metadata.
        """
        try:
            if not self.collection:
                return ToolResult(
                    error="Knowledge base not initialized",
                    output="Please ensure knowledge base is properly initialized"
                )
            
            logger.info(f"Retrieval query: {query}")
            
            # Execute vector retrieval
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            logger.info(f"Retrieved {len(results)} results")
            logger.info(results)

            if not results["ids"] or not results["ids"][0]:
                return ToolResult(
                    output="No relevant documents found, please try adjusting the query content",
                    error=None
                )
            
            # Process results
            formatted_results = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0], 
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # ChromaDB uses distance, smaller distance means higher similarity
                # Convert to similarity score (0-1)

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
                    output=f"No documents found with similarity above {threshold}, please try lowering the similarity threshold",
                    error=None
                )
            
            # Format output
            output_text = f"Found {len(formatted_results)} relevant documents:\n\n"
            for result in formatted_results:
                output_text += f"**{result['rank']}. {result['filename']}** (Similarity: {result['score']:.3f})\n"
                output_text += f"Type: {result['type']}, Size: {result['size']} characters\n"
                output_text += f"Preview: {result['content_preview']}\n"
                output_text += f"Path: {result['path']}\n\n"
            
            return ToolResult(
                output=output_text,
                system=json.dumps(formatted_results, ensure_ascii=False, indent=2)
            )
            
        except Exception as e:
            logger.error(f"Knowledge base retrieval failed: {e}")
            return ToolResult(
                error=f"Retrieval failed: {str(e)}",
                output="Please check if knowledge base is properly initialized"
            )
    
    def refresh_knowledge_base(self):
        """
        Refresh the knowledge base.
        
        This method reloads documents from the file system and rebuilds
        the vector index to incorporate any changes.
        """
        logger.info("Refreshing knowledge base...")
        try:
            # Reload documents
            self._load_documents()
            
            if self.documents:
                # Rebuild index
                self._build_index()
                logger.info(f"Knowledge base refresh completed, total {len(self.documents)} documents")
            else:
                logger.warning("No documents found after refresh")
                
        except Exception as e:
            logger.error(f"Failed to refresh knowledge base: {e}")
    
    def get_document_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Get document content by path.
        
        Args:
            path (str): File path or filename to search for.
            
        Returns:
            Optional[Dict[str, Any]]: Document metadata and content if found, None otherwise.
        """
        for doc in self.documents:
            if doc["path"] == path or doc["filename"] == path:
                return doc
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing statistics about the knowledge base
                including document count, file types, sizes, etc.
        """
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
                
                # Count file types
                for doc in self.documents:
                    file_type = doc["type"]
                    if file_type not in stats["file_types"]:
                        stats["file_types"][file_type] = 0
                    stats["file_types"][file_type] += 1
                
                if stats["total_size"] > 0:
                    stats["average_size"] = stats["total_size"] // len(self.documents)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """
        Delete collection (used for resetting knowledge base).
        
        This method removes the entire ChromaDB collection, effectively
        resetting the knowledge base to an empty state.
        """
        try:
            if self.client and self.collection_name:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection: {e}")