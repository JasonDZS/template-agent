#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM调用日志记录器
用于记录所有大模型的输入输出
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock

from app.logger import logger


class LLMCallLogger:
    """LLM调用日志记录器"""
    
    def __init__(self, log_file_path: str = "workdir/llm_response.json"):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 线程锁，确保并发写入安全
        self._lock = Lock()
        
        # 初始化日志文件
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """初始化日志文件"""
        if not self.log_file_path.exists():
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def log_llm_call(self, 
                     model: str,
                     messages: List[Dict[str, str]],
                     response: str,
                     metadata: Optional[Dict[str, Any]] = None):
        """记录LLM调用"""
        try:
            # 创建调用记录
            call_record = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "messages": messages,
                "response": response,
                "metadata": metadata or {},
                "token_count": {
                    "input_tokens": self._estimate_tokens(messages),
                    "output_tokens": self._estimate_tokens([{"content": response}])
                }
            }
            
            # 线程安全地写入文件
            with self._lock:
                self._append_to_log_file(call_record)
                
            logger.debug(f"LLM调用已记录: {model}, 输入{call_record['token_count']['input_tokens']}tokens, 输出{call_record['token_count']['output_tokens']}tokens")
            
        except Exception as e:
            logger.error(f"LLM调用记录失败: {e}")
    
    def _append_to_log_file(self, record: Dict[str, Any]):
        """追加记录到日志文件"""
        try:
            # 读取现有记录
            if self.log_file_path.exists():
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
            else:
                existing_records = []
            
            # 添加新记录
            existing_records.append(record)
            
            # 写回文件
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_records, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"写入LLM日志文件失败: {e}")
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """估算token数量（简单估算：英文按4字符1token，中文按1.5字符1token）"""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            total_chars += len(content)
        
        # 简单估算：平均2.5字符1token
        return int(total_chars / 2.5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取调用统计信息"""
        try:
            if not self.log_file_path.exists():
                return {
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "models_used": [],
                    "date_range": None
                }
            
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            if not records:
                return {
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "models_used": [],
                    "date_range": None
                }
            
            # 统计信息
            total_calls = len(records)
            total_input_tokens = sum(r.get("token_count", {}).get("input_tokens", 0) for r in records)
            total_output_tokens = sum(r.get("token_count", {}).get("output_tokens", 0) for r in records)
            models_used = list(set(r.get("model", "unknown") for r in records))
            
            timestamps = [r.get("timestamp") for r in records if r.get("timestamp")]
            date_range = {
                "earliest": min(timestamps) if timestamps else None,
                "latest": max(timestamps) if timestamps else None
            }
            
            return {
                "total_calls": total_calls,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "models_used": models_used,
                "date_range": date_range
            }
            
        except Exception as e:
            logger.error(f"获取LLM调用统计失败: {e}")
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "models_used": [],
                "date_range": None,
                "error": str(e)
            }
    
    def clear_logs(self):
        """清空日志记录"""
        try:
            with self._lock:
                with open(self.log_file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                logger.info("LLM调用日志已清空")
        except Exception as e:
            logger.error(f"清空LLM日志失败: {e}")
    
    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的调用记录"""
        try:
            if not self.log_file_path.exists():
                return []
            
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            # 按时间戳排序，返回最近的记录
            sorted_records = sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)
            return sorted_records[:limit]
            
        except Exception as e:
            logger.error(f"获取最近LLM调用记录失败: {e}")
            return []


# 全局单例实例
_llm_call_logger = None

def get_llm_logger() -> LLMCallLogger:
    """获取LLM调用记录器单例"""
    global _llm_call_logger
    if _llm_call_logger is None:
        _llm_call_logger = LLMCallLogger()
    return _llm_call_logger