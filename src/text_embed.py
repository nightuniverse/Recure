"""
텍스트 임베딩 모듈
sentence-transformers를 사용하여 텍스트를 벡터로 변환합니다.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 모델 캐시
_model = None

def _get_model() -> SentenceTransformer:
    """모델을 로드하고 캐시합니다."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    텍스트 리스트를 임베딩 벡터로 변환합니다.
    
    Args:
        texts: 임베딩할 텍스트 리스트
        
    Returns:
        numpy 배열 형태의 임베딩 벡터들 (shape: [len(texts), embedding_dim])
    """
    if not texts:
        return np.array([])
    
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    """
    단일 쿼리 텍스트를 임베딩 벡터로 변환합니다.
    
    Args:
        query: 임베딩할 쿼리 텍스트
        
    Returns:
        numpy 배열 형태의 임베딩 벡터 (shape: [embedding_dim])
    """
    if not query:
        return np.array([])
    
    model = _get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0]  # 단일 벡터 반환

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 벡터 간의 코사인 유사도를 계산합니다.
    
    Args:
        a, b: 비교할 벡터들
        
    Returns:
        코사인 유사도 값 (0-1 사이)
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    
    # 정규화
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)

def batch_cosine_similarity(query_embedding: np.ndarray, 
                          candidate_embeddings: np.ndarray) -> np.ndarray:
    """
    쿼리 임베딩과 후보 임베딩들 간의 배치 코사인 유사도를 계산합니다.
    
    Args:
        query_embedding: 쿼리 벡터
        candidate_embeddings: 후보 벡터들 (shape: [n_candidates, embedding_dim])
        
    Returns:
        유사도 배열 (shape: [n_candidates])
    """
    if query_embedding.size == 0 or candidate_embeddings.size == 0:
        return np.array([])
    
    # 정규화
    query_norm = np.linalg.norm(query_embedding)
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    
    # 0으로 나누기 방지
    valid_mask = (query_norm > 0) & (candidate_norms > 0)
    similarities = np.zeros(len(candidate_embeddings))
    
    if query_norm > 0:
        similarities[valid_mask] = np.dot(candidate_embeddings[valid_mask], query_embedding) / (
            query_norm * candidate_norms[valid_mask]
        )
    
    return similarities
