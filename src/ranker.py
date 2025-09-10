"""
랭커 모듈
그래프 기반 점수와 텍스트 임베딩 점수를 결합하여 약물 재목적화 후보를 랭킹합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .data_loader import DataLoader
from .graph_builder import GraphBuilder
from .text_embed import embed_query, batch_cosine_similarity

logger = logging.getLogger(__name__)

class DrugRepurposeRanker:
    """약물 재목적화 후보를 랭킹하는 클래스"""
    
    def __init__(self, data_loader: DataLoader, graph_builder: GraphBuilder, 
                 text_weight: float = 0.6, graph_weight: float = 0.4):
        """
        Args:
            data_loader: 데이터 로더 인스턴스
            graph_builder: 그래프 빌더 인스턴스
            text_weight: 텍스트 점수 가중치 (기본값: 0.6)
            graph_weight: 그래프 점수 가중치 (기본값: 0.4)
        """
        self.data_loader = data_loader
        self.graph_builder = graph_builder
        self.text_weight = text_weight
        self.graph_weight = graph_weight
        
        # 캐시된 임베딩들
        self.drug_embeddings: Dict[str, np.ndarray] = {}
        self._cache_drug_embeddings()
    
    def _cache_drug_embeddings(self) -> None:
        """모든 약물의 임베딩을 미리 계산하고 캐시합니다."""
        logger.info("Caching drug embeddings...")
        
        for drug in self.data_loader.get_all_drugs():
            drug_id = drug['drug_id']
            indications_text = drug['indications_text']
            
            if indications_text:
                embedding = embed_query(indications_text)
                self.drug_embeddings[drug_id] = embedding
        
        logger.info(f"Cached embeddings for {len(self.drug_embeddings)} drugs")
    
    def rank_for_disease(self, disease_query: str, top_k: int = 10) -> List[Dict]:
        """
        특정 질병에 대한 약물 재목적화 후보를 랭킹합니다.
        
        Args:
            disease_query: 질병 쿼리 텍스트
            top_k: 반환할 상위 후보 수
            
        Returns:
            랭킹된 약물 후보 리스트
        """
        logger.info(f"Ranking drugs for disease: {disease_query}")
        
        # 질병 매칭
        target_disease = self.data_loader.fuzzy_match_disease(disease_query)
        if not target_disease:
            logger.warning(f"No matching disease found for: {disease_query}")
            return []
        
        disease_id = target_disease['disease_id']
        logger.info(f"Matched disease: {target_disease['disease_name']} ({disease_id})")
        
        # 이미 알려진 약물들 제외
        known_drugs = self.data_loader.get_drugs_for_disease(disease_id)
        known_drug_ids = {drug['drug_id'] for drug in known_drugs}
        
        # 후보 약물들 (알려진 약물 제외)
        all_drugs = self.data_loader.get_all_drugs()
        candidate_drugs = [drug for drug in all_drugs if drug['drug_id'] not in known_drug_ids]
        
        if not candidate_drugs:
            logger.info("No candidate drugs found")
            return []
        
        logger.info(f"Found {len(candidate_drugs)} candidate drugs")
        
        # 점수 계산
        scored_candidates = []
        for drug in candidate_drugs:
            drug_id = drug['drug_id']
            
            # 텍스트 점수 계산
            text_score = self._compute_text_score(drug, target_disease)
            
            # 그래프 점수 계산
            graph_score = self._compute_graph_score(drug_id, disease_id)
            
            # 결합 점수
            combined_score = (self.text_weight * text_score + 
                            self.graph_weight * graph_score)
            
            scored_candidates.append({
                'drug_id': drug_id,
                'drug_name': drug['drug_name'],
                'atc': drug['atc'],
                'indications_text': drug['indications_text'],
                'score': float(combined_score),
                'text_score': float(text_score),
                'graph_score': float(graph_score),
                'target_disease_id': disease_id,
                'target_disease_name': target_disease['disease_name']
            })
        
        # 점수순으로 정렬하고 상위 k개 반환
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 점수 정규화 (0-1 범위)
        if scored_candidates:
            max_score = scored_candidates[0]['score']
            min_score = scored_candidates[-1]['score']
            score_range = max_score - min_score
            
            if score_range > 0:
                for candidate in scored_candidates:
                    candidate['normalized_score'] = (candidate['score'] - min_score) / score_range
            else:
                for candidate in scored_candidates:
                    candidate['normalized_score'] = 1.0
        
        return scored_candidates[:top_k]
    
    def _compute_text_score(self, drug: Dict, disease: Dict) -> float:
        """텍스트 기반 유사도 점수를 계산합니다."""
        drug_id = drug['drug_id']
        drug_text = drug['indications_text']
        
        # 질병 텍스트 구성 (이름 + 동의어)
        disease_text = f"{disease['disease_name']} {disease['synonyms']}"
        
        # 임베딩 계산
        if drug_id not in self.drug_embeddings:
            return 0.0
        
        drug_embedding = self.drug_embeddings[drug_id]
        disease_embedding = embed_query(disease_text)
        
        # 코사인 유사도 계산
        from .text_embed import cosine_similarity
        similarity = cosine_similarity(drug_embedding, disease_embedding)
        
        return max(0.0, similarity)  # 음수 값 방지
    
    def _compute_graph_score(self, drug_id: str, disease_id: str) -> float:
        """그래프 기반 점수를 계산합니다."""
        # 링크 예측 점수 계산
        link_scores = self.graph_builder.compute_link_prediction_scores(drug_id, disease_id)
        
        # Adamic-Adar 점수 정규화
        adamic_adar = link_scores.get('adamic_adar', 0.0)
        normalized_adamic_adar = min(1.0, adamic_adar / 2.0)  # 임의의 정규화
        
        # 공통 이웃 점수
        normalized_common_neighbors = link_scores.get('normalized_common_neighbors', 0.0)
        
        # 결합된 그래프 점수
        graph_score = 0.7 * normalized_adamic_adar + 0.3 * normalized_common_neighbors
        
        return min(1.0, graph_score)
    
    def get_candidate_drugs(self, disease_id: str) -> List[Dict]:
        """특정 질병에 대한 후보 약물들을 반환합니다 (이미 알려진 약물 제외)."""
        known_drugs = self.data_loader.get_drugs_for_disease(disease_id)
        known_drug_ids = {drug['drug_id'] for drug in known_drugs}
        
        all_drugs = self.data_loader.get_all_drugs()
        candidate_drugs = [drug for drug in all_drugs if drug['drug_id'] not in known_drug_ids]
        
        return candidate_drugs
    
    def update_weights(self, text_weight: float, graph_weight: float) -> None:
        """점수 가중치를 업데이트합니다."""
        total_weight = text_weight + graph_weight
        if total_weight > 0:
            self.text_weight = text_weight / total_weight
            self.graph_weight = graph_weight / total_weight
        else:
            logger.warning("Total weight is zero, keeping current weights")
    
    def get_ranking_stats(self, disease_query: str) -> Dict:
        """랭킹 통계를 반환합니다."""
        target_disease = self.data_loader.fuzzy_match_disease(disease_query)
        if not target_disease:
            return {"error": "Disease not found"}
        
        disease_id = target_disease['disease_id']
        known_drugs = self.data_loader.get_drugs_for_disease(disease_id)
        candidate_drugs = self.get_candidate_drugs(disease_id)
        
        return {
            "target_disease": target_disease['disease_name'],
            "known_drugs_count": len(known_drugs),
            "candidate_drugs_count": len(candidate_drugs),
            "text_weight": self.text_weight,
            "graph_weight": self.graph_weight,
            "cached_embeddings": len(self.drug_embeddings)
        }
