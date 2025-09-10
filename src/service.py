"""
서비스 계층
약물 재목적화 서비스의 메인 비즈니스 로직을 제공합니다.
"""

from typing import List, Dict, Optional
import logging
from .data_loader import DataLoader
from .graph_builder import GraphBuilder
from .ranker import DrugRepurposeRanker
from .explain import DrugDiseaseExplainer

logger = logging.getLogger(__name__)

class RepurposeService:
    """약물 재목적화 서비스의 메인 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = data_dir
        
        # 컴포넌트 초기화
        self.data_loader = DataLoader(data_dir)
        self.graph_builder = GraphBuilder(self.data_loader)
        self.ranker = DrugRepurposeRanker(self.data_loader, self.graph_builder)
        self.explainer = DrugDiseaseExplainer(self.data_loader, self.graph_builder)
        
        # 초기화 상태
        self._initialized = False
        
    def initialize(self) -> None:
        """서비스를 초기화합니다."""
        if self._initialized:
            logger.info("Service already initialized")
            return
        
        logger.info("Initializing RepurposeService...")
        
        # 데이터 로드
        self.data_loader.load_all_data()
        
        # 그래프 구축
        self.graph_builder.build_graph()
        
        self._initialized = True
        logger.info("RepurposeService initialized successfully")
    
    def rank_for_disease(self, disease_query: str, top_k: int = 10) -> List[Dict]:
        """
        특정 질병에 대한 약물 재목적화 후보를 랭킹합니다.
        
        Args:
            disease_query: 질병 쿼리 텍스트
            top_k: 반환할 상위 후보 수
            
        Returns:
            랭킹된 약물 후보 리스트
        """
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Ranking drugs for disease: {disease_query}")
        
        try:
            results = self.ranker.rank_for_disease(disease_query, top_k)
            logger.info(f"Found {len(results)} candidates")
            return results
        
        except Exception as e:
            logger.error(f"Error in rank_for_disease: {e}")
            return []
    
    def explain(self, drug_id: str, disease_query: str) -> Dict:
        """
        약물-질병 쌍에 대한 설명과 근거를 생성합니다.
        
        Args:
            drug_id: 약물 ID
            disease_query: 질병 쿼리 텍스트
            
        Returns:
            설명과 근거가 포함된 딕셔너리
        """
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Generating explanation for drug {drug_id} and disease {disease_query}")
        
        try:
            explanation = self.explainer.explain(drug_id, disease_query)
            return explanation
        
        except Exception as e:
            logger.error(f"Error in explain: {e}")
            return {"error": f"Failed to generate explanation: {str(e)}"}
    
    def get_drug_info(self, drug_id: str) -> Optional[Dict]:
        """약물 정보를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.data_loader.get_drug_by_id(drug_id)
    
    def get_disease_info(self, disease_id: str) -> Optional[Dict]:
        """질병 정보를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.data_loader.get_disease_by_id(disease_id)
    
    def search_diseases(self, query: str) -> List[Dict]:
        """질병을 검색합니다."""
        if not self._initialized:
            self.initialize()
        
        all_diseases = self.data_loader.get_all_diseases()
        query_lower = query.lower()
        
        # 간단한 검색 (이름이나 동의어에 포함)
        matching_diseases = []
        for disease in all_diseases:
            if (query_lower in disease['disease_name'].lower() or 
                query_lower in disease['synonyms'].lower()):
                matching_diseases.append(disease)
        
        return matching_diseases
    
    def get_ranking_stats(self, disease_query: str) -> Dict:
        """랭킹 통계를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.ranker.get_ranking_stats(disease_query)
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.graph_builder.get_graph_stats()
    
    def update_ranking_weights(self, text_weight: float, graph_weight: float) -> None:
        """랭킹 가중치를 업데이트합니다."""
        if not self._initialized:
            self.initialize()
        
        self.ranker.update_weights(text_weight, graph_weight)
        logger.info(f"Updated weights: text={text_weight}, graph={graph_weight}")
    
    def get_drug_mechanism_info(self, drug_id: str) -> Dict:
        """약물의 작용 메커니즘 정보를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.explainer.get_drug_mechanism_info(drug_id)
    
    def get_disease_profile(self, disease_id: str) -> Dict:
        """질병 프로필 정보를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.explainer.get_disease_profile(disease_id)
    
    def get_all_drugs(self) -> List[Dict]:
        """모든 약물 정보를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.data_loader.get_all_drugs()
    
    def get_all_diseases(self) -> List[Dict]:
        """모든 질병 정보를 반환합니다."""
        if not self._initialized:
            self.initialize()
        
        return self.data_loader.get_all_diseases()
    
    def health_check(self) -> Dict:
        """서비스 상태를 확인합니다."""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}
            
            # 기본 통계 확인
            graph_stats = self.get_graph_stats()
            drug_count = len(self.get_all_drugs())
            disease_count = len(self.get_all_diseases())
            
            return {
                "status": "healthy",
                "healthy": True,
                "initialized": self._initialized,
                "drugs_count": drug_count,
                "diseases_count": disease_count,
                "graph_nodes": graph_stats.get("total_nodes", 0),
                "graph_edges": graph_stats.get("total_edges", 0)
            }
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }
