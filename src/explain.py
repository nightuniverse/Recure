"""
설명/근거 모듈
약물-질병 쌍에 대한 설명과 근거를 생성합니다.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
import logging
from .data_loader import DataLoader
from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

class DrugDiseaseExplainer:
    """약물-질병 쌍에 대한 설명과 근거를 생성하는 클래스"""
    
    def __init__(self, data_loader: DataLoader, graph_builder: GraphBuilder):
        """
        Args:
            data_loader: 데이터 로더 인스턴스
            graph_builder: 그래프 빌더 인스턴스
        """
        self.data_loader = data_loader
        self.graph_builder = graph_builder
    
    def explain(self, drug_id: str, disease_query: str) -> Dict:
        """
        약물-질병 쌍에 대한 설명과 근거를 생성합니다.
        
        Args:
            drug_id: 약물 ID
            disease_query: 질병 쿼리 텍스트
            
        Returns:
            설명과 근거가 포함된 딕셔너리
        """
        logger.info(f"Generating explanation for drug {drug_id} and disease {disease_query}")
        
        # 질병 매칭
        target_disease = self.data_loader.fuzzy_match_disease(disease_query)
        if not target_disease:
            return {"error": f"No matching disease found for: {disease_query}"}
        
        disease_id = target_disease['disease_id']
        
        # 약물 정보 조회
        drug_info = self.data_loader.get_drug_by_id(drug_id)
        if not drug_info:
            return {"error": f"Drug not found: {drug_id}"}
        
        # 설명 구성
        explanation = {
            "drug_id": drug_id,
            "drug_name": drug_info['drug_name'],
            "disease_id": disease_id,
            "disease_name": target_disease['disease_name'],
            "disease_query": disease_query,
            "graph_paths": self._get_graph_paths(drug_id, disease_id),
            "text_overlaps": self._get_text_overlaps(drug_info, target_disease),
            "known_evidence": self._get_known_evidence(drug_id, disease_id),
            "drug_info": {
                "atc": drug_info['atc'],
                "indications_text": drug_info['indications_text']
            },
            "disease_info": {
                "synonyms": target_disease['synonyms']
            }
        }
        
        return explanation
    
    def _get_graph_paths(self, drug_id: str, disease_id: str, max_paths: int = 3) -> List[Dict]:
        """그래프에서 두 노드 간의 경로들을 찾아 설명을 생성합니다."""
        paths = self.graph_builder.get_shortest_paths(drug_id, disease_id, max_length=3)
        
        path_explanations = []
        for i, path in enumerate(paths[:max_paths]):
            path_explanation = self._explain_path(path)
            path_explanations.append({
                "path_id": i + 1,
                "path": path,
                "length": len(path) - 1,
                "explanation": path_explanation
            })
        
        return path_explanations
    
    def _explain_path(self, path: List[str]) -> str:
        """경로를 텍스트 설명으로 변환합니다."""
        if len(path) < 2:
            return "No path found"
        
        explanations = []
        
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            edge_explanation = self._explain_edge(current_node, next_node)
            explanations.append(edge_explanation)
        
        return " → ".join(explanations)
    
    def _explain_edge(self, node1: str, node2: str) -> str:
        """두 노드 간의 엣지를 설명합니다."""
        if not self.graph_builder.graph.has_edge(node1, node2):
            return "No connection"
        
        edge_data = self.graph_builder.graph[node1][node2]
        edge_type = edge_data.get('edge_type', 'unknown')
        
        # 노드 정보 추출
        node1_info = self._get_node_display_name(node1)
        node2_info = self._get_node_display_name(node2)
        
        if edge_type == 'drug_disease':
            evidence = edge_data.get('evidence', '')
            return f"{node1_info} treats {node2_info} ({evidence})"
        
        elif edge_type == 'drug_gene':
            note = edge_data.get('note', '')
            return f"{node1_info} targets {node2_info} ({note})"
        
        elif edge_type == 'disease_gene_propagated':
            via_drug = edge_data.get('via_drug', '')
            via_drug_name = self._get_node_display_name(via_drug)
            return f"{node1_info} associated with {node2_info} (via {via_drug_name})"
        
        else:
            return f"{node1_info} connected to {node2_info}"
    
    def _get_node_display_name(self, node_id: str) -> str:
        """노드 ID를 표시용 이름으로 변환합니다."""
        if node_id.startswith('drug:'):
            drug_id = node_id.replace('drug:', '')
            drug_info = self.data_loader.get_drug_by_id(drug_id)
            return drug_info['drug_name'] if drug_info else drug_id
        
        elif node_id.startswith('dis:'):
            disease_id = node_id.replace('dis:', '')
            disease_info = self.data_loader.get_disease_by_id(disease_id)
            return disease_info['disease_name'] if disease_info else disease_id
        
        elif node_id.startswith('gene:'):
            gene_symbol = node_id.replace('gene:', '')
            return gene_symbol
        
        return node_id
    
    def _get_text_overlaps(self, drug_info: Dict, disease_info: Dict) -> Dict:
        """약물의 적응증 텍스트와 질병 동의어 간의 겹치는 토큰들을 찾습니다."""
        drug_text = drug_info.get('indications_text', '').lower()
        disease_name = disease_info.get('disease_name', '').lower()
        disease_synonyms = disease_info.get('synonyms', '').lower()
        
        # 질병 관련 텍스트 결합
        disease_text = f"{disease_name} {disease_synonyms}"
        
        # 토큰화 (단어 단위)
        drug_tokens = set(re.findall(r'\b\w+\b', drug_text))
        disease_tokens = set(re.findall(r'\b\w+\b', disease_text))
        
        # 겹치는 토큰들
        overlapping_tokens = drug_tokens.intersection(disease_tokens)
        
        # 의미있는 토큰들만 필터링 (길이 3 이상)
        meaningful_overlaps = {token for token in overlapping_tokens if len(token) >= 3}
        
        return {
            "overlapping_tokens": list(meaningful_overlaps),
            "overlap_count": len(meaningful_overlaps),
            "drug_tokens": list(drug_tokens),
            "disease_tokens": list(disease_tokens),
            "overlap_ratio": len(meaningful_overlaps) / max(len(disease_tokens), 1)
        }
    
    def _get_known_evidence(self, drug_id: str, disease_id: str) -> Optional[Dict]:
        """알려진 약물-질병 관계의 증거를 조회합니다."""
        if self.data_loader.drug_disease_df is not None:
            evidence_row = self.data_loader.drug_disease_df[
                (self.data_loader.drug_disease_df['drug_id'] == drug_id) &
                (self.data_loader.drug_disease_df['disease_id'] == disease_id)
            ]
            
            if not evidence_row.empty:
                return {
                    "has_known_evidence": True,
                    "evidence": evidence_row.iloc[0]['evidence']
                }
        
        return {
            "has_known_evidence": False,
            "evidence": None
        }
    
    def get_drug_mechanism_info(self, drug_id: str) -> Dict:
        """약물의 작용 메커니즘 정보를 반환합니다."""
        drug_info = self.data_loader.get_drug_by_id(drug_id)
        if not drug_info:
            return {"error": f"Drug not found: {drug_id}"}
        
        # 관련 유전자들
        related_genes = self.data_loader.get_genes_for_drug(drug_id)
        
        # 관련 질병들
        related_diseases = self.data_loader.get_diseases_for_drug(drug_id)
        
        return {
            "drug_id": drug_id,
            "drug_name": drug_info['drug_name'],
            "atc_code": drug_info['atc'],
            "indications": drug_info['indications_text'],
            "related_genes": related_genes,
            "related_diseases": related_diseases,
            "gene_count": len(related_genes),
            "disease_count": len(related_diseases)
        }
    
    def get_disease_profile(self, disease_id: str) -> Dict:
        """질병 프로필 정보를 반환합니다."""
        disease_info = self.data_loader.get_disease_by_id(disease_id)
        if not disease_info:
            return {"error": f"Disease not found: {disease_id}"}
        
        # 관련 약물들
        related_drugs = self.data_loader.get_drugs_for_disease(disease_id)
        
        return {
            "disease_id": disease_id,
            "disease_name": disease_info['disease_name'],
            "synonyms": disease_info['synonyms'],
            "related_drugs": related_drugs,
            "drug_count": len(related_drugs)
        }
