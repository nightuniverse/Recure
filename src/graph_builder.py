"""
그래프 빌더 모듈
NetworkX를 사용하여 약물-질병-유전자 그래프를 구축합니다.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from .data_loader import DataLoader

logger = logging.getLogger(__name__)

class GraphBuilder:
    """약물-질병-유전자 그래프를 구축하고 분석하는 클래스"""
    
    def __init__(self, data_loader: DataLoader):
        """
        Args:
            data_loader: 데이터 로더 인스턴스
        """
        self.data_loader = data_loader
        self.graph: nx.Graph = nx.Graph()
        self.drug_nodes: Set[str] = set()
        self.disease_nodes: Set[str] = set()
        self.gene_nodes: Set[str] = set()
        
    def build_graph(self) -> nx.Graph:
        """데이터를 기반으로 그래프를 구축합니다."""
        logger.info("Building drug-disease-gene graph...")
        
        # 그래프 초기화
        self.graph.clear()
        self.drug_nodes.clear()
        self.disease_nodes.clear()
        self.gene_nodes.clear()
        
        # 노드 추가
        self._add_drug_nodes()
        self._add_disease_nodes()
        self._add_gene_nodes()
        
        # 엣지 추가
        self._add_drug_disease_edges()
        self._add_drug_gene_edges()
        self._add_disease_gene_edges_by_propagation()
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        logger.info(f"Drug nodes: {len(self.drug_nodes)}, Disease nodes: {len(self.disease_nodes)}, Gene nodes: {len(self.gene_nodes)}")
        
        return self.graph
    
    def _add_drug_nodes(self) -> None:
        """약물 노드들을 추가합니다."""
        for drug in self.data_loader.get_all_drugs():
            node_id = f"drug:{drug['drug_id']}"
            self.graph.add_node(node_id, 
                              type='drug',
                              name=drug['drug_name'],
                              atc=drug['atc'],
                              indications=drug['indications_text'])
            self.drug_nodes.add(node_id)
    
    def _add_disease_nodes(self) -> None:
        """질병 노드들을 추가합니다."""
        for disease in self.data_loader.get_all_diseases():
            node_id = f"dis:{disease['disease_id']}"
            self.graph.add_node(node_id,
                              type='disease',
                              name=disease['disease_name'],
                              synonyms=disease['synonyms'])
            self.disease_nodes.add(node_id)
    
    def _add_gene_nodes(self) -> None:
        """유전자 노드들을 추가합니다."""
        if self.data_loader.drug_gene_df is not None:
            unique_genes = self.data_loader.drug_gene_df['gene_symbol'].unique()
            for gene in unique_genes:
                node_id = f"gene:{gene}"
                self.graph.add_node(node_id,
                                  type='gene',
                                  symbol=gene)
                self.gene_nodes.add(node_id)
    
    def _add_drug_disease_edges(self) -> None:
        """약물-질병 엣지들을 추가합니다 (가중치: 2)."""
        if self.data_loader.drug_disease_df is not None:
            for _, row in self.data_loader.drug_disease_df.iterrows():
                drug_node = f"drug:{row['drug_id']}"
                disease_node = f"dis:{row['disease_id']}"
                
                if drug_node in self.graph and disease_node in self.graph:
                    self.graph.add_edge(drug_node, disease_node,
                                      weight=2.0,
                                      evidence=row['evidence'],
                                      edge_type='drug_disease')
    
    def _add_drug_gene_edges(self) -> None:
        """약물-유전자 엣지들을 추가합니다 (가중치: 1)."""
        if self.data_loader.drug_gene_df is not None:
            for _, row in self.data_loader.drug_gene_df.iterrows():
                drug_node = f"drug:{row['drug_id']}"
                gene_node = f"gene:{row['gene_symbol']}"
                
                if drug_node in self.graph and gene_node in self.graph:
                    self.graph.add_edge(drug_node, gene_node,
                                      weight=1.0,
                                      note=row['note'],
                                      edge_type='drug_gene')
    
    def _add_disease_gene_edges_by_propagation(self) -> None:
        """질병-유전자 엣지들을 전파를 통해 추가합니다."""
        # 약물을 통해 연결된 질병-유전자 쌍들을 찾아서 엣지 추가
        for drug_node in self.drug_nodes:
            # 이 약물과 연결된 질병들
            disease_neighbors = [n for n in self.graph.neighbors(drug_node) 
                               if n.startswith('dis:')]
            # 이 약물과 연결된 유전자들
            gene_neighbors = [n for n in self.graph.neighbors(drug_node) 
                            if n.startswith('gene:')]
            
            # 질병-유전자 엣지 추가 (가중치: 0.5)
            for disease_node in disease_neighbors:
                for gene_node in gene_neighbors:
                    if not self.graph.has_edge(disease_node, gene_node):
                        self.graph.add_edge(disease_node, gene_node,
                                          weight=0.5,
                                          edge_type='disease_gene_propagated',
                                          via_drug=drug_node)
    
    def compute_link_prediction_scores(self, drug_id: str, disease_id: str) -> Dict[str, float]:
        """
        특정 약물-질병 쌍에 대한 링크 예측 점수를 계산합니다.
        
        Args:
            drug_id: 약물 ID
            disease_id: 질병 ID
            
        Returns:
            링크 예측 점수 딕셔너리
        """
        drug_node = f"drug:{drug_id}"
        disease_node = f"dis:{disease_id}"
        
        if drug_node not in self.graph or disease_node not in self.graph:
            return {"adamic_adar": 0.0, "common_neighbors": 0.0}
        
        # Adamic-Adar 점수
        adamic_adar = nx.adamic_adar_index(self.graph, [(drug_node, disease_node)])
        adamic_adar_score = next(adamic_adar)[2] if adamic_adar else 0.0
        
        # 공통 이웃 수
        common_neighbors = len(list(nx.common_neighbors(self.graph, drug_node, disease_node)))
        
        # 정규화된 점수 (0-1 범위)
        max_possible_neighbors = min(self.graph.degree(drug_node), self.graph.degree(disease_node))
        normalized_common_neighbors = common_neighbors / max(max_possible_neighbors, 1)
        
        return {
            "adamic_adar": float(adamic_adar_score),
            "common_neighbors": float(common_neighbors),
            "normalized_common_neighbors": float(normalized_common_neighbors)
        }
    
    def get_shortest_paths(self, drug_id: str, disease_id: str, max_length: int = 3) -> List[List[str]]:
        """
        두 노드 간의 최단 경로들을 반환합니다.
        
        Args:
            drug_id: 약물 ID
            disease_id: 질병 ID
            max_length: 최대 경로 길이
            
        Returns:
            경로 리스트 (각 경로는 노드 ID 리스트)
        """
        drug_node = f"drug:{drug_id}"
        disease_node = f"dis:{disease_id}"
        
        if drug_node not in self.graph or disease_node not in self.graph:
            return []
        
        try:
            # 최단 경로 계산
            shortest_path = nx.shortest_path(self.graph, drug_node, disease_node)
            if len(shortest_path) <= max_length + 1:  # +1 because path includes both endpoints
                return [shortest_path]
            
            # 최대 길이를 초과하는 경우, 간단한 경로들 반환
            paths = []
            for path in nx.all_simple_paths(self.graph, drug_node, disease_node, cutoff=max_length):
                if len(path) <= max_length + 1:
                    paths.append(path)
                    if len(paths) >= 3:  # 최대 3개 경로만 반환
                        break
            
            return paths
            
        except nx.NetworkXNoPath:
            return []
    
    def get_node_info(self, node_id: str) -> Optional[Dict]:
        """노드 정보를 반환합니다."""
        if node_id in self.graph:
            return self.graph.nodes[node_id]
        return None
    
    def get_neighbors(self, node_id: str, node_type: Optional[str] = None) -> List[str]:
        """
        노드의 이웃들을 반환합니다.
        
        Args:
            node_id: 노드 ID
            node_type: 필터링할 노드 타입 ('drug', 'disease', 'gene')
            
        Returns:
            이웃 노드 ID 리스트
        """
        if node_id not in self.graph:
            return []
        
        neighbors = list(self.graph.neighbors(node_id))
        
        if node_type:
            neighbors = [n for n in neighbors if n.startswith(f"{node_type}:")]
        
        return neighbors
    
    def get_graph_stats(self) -> Dict:
        """그래프 통계를 반환합니다."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "drug_nodes": len(self.drug_nodes),
            "disease_nodes": len(self.disease_nodes),
            "gene_nodes": len(self.gene_nodes),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph)
        }
