"""
데이터 로더 모듈
CSV 파일들을 로드하고 기본적인 정제 작업을 수행합니다.
"""

import pandas as pd
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """CSV 데이터를 로드하고 정제하는 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: CSV 파일들이 있는 디렉토리 경로
        """
        self.data_dir = data_dir
        self.drugs_df: Optional[pd.DataFrame] = None
        self.diseases_df: Optional[pd.DataFrame] = None
        self.drug_disease_df: Optional[pd.DataFrame] = None
        self.drug_gene_df: Optional[pd.DataFrame] = None
        
        # 빠른 조회를 위한 딕셔너리들
        self.drugs_by_id: Dict[str, Dict] = {}
        self.drugs_by_name: Dict[str, Dict] = {}
        self.diseases_by_id: Dict[str, Dict] = {}
        self.diseases_by_name: Dict[str, Dict] = {}
        
    def load_all_data(self) -> None:
        """모든 CSV 파일을 로드하고 정제합니다."""
        logger.info("Loading all CSV data...")
        
        self.drugs_df = self._load_and_clean_csv("seed_drugs.csv")
        self.diseases_df = self._load_and_clean_csv("seed_diseases.csv")
        self.drug_disease_df = self._load_and_clean_csv("seed_drug_disease.csv")
        self.drug_gene_df = self._load_and_clean_csv("seed_drug_gene.csv")
        
        # 딕셔너리 생성
        self._build_lookup_dictionaries()
        
        logger.info("Data loading completed")
        
    def _load_and_clean_csv(self, filename: str) -> pd.DataFrame:
        """CSV 파일을 로드하고 기본 정제를 수행합니다."""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {filename}: {len(df)} rows")
        
        # 기본 정제
        df = self._clean_dataframe(df)
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임의 기본 정제 작업을 수행합니다."""
        # 결측값 처리
        df = df.fillna("")
        
        # 문자열 컬럼들을 소문자로 변환 (ID 컬럼 제외)
        for col in df.columns:
            if col.endswith('_id') or col == 'atc':
                continue
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.lower()
        
        return df
    
    def _build_lookup_dictionaries(self) -> None:
        """빠른 조회를 위한 딕셔너리들을 생성합니다."""
        # 약물 딕셔너리
        if self.drugs_df is not None:
            for _, row in self.drugs_df.iterrows():
                drug_data = row.to_dict()
                self.drugs_by_id[row['drug_id']] = drug_data
                self.drugs_by_name[row['drug_name']] = drug_data
        
        # 질병 딕셔너리
        if self.diseases_df is not None:
            for _, row in self.diseases_df.iterrows():
                disease_data = row.to_dict()
                self.diseases_by_id[row['disease_id']] = disease_data
                self.diseases_by_name[row['disease_name']] = disease_data
        
        logger.info(f"Built lookup dictionaries: {len(self.drugs_by_id)} drugs, {len(self.diseases_by_id)} diseases")
    
    def get_drug_by_id(self, drug_id: str) -> Optional[Dict]:
        """ID로 약물 정보를 조회합니다."""
        return self.drugs_by_id.get(drug_id)
    
    def get_drug_by_name(self, drug_name: str) -> Optional[Dict]:
        """이름으로 약물 정보를 조회합니다."""
        return self.drugs_by_name.get(drug_name.lower())
    
    def get_disease_by_id(self, disease_id: str) -> Optional[Dict]:
        """ID로 질병 정보를 조회합니다."""
        return self.diseases_by_id.get(disease_id)
    
    def get_disease_by_name(self, disease_name: str) -> Optional[Dict]:
        """이름으로 질병 정보를 조회합니다."""
        return self.diseases_by_name.get(disease_name.lower())
    
    def get_drugs_for_disease(self, disease_id: str) -> List[Dict]:
        """특정 질병에 대한 알려진 약물들을 반환합니다."""
        if self.drug_disease_df is None:
            return []
        
        drug_ids = self.drug_disease_df[
            self.drug_disease_df['disease_id'] == disease_id
        ]['drug_id'].tolist()
        
        return [self.get_drug_by_id(drug_id) for drug_id in drug_ids if self.get_drug_by_id(drug_id)]
    
    def get_diseases_for_drug(self, drug_id: str) -> List[Dict]:
        """특정 약물에 대한 알려진 질병들을 반환합니다."""
        if self.drug_disease_df is None:
            return []
        
        disease_ids = self.drug_disease_df[
            self.drug_disease_df['drug_id'] == drug_id
        ]['disease_id'].tolist()
        
        return [self.get_disease_by_id(disease_id) for disease_id in disease_ids if self.get_disease_by_id(disease_id)]
    
    def get_genes_for_drug(self, drug_id: str) -> List[Dict]:
        """특정 약물에 대한 알려진 유전자들을 반환합니다."""
        if self.drug_gene_df is None:
            return []
        
        gene_data = self.drug_gene_df[
            self.drug_gene_df['drug_id'] == drug_id
        ].to_dict('records')
        
        return gene_data
    
    def get_all_drugs(self) -> List[Dict]:
        """모든 약물 정보를 반환합니다."""
        if self.drugs_df is None:
            return []
        return self.drugs_df.to_dict('records')
    
    def get_all_diseases(self) -> List[Dict]:
        """모든 질병 정보를 반환합니다."""
        if self.diseases_df is None:
            return []
        return self.diseases_df.to_dict('records')
    
    def fuzzy_match_disease(self, query: str, threshold: float = 0.3) -> Optional[Dict]:
        """
        질병 이름에 대한 퍼지 매칭을 수행합니다.
        
        Args:
            query: 검색할 질병 이름
            threshold: 매칭 임계값 (0-1)
            
        Returns:
            가장 유사한 질병 정보 또는 None
        """
        query = query.lower().strip()
        
        # 정확한 매칭 먼저 시도
        if query in self.diseases_by_name:
            return self.diseases_by_name[query]
        
        # 부분 문자열 매칭 시도
        for disease_name, disease_data in self.diseases_by_name.items():
            if query in disease_name or disease_name in query:
                return disease_data
        
        # 퍼지 매칭 (단어 단위)
        best_match = None
        best_score = 0
        
        for disease_name, disease_data in self.diseases_by_name.items():
            # 단어 단위로 매칭
            query_words = set(query.split())
            disease_words = set(disease_name.split())
            
            if query_words & disease_words:  # 교집합이 있으면
                # Jaccard 유사도 계산
                intersection = len(query_words & disease_words)
                union = len(query_words | disease_words)
                score = intersection / union if union > 0 else 0
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = disease_data
        
        return best_match
