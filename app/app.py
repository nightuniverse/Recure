"""
Streamlit UI 애플리케이션
약물 재목적화 웹 인터페이스를 제공합니다.
"""

import streamlit as st
import pandas as pd
import requests
import json
from typing import Dict, List, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="Drug Repurposing Tool",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API 기본 URL
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """API 서비스 상태를 확인합니다."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False

def call_api(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """API를 호출하고 결과를 반환합니다."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        st.error(f"API 호출 실패: {e}")
        return None

def display_ranking_results(results: List[Dict]) -> None:
    """랭킹 결과를 표시합니다."""
    if not results:
        st.warning("결과가 없습니다.")
        return
    
    # DataFrame 생성
    df_data = []
    for result in results:
        df_data.append({
            "순위": len(df_data) + 1,
            "약물 ID": result["drug_id"],
            "약물명": result["drug_name"],
            "ATC 코드": result["atc"],
            "종합 점수": f"{result['score']:.4f}",
            "텍스트 점수": f"{result['text_score']:.4f}",
            "그래프 점수": f"{result['graph_score']:.4f}",
            "정규화 점수": f"{result.get('normalized_score', 0):.4f}",
            "적응증": result["indications_text"]
        })
    
    df = pd.DataFrame(df_data)
    
    # 결과 표시
    st.subheader("📊 약물 재목적화 후보 랭킹")
    st.dataframe(df, use_container_width=True)
    
    # 선택 가능한 행
    st.subheader("🔍 상세 분석")
    selected_idx = st.selectbox(
        "분석할 약물을 선택하세요:",
        range(len(results)),
        format_func=lambda x: f"{results[x]['drug_name']} ({results[x]['drug_id']})"
    )
    
    if selected_idx is not None:
        selected_drug = results[selected_idx]
        st.write(f"**선택된 약물:** {selected_drug['drug_name']} ({selected_drug['drug_id']})")
        
        # 상세 정보 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("종합 점수", f"{selected_drug['score']:.4f}")
            st.metric("텍스트 점수", f"{selected_drug['text_score']:.4f}")
        
        with col2:
            st.metric("그래프 점수", f"{selected_drug['graph_score']:.4f}")
            st.metric("정규화 점수", f"{selected_drug.get('normalized_score', 0):.4f}")
        
        # 적응증 텍스트
        st.write("**적응증:**")
        st.info(selected_drug['indications_text'])

def display_explanation(explanation: Dict) -> None:
    """설명 결과를 표시합니다."""
    if "error" in explanation:
        st.error(f"설명 생성 실패: {explanation['error']}")
        return
    
    st.subheader("🔬 약물-질병 관계 분석")
    
    # 기본 정보
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**약물 정보:**")
        st.write(f"- ID: {explanation['drug_id']}")
        st.write(f"- 이름: {explanation['drug_name']}")
        st.write(f"- ATC: {explanation['drug_info']['atc']}")
        st.write(f"- 적응증: {explanation['drug_info']['indications_text']}")
    
    with col2:
        st.write("**질병 정보:**")
        st.write(f"- ID: {explanation['disease_id']}")
        st.write(f"- 이름: {explanation['disease_name']}")
        st.write(f"- 동의어: {explanation['disease_info']['synonyms']}")
    
    # 알려진 증거
    if explanation['known_evidence']['has_known_evidence']:
        st.success(f"✅ 알려진 증거: {explanation['known_evidence']['evidence']}")
    else:
        st.info("ℹ️ 알려진 직접적인 증거가 없습니다.")
    
    # 그래프 경로
    st.subheader("🕸️ 그래프 경로 분석")
    if explanation['graph_paths']:
        for i, path_info in enumerate(explanation['graph_paths']):
            with st.expander(f"경로 {path_info['path_id']} (길이: {path_info['length']})"):
                st.write(f"**경로:** {' → '.join(path_info['path'])}")
                st.write(f"**설명:** {path_info['explanation']}")
    else:
        st.warning("그래프에서 직접적인 경로를 찾을 수 없습니다.")
    
    # 텍스트 겹침 분석
    st.subheader("📝 텍스트 유사성 분석")
    overlaps = explanation['text_overlaps']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("겹치는 토큰 수", overlaps['overlap_count'])
        st.metric("겹침 비율", f"{overlaps['overlap_ratio']:.2%}")
    
    with col2:
        if overlaps['overlapping_tokens']:
            st.write("**겹치는 토큰들:**")
            st.write(", ".join(overlaps['overlapping_tokens']))
        else:
            st.write("겹치는 토큰이 없습니다.")

def main():
    """메인 애플리케이션"""
    st.title("💊 Drug Repurposing Tool")
    st.markdown("약물 재목적화를 위한 AI 기반 분석 도구")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API 상태 확인
        if check_api_health():
            st.success("✅ API 서비스 연결됨")
        else:
            st.error("❌ API 서비스 연결 실패")
            st.info("API 서버가 실행 중인지 확인하세요: `make run-api`")
            return
        
        # K 값 선택
        k_options = [5, 10, 20]
        selected_k = st.selectbox("반환할 후보 수:", k_options, index=1)
        
        st.markdown("---")
        st.markdown("### 📚 사용법")
        st.markdown("""
        1. 질병 이름을 입력하세요
        2. '랭킹' 버튼을 클릭하세요
        3. 결과에서 약물을 선택하여 상세 분석을 확인하세요
        """)
    
    # 메인 컨텐츠
    st.header("🔍 약물 재목적화 검색")
    
    # 질병 입력
    disease_query = st.text_input(
        "질병 이름을 입력하세요:",
        placeholder="예: Parkinson's disease, Alzheimer's disease",
        help="질병의 정확한 이름이나 일반적인 이름을 입력하세요."
    )
    
    # 검색 버튼
    if st.button("🔍 랭킹", type="primary"):
        if not disease_query.strip():
            st.warning("질병 이름을 입력해주세요.")
        else:
            with st.spinner("약물 재목적화 후보를 분석 중..."):
                # API 호출
                result = call_api("/rank", {"disease": disease_query, "k": selected_k})
                
                if result:
                    st.session_state.ranking_results = result["candidates"]
                    st.session_state.target_disease = disease_query
    
    # 랭킹 결과 표시
    if "ranking_results" in st.session_state:
        display_ranking_results(st.session_state.ranking_results)
        
        # 상세 분석 버튼
        if st.button("🔬 선택된 약물 상세 분석"):
            if "ranking_results" in st.session_state and st.session_state.ranking_results:
                # 첫 번째 약물을 기본 선택
                selected_drug = st.session_state.ranking_results[0]
                target_disease = st.session_state.get("target_disease", "")
                
                with st.spinner("상세 분석을 생성 중..."):
                    explanation = call_api("/explain", {
                        "disease": target_disease,
                        "drug_id": selected_drug["drug_id"]
                    })
                    
                    if explanation:
                        display_explanation(explanation)
    
    # 추가 정보 섹션
    st.markdown("---")
    st.header("📊 서비스 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 서비스 통계"):
            stats = call_api("/stats")
            if stats:
                st.json(stats)
    
    with col2:
        if st.button("💊 모든 약물 목록"):
            drugs = call_api("/drugs")
            if drugs:
                df = pd.DataFrame(drugs["drugs"])
                st.dataframe(df, use_container_width=True)
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Drug Repurposing Tool v1.0.0 | AI-powered drug discovery</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
