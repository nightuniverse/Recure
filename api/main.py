"""
FastAPI 메인 애플리케이션
약물 재목적화 API를 제공합니다.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import logging
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.service import RepurposeService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Drug Repurposing API",
    description="약물 재목적화를 위한 API 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 서비스 인스턴스
service = None

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 서비스를 초기화합니다."""
    global service
    logger.info("Starting up Drug Repurposing API...")
    service = RepurposeService()
    service.initialize()
    logger.info("API startup completed")

@app.get("/health")
async def health_check():
    """서비스 상태를 확인합니다."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    health_status = service.health_check()
    
    if not health_status.get("healthy", False):
        raise HTTPException(status_code=503, detail="Service unhealthy")
    
    return {"ok": True, **health_status}

@app.get("/rank")
async def rank_drugs(
    disease: str = Query(..., description="질병 이름"),
    k: int = Query(10, ge=1, le=50, description="반환할 상위 후보 수")
):
    """
    특정 질병에 대한 약물 재목적화 후보를 랭킹합니다.
    
    Args:
        disease: 질병 이름
        k: 반환할 상위 후보 수 (1-50)
        
    Returns:
        랭킹된 약물 후보 리스트
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = service.rank_for_disease(disease, top_k=k)
        
        if not results:
            return {
                "disease": disease,
                "k": k,
                "candidates": [],
                "message": "No candidates found"
            }
        
        return {
            "disease": disease,
            "k": k,
            "candidates": results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error in rank_drugs: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/explain")
async def explain_drug_disease(
    disease: str = Query(..., description="질병 이름"),
    drug_id: str = Query(..., description="약물 ID")
):
    """
    약물-질병 쌍에 대한 설명과 근거를 생성합니다.
    
    Args:
        disease: 질병 이름
        drug_id: 약물 ID
        
    Returns:
        설명과 근거가 포함된 딕셔너리
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        explanation = service.explain(drug_id, disease)
        
        if "error" in explanation:
            raise HTTPException(status_code=400, detail=explanation["error"])
        
        return explanation
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in explain_drug_disease: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/drugs")
async def get_all_drugs():
    """모든 약물 정보를 반환합니다."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        drugs = service.get_all_drugs()
        return {
            "drugs": drugs,
            "count": len(drugs)
        }
    
    except Exception as e:
        logger.error(f"Error in get_all_drugs: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/diseases")
async def get_all_diseases():
    """모든 질병 정보를 반환합니다."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        diseases = service.get_all_diseases()
        return {
            "diseases": diseases,
            "count": len(diseases)
        }
    
    except Exception as e:
        logger.error(f"Error in get_all_diseases: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/drugs/{drug_id}")
async def get_drug_info(drug_id: str):
    """특정 약물의 정보를 반환합니다."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        drug_info = service.get_drug_info(drug_id)
        
        if not drug_info:
            raise HTTPException(status_code=404, detail=f"Drug not found: {drug_id}")
        
        return drug_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_drug_info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/diseases/{disease_id}")
async def get_disease_info(disease_id: str):
    """특정 질병의 정보를 반환합니다."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        disease_info = service.get_disease_info(disease_id)
        
        if not disease_info:
            raise HTTPException(status_code=404, detail=f"Disease not found: {disease_id}")
        
        return disease_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_disease_info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/search/diseases")
async def search_diseases(q: str = Query(..., description="검색 쿼리")):
    """질병을 검색합니다."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        results = service.search_diseases(q)
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error(f"Error in search_diseases: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/stats")
async def get_service_stats():
    """서비스 통계를 반환합니다."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        graph_stats = service.get_graph_stats()
        return {
            "graph_stats": graph_stats,
            "service_status": "healthy"
        }
    
    except Exception as e:
        logger.error(f"Error in get_service_stats: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Drug Repurposing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
