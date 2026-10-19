from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.skills_extractor import SkillsExtractor
from src.matcher import compute_ats_score
from src.report import generate_recommendations, generate_summary_text

app = FastAPI(
    title="Smart ATS Resume Matcher API",
    description="Production-ready REST API for evaluating candidate resume relevance using NLP & Machine Learning.",
    version="1.0.0"
)

extractor = SkillsExtractor()


# --- Request & Response Models ---

class TextPayload(BaseModel):
    text: str = Field(..., min_length=5, description="Input raw text from resume or job description.")


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="Extracted resume text.")
    job_description: str = Field(..., min_length=10, description="Target job description text.")
    candidate_name: Optional[str] = "Candidate"


class BatchCandidateItem(BaseModel):
    candidate_id: str
    candidate_name: str
    resume_text: str


class BatchMatchRequest(BaseModel):
    job_description: str = Field(..., min_length=10)
    candidates: List[BatchCandidateItem] = Field(..., min_items=1)


class SkillExtractionResponse(BaseModel):
    categorized_skills: Dict[str, List[str]]
    total_skills_count: int
    flat_skills: List[str]


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Smart ATS Resume Matcher API",
        "version": "1.0.0"
    }


@app.post("/api/v1/extract-skills", response_model=SkillExtractionResponse)
def extract_skills_endpoint(payload: TextPayload):
    try:
        categorized = extractor.extract_skills(payload.text)
        flat = extractor.extract_flat_skills(payload.text)
        return {
            "categorized_skills": categorized,
            "total_skills_count": len(flat),
            "flat_skills": flat
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/match")
def match_resume_endpoint(request: MatchRequest):
    try:
        resume_skills = extractor.extract_flat_skills(request.resume_text)
        job_skills = extractor.extract_flat_skills(request.job_description)

        ats_result = compute_ats_score(
            resume_text=request.resume_text,
            job_text=request.job_description,
            resume_skills=resume_skills,
            job_skills=job_skills
        )

        recommendations = generate_recommendations(ats_result)
        summary = generate_summary_text(ats_result, candidate_name=request.candidate_name)

        return {
            "candidate_name": request.candidate_name,
            "overall_score": ats_result["overall_score"],
            "match_tier": ats_result["match_tier"],
            "status_color": ats_result["status_color"],
            "breakdown": ats_result["breakdown"],
            "skill_analysis": ats_result["skill_analysis"],
            "recommendations": recommendations,
            "executive_summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch-match")
def batch_match_endpoint(request: BatchMatchRequest):
    try:
        job_skills = extractor.extract_flat_skills(request.job_description)
        ranked_candidates = []

        for cand in request.candidates:
            cand_skills = extractor.extract_flat_skills(cand.resume_text)
            ats_result = compute_ats_score(
                resume_text=cand.resume_text,
                job_text=request.job_description,
                resume_skills=cand_skills,
                job_skills=job_skills
            )
            ranked_candidates.append({
                "candidate_id": cand.candidate_id,
                "candidate_name": cand.candidate_name,
                "overall_score": ats_result["overall_score"],
                "match_tier": ats_result["match_tier"],
                "skill_coverage_score": ats_result["breakdown"]["skill_coverage_score"],
                "semantic_similarity_score": ats_result["breakdown"]["semantic_similarity_score"],
                "matched_skills_count": ats_result["skill_analysis"]["matched_count"],
                "matched_skills": ats_result["skill_analysis"]["matched_skills"]
            })

        # Sort descending by overall_score
        ranked_candidates.sort(key=lambda x: x["overall_score"], reverse=True)

        for rank, cand in enumerate(ranked_candidates, start=1):
            cand["rank"] = rank

        return {
            "total_candidates": len(ranked_candidates),
            "job_required_skills": job_skills,
            "rankings": ranked_candidates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
