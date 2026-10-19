from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_tfidf_similarity(resume_text: str, job_text: str) -> float:
    """
    Calculate semantic Cosine Similarity between resume and job description using TF-IDF.
    """
    if not resume_text.strip() or not job_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(similarity, 0.0, 1.0))
    except ValueError:
        return 0.0


def calculate_skill_overlap(resume_skills: List[str], job_skills: List[str]) -> Dict:
    """
    Calculate skill set overlap, matching, and missing requirements.
    """
    r_set = set(resume_skills)
    j_set = set(job_skills)

    matched = sorted(list(r_set.intersection(j_set)))
    missing = sorted(list(j_set - r_set))
    extra = sorted(list(r_set - j_set))

    total_required = len(j_set)
    coverage_ratio = (len(matched) / total_required) if total_required > 0 else 1.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "matched_count": len(matched),
        "total_required": total_required,
        "coverage_ratio": round(coverage_ratio, 4)
    }


def compute_ats_score(
    resume_text: str,
    job_text: str,
    resume_skills: List[str],
    job_skills: List[str],
    weights: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Comprehensive ATS score calculation combining:
    1. Skill Coverage (default: 55%)
    2. Semantic TF-IDF Text Similarity (default: 35%)
    3. Resume Length & Detail Fitness (default: 10%)
    """
    if weights is None:
        weights = {
            "skills": 0.55,
            "semantic": 0.35,
            "structure": 0.10
        }

    # 1. Semantic Similarity
    semantic_sim = calculate_tfidf_similarity(resume_text, job_text)

    # 2. Skill Overlap
    skill_stats = calculate_skill_overlap(resume_skills, job_skills)
    skill_coverage = skill_stats["coverage_ratio"]

    # 3. Structural Length fitness
    word_count = len(resume_text.split())
    # Standard 1-2 page resume is 300 to 1200 words
    if word_count < 100:
        length_score = 0.3
    elif 100 <= word_count < 300:
        length_score = 0.7
    elif 300 <= word_count <= 1500:
        length_score = 1.0
    else:
        length_score = 0.85

    overall_score = (
        (skill_coverage * weights["skills"]) +
        (semantic_sim * weights["semantic"]) +
        (length_score * weights["structure"])
    ) * 100.0

    overall_score = round(float(np.clip(overall_score, 0.0, 100.0)), 2)

    if overall_score >= 80.0:
        match_tier = "High Match"
        status_color = "green"
    elif overall_score >= 60.0:
        match_tier = "Moderate Match"
        status_color = "orange"
    else:
        match_tier = "Low Match"
        status_color = "red"

    return {
        "overall_score": overall_score,
        "match_tier": match_tier,
        "status_color": status_color,
        "breakdown": {
            "skill_coverage_score": round(skill_coverage * 100.0, 2),
            "semantic_similarity_score": round(semantic_sim * 100.0, 2),
            "structure_score": round(length_score * 100.0, 2),
            "word_count": word_count
        },
        "skill_analysis": skill_stats
    }
