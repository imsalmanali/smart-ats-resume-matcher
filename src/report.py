from typing import Dict, List


def generate_recommendations(ats_result: Dict) -> List[str]:
    """
    Generate actionable advice to help candidate optimize resume for ATS.
    """
    recommendations = []
    overall_score = ats_result.get("overall_score", 0.0)
    breakdown = ats_result.get("breakdown", {})
    skill_analysis = ats_result.get("skill_analysis", {})
    missing_skills = skill_analysis.get("missing_skills", [])
    word_count = breakdown.get("word_count", 0)

    # 1. Missing skills advice
    if missing_skills:
        top_missing = missing_skills[:5]
        recommendations.append(
            f"Add missing required skills: Consider highlighting experience with {', '.join(top_missing)} in your projects or summary."
        )

    # 2. Semantic alignment advice
    semantic_score = breakdown.get("semantic_similarity_score", 0.0)
    if semantic_score < 35.0:
        recommendations.append(
            "Low contextual vocabulary alignment: Mirror key phrases, tools, and technical terms directly from the job description."
        )

    # 3. Length & formatting advice
    if word_count < 250:
        recommendations.append(
            "Resume length is short: Expand on impact metrics, quantitative results, and tools used in each role (aim for 350-700 words)."
        )
    elif word_count > 1200:
        recommendations.append(
            "Resume length is very long: Tighten bullet points to ensure key technical highlights are easily readable in under 2 pages."
        )

    # 4. Success congratulations
    if overall_score >= 80.0:
        recommendations.append(
            "Strong candidate match! Ensure formatting is clean and links to GitHub or portfolio are active."
        )

    return recommendations


def generate_summary_text(ats_result: Dict, candidate_name: str = "Candidate") -> str:
    """Format ATS result into an executive summary."""
    score = ats_result["overall_score"]
    tier = ats_result["match_tier"]
    matched_count = ats_result["skill_analysis"]["matched_count"]
    total_req = ats_result["skill_analysis"]["total_required"]

    return (
        f"Evaluation for {candidate_name}:\n"
        f"Overall ATS Match: {score}% ({tier})\n"
        f"Skills Overlap: {matched_count}/{total_req} critical competencies verified.\n"
        f"Semantic Content Match: {ats_result['breakdown']['semantic_similarity_score']}%"
    )
