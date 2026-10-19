import unittest
from src.matcher import calculate_tfidf_similarity, calculate_skill_overlap, compute_ats_score


class TestMatcher(unittest.TestCase):
    def test_calculate_skill_overlap(self):
        resume_skills = ["Python", "Docker", "Git", "PostgreSQL"]
        job_skills = ["Python", "Docker", "Kubernetes", "AWS"]

        overlap = calculate_skill_overlap(resume_skills, job_skills)
        self.assertEqual(overlap["matched_skills"], ["Docker", "Python"])
        self.assertEqual(overlap["missing_skills"], ["AWS", "Kubernetes"])
        self.assertEqual(overlap["extra_skills"], ["Git", "PostgreSQL"])
        self.assertEqual(overlap["matched_count"], 2)
        self.assertEqual(overlap["total_required"], 4)
        self.assertEqual(overlap["coverage_ratio"], 0.5)

    def test_tfidf_identical_texts(self):
        text = "Machine Learning Engineer with Python PyTorch and Scikit-Learn."
        similarity = calculate_tfidf_similarity(text, text)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_compute_ats_score_high_match(self):
        text = "Senior Machine Learning Engineer specializing in Python, PyTorch, Scikit-Learn, Docker and AWS cloud deployments."
        skills = ["Python", "PyTorch", "Scikit-Learn", "Docker", "AWS"]

        result = compute_ats_score(
            resume_text=text,
            job_text=text,
            resume_skills=skills,
            job_skills=skills
        )
        self.assertGreaterEqual(result["overall_score"], 80.0)
        self.assertEqual(result["match_tier"], "High Match")
        self.assertEqual(result["status_color"], "green")


if __name__ == "__main__":
    unittest.main()
