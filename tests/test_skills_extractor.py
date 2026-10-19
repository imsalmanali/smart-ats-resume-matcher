import unittest
from src.skills_extractor import SkillsExtractor


class TestSkillsExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = SkillsExtractor()

    def test_extract_basic_skills(self):
        sample = "Skilled in Python, PyTorch, Docker, and PostgreSQL databases."
        skills = self.extractor.extract_flat_skills(sample)
        self.assertIn("Python", skills)
        self.assertIn("PyTorch", skills)
        self.assertIn("Docker", skills)
        self.assertIn("PostgreSQL", skills)

    def test_special_characters_cpp_and_csharp(self):
        sample = "Strong experience in C++ and C# backend architectures."
        skills = self.extractor.extract_flat_skills(sample)
        self.assertIn("C++", skills)
        self.assertIn("C#", skills)

    def test_extract_empty(self):
        skills = self.extractor.extract_flat_skills("")
        self.assertEqual(skills, [])

    def test_categorized_extraction(self):
        sample = "Proficient in Python, AWS, and Git."
        cat = self.extractor.extract_skills(sample)
        self.assertIn("Machine Learning & AI", cat)
        self.assertIn("Cloud & DevOps", cat)
        self.assertIn("Software Engineering & Methodologies", cat)


if __name__ == "__main__":
    unittest.main()
