import json
import re
from pathlib import Path
from typing import Dict, List, Set, Union


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "skills_taxonomy.json"


class SkillsExtractor:
    """Extracts domain-specific technical skills from unstructured text using a taxonomy."""

    def __init__(self, taxonomy: Union[str, Path, Dict[str, List[str]]] = None):
        if taxonomy is None:
            taxonomy = DEFAULT_TAXONOMY_PATH

        if isinstance(taxonomy, (str, Path)):
            path = Path(taxonomy)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.taxonomy = json.load(f)
            else:
                self.taxonomy = self._fallback_taxonomy()
        elif isinstance(taxonomy, dict):
            self.taxonomy = taxonomy
        else:
            self.taxonomy = self._fallback_taxonomy()

        self._compiled_patterns = self._build_patterns()

    def _fallback_taxonomy(self) -> Dict[str, List[str]]:
        return {
            "Machine Learning & AI": [
                "Python", "Machine Learning", "Deep Learning", "Scikit-Learn",
                "TensorFlow", "PyTorch", "Keras", "NLP", "Computer Vision",
                "OpenCV", "LLM", "LangChain", "RAG", "Transformers",
                "HuggingFace", "XGBoost", "Pandas", "NumPy", "MLOps", "MLflow"
            ],
            "Programming Languages": [
                "Python", "Java", "C++", "C#", "JavaScript", "TypeScript",
                "Go", "Rust", "SQL", "R", "Bash"
            ],
            "Web & Backend": [
                "FastAPI", "Flask", "Django", "Node.js", "React", "Next.js",
                "REST API", "GraphQL", "HTML", "CSS", "Tailwind CSS"
            ],
            "Cloud & DevOps": [
                "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD",
                "GitHub Actions", "Terraform", "Linux"
            ],
            "Databases": [
                "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite",
                "Snowflake", "Pinecone", "ChromaDB"
            ]
        }

    def _build_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns with boundary awareness for symbols like C++, C#, .NET."""
        patterns = {}
        for category, skills in self.taxonomy.items():
            for skill in skills:
                if skill in patterns:
                    continue
                escaped = re.escape(skill)
                # If skill starts or ends with non-alphanumeric, use boundary lookarounds
                pattern_str = rf"(?i)(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])"
                patterns[skill] = re.compile(pattern_str)
        return patterns

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text, returning matches categorized by taxonomy domain.
        """
        if not text:
            return {}

        found_skills: Set[str] = set()
        for skill_name, pattern in self._compiled_patterns.items():
            if pattern.search(text):
                found_skills.add(skill_name)

        categorized_matches: Dict[str, List[str]] = {}
        for category, skills in self.taxonomy.items():
            matches_in_cat = [s for s in skills if s in found_skills]
            if matches_in_cat:
                categorized_matches[category] = sorted(list(set(matches_in_cat)))

        return categorized_matches

    def extract_flat_skills(self, text: str) -> List[str]:
        """Return a sorted, deduplicated list of all detected skills."""
        cat_dict = self.extract_skills(text)
        all_skills = set()
        for skills in cat_dict.values():
            all_skills.update(skills)
        return sorted(list(all_skills))
