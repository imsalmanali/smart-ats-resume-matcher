import unittest
import io
from src.parser import clean_text, extract_text


class TestParser(unittest.TestCase):
    def test_clean_text_spaces_and_newlines(self):
        raw = "Hello   world!\r\n\r\n\r\nThis   is a    test.   "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Hello world!\n\nThis is a test.")

    def test_clean_text_empty(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")

    def test_extract_text_from_raw_string(self):
        text = "Experienced Senior Python Engineer."
        extracted = extract_text(text)
        self.assertEqual(extracted, text)

    def test_extract_text_from_bytes(self):
        raw_bytes = b"Skills: Docker, Kubernetes, AWS."
        extracted = extract_text(raw_bytes, filename="resume.txt")
        self.assertEqual(extracted, "Skills: Docker, Kubernetes, AWS.")


if __name__ == "__main__":
    unittest.main()
