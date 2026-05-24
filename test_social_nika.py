import unittest
from src.text_utils import normalize_compare_text, is_too_similar

class TestSocialNika(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_compare_text("Привет, как дела? Ёж!"), "привет как дела еж")
        self.assertEqual(normalize_compare_text("  Много   пробелов  "), "много пробелов")
        self.assertEqual(normalize_compare_text("!@#$%^&*()"), "")

    def test_similarity(self):
        self.assertTrue(is_too_similar("Привет", "привет"))
        self.assertTrue(is_too_similar("Как дела", "как дела?"))
        self.assertFalse(is_too_similar("Привет", "Пока"))
        self.assertTrue(is_too_similar("Очень длинная фраза которая почти совпадает", "Очень длинная фраза которая почти совпадает!!!"))

if __name__ == "__main__":
    unittest.main()
