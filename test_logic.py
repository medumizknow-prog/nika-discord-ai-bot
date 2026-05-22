
from src.text_utils import normalize_compare_text, is_too_similar

def test_normalization():
    print("Testing normalization...")
    assert normalize_compare_text("Привет, мир!") == "привет мир"
    assert normalize_compare_text("Кот и Ёлка") == "кот и елка"
    assert normalize_compare_text("  Space   test  ") == "space test"
    print("Normalization: OK")

def test_similarity():
    print("Testing similarity...")
    assert is_too_similar("привет как дела", "привет как дела", 0.9)
    assert is_too_similar("привет как дела", "Привет, как дела?", 0.9)
    assert not is_too_similar("привет как дела", "пока как дела", 0.9)
    print("Similarity: OK")

if __name__ == "__main__":
    test_normalization()
    test_similarity()
