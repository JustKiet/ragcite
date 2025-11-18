from typing import Optional

from langdetect import LangDetectException, detect  # type: ignore

from ragcite.core.interfaces.language_detector import BaseLanguageDetector


class LangDetectLanguageDetector(BaseLanguageDetector):
    """Language detector using the langdetect library"""

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the given text. Returns ISO 639-1 language code.

        :param str text: The text to analyze.
        :return: The detected language code (e.g., 'en' for English) or None if detection fails.
        :rtype: Optional[str]
        """
        try:
            cleaned = text.strip()
            if not cleaned:
                return None
            return detect(cleaned)  # type: ignore
        except LangDetectException:
            return None

    def are_same_language(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are in the same language

        :param str text1: The first text to analyze.
        :param str text2: The second text to analyze.
        :return: True if both texts are in the same language, False otherwise.
        :rtype: bool
        """
        lang1 = self.detect(text1)
        lang2 = self.detect(text2)

        # If either is unknown, assume different languages (conservative approach)
        if lang1 is None or lang2 is None:
            return False

        return lang1 == lang2
