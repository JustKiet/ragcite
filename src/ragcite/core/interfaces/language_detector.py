from abc import ABC, abstractmethod
from typing import Optional


class BaseLanguageDetector(ABC):
    @abstractmethod
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the given text. Returns ISO 639-1 language code.

        :param str text: The text to analyze.
        :return: The detected language code (e.g., 'en' for English) or None if detection fails.
        :rtype: Optional[str]
        """
        pass

    @abstractmethod
    def are_same_language(self, text1: str, text2: str) -> bool:
        """Check if two texts are in the same language.

        :param str text1: The first text to analyze.
        :param str text2: The second text to analyze.

        :return: True if both texts are in the same language, False otherwise.
        :rtype: bool
        """
        pass


class AsyncBaseLanguageDetector(ABC):
    @abstractmethod
    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the given text. Returns ISO 639-1 language code.

        :param str text: The text to analyze.
        :return: The detected language code (e.g., 'en' for English) or None if detection fails.
        :rtype: Optional[str]
        """
        pass

    @abstractmethod
    async def are_same_language(self, text1: str, text2: str) -> bool:
        """Asynchronously check if two texts are in the same language.

        :param str text1: The first text to analyze.
        :param str text2: The second text to analyze.

        :return: True if both texts are in the same language, False otherwise.
        :rtype: bool
        """
        pass
