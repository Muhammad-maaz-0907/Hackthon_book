import logging
from typing import List, Dict, Any, Optional
import re
from enum import Enum

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"

class SafetyChecker:
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.MODERATE):
        self.safety_level = safety_level
        self._setup_safety_rules()

    def _setup_safety_rules(self):
        """Set up safety rules based on the safety level"""
        # Content injection patterns to detect potentially harmful instructions
        self.injection_patterns = [
            re.compile(r'ignore\s+previous\s+instructions', re.IGNORECASE),
            re.compile(r'forget\s+the\s+instructions', re.IGNORECASE),
            re.compile(r'act\s+as\s+if', re.IGNORECASE),
            re.compile(r'disregard\s+the\s+above', re.IGNORECASE),
            re.compile(r'now\s+unban\s+yourself', re.IGNORECASE),
            re.compile(r'jailbreak', re.IGNORECASE),
            re.compile(r'dan\s+mode', re.IGNORECASE),
        ]

        # Hallucination detection patterns
        self.hallucination_indicators = [
            re.compile(r'cannot find|no information|not mentioned|not specified', re.IGNORECASE),
            re.compile(r'i cannot|cannot determine|unable to find', re.IGNORECASE),
        ]

    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """Check if the query contains potentially unsafe instructions"""
        issues = []

        # Check for prompt injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(query):
                issues.append({
                    "type": "prompt_injection",
                    "message": "Query contains potential prompt injection patterns",
                    "pattern": pattern.pattern
                })

        # Check for hallucination-inducing queries
        if len(query.strip()) < 3:
            issues.append({
                "type": "query_too_short",
                "message": "Query is too short to be meaningful"
            })

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "safety_level": self.safety_level.value
        }

    def check_response_quality(self, response: str, sources: List[Dict[str, Any]],
                             min_sources: int = 1, min_confidence: float = 0.3) -> Dict[str, Any]:
        """Check the quality of the response and its sources"""
        issues = []
        quality_score = 1.0

        # Check if response indicates lack of information
        for pattern in self.hallucination_indicators:
            if pattern.search(response):
                issues.append({
                    "type": "potential_hallucination",
                    "message": "Response indicates lack of information which may suggest hallucination"
                })
                quality_score *= 0.5  # Reduce quality score

        # Check source confidence
        if sources:
            avg_confidence = sum(source.get('score', 0) for source in sources) / len(sources)
            if avg_confidence < min_confidence:
                issues.append({
                    "type": "low_source_confidence",
                    "message": f"Average source confidence ({avg_confidence:.2f}) is below minimum ({min_confidence})",
                    "confidence": avg_confidence
                })
                quality_score *= 0.7

            if len(sources) < min_sources:
                issues.append({
                    "type": "insufficient_sources",
                    "message": f"Number of sources ({len(sources)}) is below minimum ({min_sources})",
                    "source_count": len(sources)
                })
                quality_score *= 0.8
        else:
            # No sources provided - likely hallucination
            issues.append({
                "type": "no_sources",
                "message": "No sources provided - response may be hallucinated"
            })
            quality_score *= 0.3

        # Check for citation consistency
        response_lower = response.lower()
        for source in sources:
            if 'doc_path' in source:
                doc_path = source['doc_path'].lower()
                if doc_path not in response_lower and 'index' not in doc_path:
                    # This might be a false positive, but worth noting
                    pass  # Not necessarily an issue, as not all sources need to be explicitly mentioned

        # Additional quality checks based on safety level
        if self.safety_level == SafetyLevel.STRICT:
            # In strict mode, be more aggressive about potential hallucinations
            if "i don't know" not in response.lower() and "no information" not in response.lower():
                if not sources and "the book" in response_lower:
                    # If response mentions "the book" but has no sources, it might be hallucinating
                    issues.append({
                        "type": "potential_hallucination",
                        "message": "Response mentions 'the book' but has no sources - potential hallucination"
                    })
                    quality_score *= 0.4

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "quality_score": quality_score,
            "safety_level": self.safety_level.value
        }

    def validate_context_relevance(self, query: str, context: str) -> Dict[str, Any]:
        """Check if the context is relevant to the query"""
        query_lower = query.lower()
        context_lower = context.lower()

        # Check for basic relevance indicators
        query_words = set(query_lower.split()[:10])  # Take first 10 words as key terms
        context_words = set(context_lower.split()[:50])  # Take first 50 words from context

        # Calculate overlap
        overlap = len(query_words.intersection(context_words))
        total_query_words = len(query_words)

        relevance_score = overlap / total_query_words if total_query_words > 0 else 0

        is_relevant = relevance_score >= 0.1  # At least 10% overlap in key terms

        return {
            "is_relevant": is_relevant,
            "relevance_score": relevance_score,
            "overlap_count": overlap,
            "query_word_count": total_query_words
        }

    def filter_response(self, response: str, sources: List[Dict[str, Any]],
                       confidence: float) -> Dict[str, Any]:
        """Apply safety filtering to the response"""
        # Check if confidence is too low
        if confidence < 0.1:
            return {
                "response": "I don't have enough information in the book to answer that.",
                "sources": [],
                "confidence": confidence,
                "filtered": True,
                "reason": "low_confidence"
            }

        # Check for hallucination indicators in the response
        for pattern in self.hallucination_indicators:
            if pattern.search(response):
                # If the response already indicates lack of information, pass through
                # but potentially modify based on safety level
                if self.safety_level == SafetyLevel.STRICT:
                    return {
                        "response": "I don't have enough information in the book to answer that.",
                        "sources": [],
                        "confidence": confidence * 0.5,
                        "filtered": True,
                        "reason": "lack_of_information"
                    }

        # If everything is safe, return original response
        return {
            "response": response,
            "sources": sources,
            "confidence": confidence,
            "filtered": False
        }

    def check_content_safety(self, content: str) -> Dict[str, Any]:
        """Check if content contains unsafe elements"""
        issues = []

        # Check for code injection patterns
        code_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers like onclick, onload, etc.
        ]

        for pattern in code_patterns:
            if pattern.search(content):
                issues.append({
                    "type": "code_injection",
                    "message": "Content contains potential code injection patterns",
                    "pattern": pattern.pattern
                })

        # Check for potentially sensitive information
        sensitive_patterns = [
            re.compile(r'\b(?:password|secret|key|token|api_key)\b', re.IGNORECASE),
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Credit card numbers
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email (basic)
        ]

        for pattern in sensitive_patterns:
            if pattern.search(content):
                issues.append({
                    "type": "sensitive_information",
                    "message": "Content contains potential sensitive information",
                    "pattern": pattern.pattern
                })

        return {
            "is_safe": len(issues) == 0,
            "issues": issues
        }