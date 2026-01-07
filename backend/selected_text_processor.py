import logging
from typing import Dict, Any
import os
import uuid
from openai import AsyncOpenAI
import tiktoken
from safety_checker import SafetyChecker, SafetyLevel

logger = logging.getLogger(__name__)

class SelectedTextProcessor:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize tokenizer for context management
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Initialize safety checker
        self.safety_checker = SafetyChecker(safety_level=SafetyLevel.MODERATE)

    async def process(self, user_message: str, selected_text: str) -> Dict[str, Any]:
        """Process user query against selected text only"""
        conversation_id = str(uuid.uuid4())

        # Check query safety first
        query_safety = self.safety_checker.check_query_safety(user_message)
        if not query_safety["is_safe"]:
            logger.warning(f"Unsafe query detected in selected text mode: {query_safety['issues']}")
            return {
                "response": "I cannot process this query as it appears to contain unsafe instructions.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

        # Check if selected text is sufficient to answer the question
        if not selected_text or len(selected_text.strip()) < 10:
            return {
                "response": "The selected text is insufficient to answer your question. Please select more text.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

        # Check content safety in selected text
        content_safety = self.safety_checker.check_content_safety(selected_text)
        if not content_safety["is_safe"]:
            logger.warning(f"Unsafe content detected in selected text: {content_safety['issues']}")
            return {
                "response": "The selected text contains potentially unsafe content and cannot be processed.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

        # Build the prompt with selected text and user query
        prompt = self._build_selected_text_prompt(selected_text, user_message)

        # Generate response using OpenAI
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant for the Humanoid Robotics Book. "
                            "Answer questions based only on the provided selected text. "
                            "If the selected text doesn't contain the information to answer the question, "
                            "say 'The selected text is insufficient to answer that question.' "
                            "Do not use any external knowledge or information beyond what is provided in the selected text."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Apply safety filtering to the response
            filtered_result = self.safety_checker.filter_response(response_text,
                [{
                    "type": "selected_text",
                    "content_preview": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
                    "length": len(selected_text)
                }],
                0.8)  # Default confidence for selected text mode

            return {
                "response": filtered_result["response"],
                "sources": filtered_result["sources"],
                "conversation_id": conversation_id,
                "confidence": filtered_result["confidence"]
            }

        except Exception as e:
            logger.error(f"Error processing selected text: {str(e)}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "sources": [],
                "conversation_id": conversation_id,
                "confidence": 0.0
            }

    def _build_selected_text_prompt(self, selected_text: str, query: str) -> str:
        """Build the prompt for selected text processing"""
        prompt = (
            f"Selected Text:\n{selected_text}\n\n"
            f"Question: {query}\n\n"
            f"Instructions:\n"
            f"- Answer the question based ONLY on the provided selected text\n"
            f"- If the selected text doesn't contain the information to answer the question, say 'The selected text is insufficient to answer that question.'\n"
            f"- Do not use any external knowledge\n"
            f"- Be concise but comprehensive in your response"
        )

        return prompt

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))