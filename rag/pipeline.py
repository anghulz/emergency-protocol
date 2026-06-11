"""RAGPipeline: ties retrieval, context assembly, and the LLM together.

This module owns the chat completion call and all of its error
handling, so the UI layer never has to know about API exceptions.
"""

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from .prompts import SYSTEM_PROMPT, USER_PROMPT, build_context
from .store import DocumentStore

CHAT_MODEL = "gpt-4o-mini"


class RAGPipeline:
    """Answers questions grounded in the DocumentStore's contents."""

    def __init__(self, store: DocumentStore, api_key: str, top_k: int = 4):
        """
        Args:
            store: the DocumentStore to retrieve context from.
            api_key: OpenAI API key for chat completions.
            top_k: how many chunks to retrieve per question.
        """
        self.store = store
        self.top_k = top_k
        self._client = OpenAI(api_key=api_key, timeout=45)

    def answer(self, question: str) -> dict:
        """Run the full RAG loop for one question.

        Args:
            question: the user's natural-language question.

        Returns:
            A dict with:
              "answer":  the model's response text (or an error message),
              "chunks":  the retrieved chunks used as context,
              "ok":      False if an API error occurred.
        """
        question = question.strip()
        if not question:
            return {"answer": "Please enter a question.", "chunks": [], "ok": False}

        if self.store.chunk_count == 0:
            return {
                "answer": (
                    "No documents are loaded yet. Add documents in the "
                    "Document Library before asking questions."
                ),
                "chunks": [],
                "ok": False,
            }

        try:
            chunks = self.store.search(question, k=self.top_k)
            context = build_context(chunks)

            response = self._client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=0.2,  # low temperature: precision over creativity
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                    {"role": "user", "content": USER_PROMPT.format(question=question)},
                ],
            )
            answer = response.choices[0].message.content
            return {"answer": answer, "chunks": chunks, "ok": True}

        except AuthenticationError:
            msg = (
                "The OpenAI API key is missing or invalid. Set the "
                "OPENAI_API_KEY secret and reload the app."
            )
        except RateLimitError:
            msg = "The API rate limit was hit. Wait a moment and try again."
        except APITimeoutError:
            msg = "The request to the language model timed out. Try again."
        except APIConnectionError:
            msg = "Could not reach the OpenAI API. Check the network connection."
        except APIError as exc:
            msg = f"The OpenAI API returned an error: {exc.__class__.__name__}."

        return {"answer": msg, "chunks": [], "ok": False}
