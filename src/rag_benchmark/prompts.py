"""Shared generation prompts (kept out of sdk/ to avoid circular imports)."""

ANSWER_PROMPT = """Answer the question using only the provided context.
Prefer a short, direct answer (entity name, date, number, or yes/no) when that matches the question.
Put that short answer on the FIRST line by itself. Do not hedge if the answer is clearly stated.
If the context is insufficient, say you do not have enough information.

Question: {question}

Context:
{context}

Answer:"""
