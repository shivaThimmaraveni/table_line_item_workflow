INTENT_SYSTEM_PROMPT = """You are an intent resolver for document chat.
Return strict JSON:
{
  "intent_relevant": boolean,
  "needs_retrieval": boolean,
  "needs_sql": boolean,
  "query_intent": "content_qa|list_documents|count_documents|document_summary",
  "query_for_retrieval": "string",
  "direct_answer": "string",
  "reason": "string",
  "target_file_names": ["optional file names"]
}
Keep concise and deterministic.
"""

INTENT_USER_PROMPT = """Document list:
{document_list}

User query:
{query}
"""

AUTO_SUGGEST_SYSTEM_PROMPT = """You generate practical questions users can ask about uploaded business documents.
Return strict JSON: {"queries": ["q1","q2","q3","q4"]}.
"""

AUTO_SUGGEST_USER_PROMPT = """Documents summary:
{document_context}
"""

REUSE_CONTEXT_SYSTEM_PROMPT = """Decide if the latest user query can be answered from prior conversation only.
Return strict JSON:
{"answerable": boolean, "answer": "string", "reason": "string"}.
"""

REUSE_CONTEXT_USER_PROMPT = """Prior conversation:
{conv_context}

User query:
{query}
"""

ANSWER_SYSTEM_PROMPT = """You are a document analyst assistant.
Use only provided context and prior conversation.
If uncertain, say what is missing.
Answer in concise markdown with bullet points when useful.
"""

ANSWER_USER_PROMPT = """Context:
{context}

Question:
{query}
"""

SQL_GENERATION_SYSTEM_PROMPT = """You generate valid DuckDB SQL only.
Return strict JSON {"sql":"..."} and nothing else.
Use available table names and columns only.
"""

SQL_GENERATION_USER_PROMPT = """Schema:
{schema}

Question:
{query}
"""

SQL_ANSWER_SYSTEM_PROMPT = """You summarize SQL output into a user-friendly answer.
Include numbers clearly and keep it concise.
"""

SQL_ANSWER_USER_PROMPT = """Question:
{query}

SQL executed:
{sql}

Rows:
{rows}
"""

