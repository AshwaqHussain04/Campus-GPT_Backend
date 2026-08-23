"""
Canned responses for MOCK_MODE=true, used during frontend development
to avoid hitting MongoDB Atlas / Gemini API on every request.
"""

MOCK_RESPONSES = {
    "campus_documents": {
        "answer": (
            "The Central Library is open Monday to Saturday from 8:00 AM to 8:00 PM, "
            "and remains closed on Sundays and public holidays. During examination periods, "
            "weekday hours are extended until 11:00 PM."
        ),
        "source": "campus_documents",
        "sources_used": ["campus_info.txt"],
    },
    "general_knowledge": {
        "answer": (
            "Note: this is not from official campus documents.\n\n"
            "This is a mocked general-knowledge answer for frontend testing purposes."
        ),
        "source": "general_knowledge",
        "sources_used": [],
    },
}


def get_mock_query_response(question: str) -> dict:
    """
    Alternates response type based on a simple keyword check so you can
    trigger either UI state on demand while testing.
    Include 'mock:general' in your question to force the fallback-style response.
    """
    if "mock:general" in question.lower():
        return MOCK_RESPONSES["general_knowledge"]
    return MOCK_RESPONSES["campus_documents"]


MOCK_UPLOAD_RESPONSE = {
    "filename": "mock_file.pdf",
    "chunks_ingested": True,
    "message": "Uploaded and ingested 1 new/changed file(s). (MOCK MODE — nothing was actually written to Mongo/Gemini)",
}