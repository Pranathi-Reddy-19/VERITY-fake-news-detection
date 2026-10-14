# import os
# import json
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY = os.getenv("OPENROUTER_API_KEY")

# if not API_KEY:
#     raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

# client = OpenAI(
#     api_key=API_KEY,
#     base_url="https://openrouter.ai/api/v1"
# )


# MODEL = "openai/gpt-4o-mini"


# def verify_claim_with_llm(claim, evidence):

#     evidence_text = ""

#     for i, item in enumerate(evidence, 1):

#         title = item.get("title", "")
#         snippet = item.get("snippet", "")
#         url = item.get("url", "")

#         evidence_text += (
#             f"\nSOURCE {i}\n"
#             f"TITLE: {title}\n"
#             f"URL: {url}\n"
#             f"EVIDENCE: {snippet[:5000]}\n"
#         )

#     prompt = f"""
# You are a strict factual verification system.

# Your task is to verify the CLAIM using ONLY the supplied WEB EVIDENCE.

# Do NOT use your own prior knowledge.

# Do NOT decide based only on keyword overlap.

# A source SUPPORTS the claim only if the source evidence directly
# establishes that the claim is true.

# A source CONTRADICTS the claim only if the source evidence directly
# establishes that the claim is false.

# If the evidence is insufficient, ambiguous, unrelated, or does not
# directly establish whether the claim is true or false, return UNCERTAIN.

# IMPORTANT SOURCE RULES:

# 1. source_indices MUST contain the source numbers that directly support
#    or contradict the claim.

# 2. If verdict is SUPPORTS, source_indices MUST NOT be empty when there
#    is relevant evidence.

# 3. If verdict is CONTRADICTS, source_indices MUST NOT be empty when there
#    is relevant evidence.

# 4. Do NOT include sources merely because they contain similar words.

# 5. If no supplied source directly supports or contradicts the claim,
#    return UNCERTAIN with an empty source_indices list.

# 6. Use the exact source numbers shown in the evidence.

# 7. The reason must explain WHY the selected sources support or contradict
#    the claim. Do not simply repeat the verdict.

# Pay special attention to:

# - dates
# - locations
# - names
# - numbers
# - historical versus current information
# - negations
# - similar but different claims
# - exaggerated scientific claims
# - whether the source actually establishes the proposition

# CLAIM:
# {claim}

# WEB EVIDENCE:
# {evidence_text}

# Return ONLY valid JSON.

# Required format:

# {{
#     "verdict": "SUPPORTS",
#     "confidence": 0,
#     "reason": "Explain specifically why the evidence supports or contradicts the claim.",
#     "source_indices": [1]
# }}

# verdict must be exactly one of:

# SUPPORTS
# CONTRADICTS
# UNCERTAIN

# confidence must be a number from 0 to 100.

# source_indices must contain valid source numbers from the supplied
# WEB EVIDENCE.
# """

#     try:

#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are a strict evidence-based fact checker. "
#                         "Never invent evidence or source numbers."
#                     )
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0,
#             max_tokens=500
#         )

#         content = response.choices[0].message.content.strip()

#         if content.startswith("```"):
#             content = content.replace("```json", "")
#             content = content.replace("```", "")
#             content = content.strip()

#         result = json.loads(content)

#         # -------------------------------------------------
#         # VALIDATE RESULT
#         # -------------------------------------------------

#         verdict = result.get("verdict", "UNCERTAIN")

#         if verdict not in {
#             "SUPPORTS",
#             "CONTRADICTS",
#             "UNCERTAIN"
#         }:
#             verdict = "UNCERTAIN"

#         try:
#             confidence = float(
#                 result.get("confidence", 0)
#             )
#         except Exception:
#             confidence = 0.0

#         confidence = max(
#             0.0,
#             min(100.0, confidence)
#         )

#         reason = str(
#             result.get("reason", "")
#         ).strip()

#         source_indices = result.get(
#             "source_indices",
#             []
#         )

#         if not isinstance(source_indices, list):
#             source_indices = []

#         # Keep only valid source numbers
#         valid_indices = []

#         for index in source_indices:

#             if isinstance(index, int):
#                 if 1 <= index <= len(evidence):
#                     valid_indices.append(index)

#         source_indices = list(
#             dict.fromkeys(valid_indices)
#         )

#         # -------------------------------------------------
#         # SAFETY CHECK
#         # -------------------------------------------------

#         # If the LLM claims SUPPORTS/CONTRADICTS but provides
#         # no valid source, do not pretend the claim is verified.

#         if verdict in {
#             "SUPPORTS",
#             "CONTRADICTS"
#         } and not source_indices:

#             verdict = "UNCERTAIN"
#             confidence = min(
#                 confidence,
#                 50.0
#             )

#             reason = (
#                 "The supplied web evidence was not sufficient "
#                 "to identify a specific source that directly "
#                 "supports or contradicts the claim."
#             )

#         return {
#             "verdict": verdict,
#             "confidence": confidence,
#             "reason": reason,
#             "source_indices": source_indices
#         }

#     except Exception as e:

#         print(
#             "LLM VERIFIER ERROR:",
#             repr(e)
#         )

#         return {
#             "verdict": "UNCERTAIN",
#             "confidence": 0,
#             "reason": "LLM verification failed.",
#             "source_indices": []
#         }

# def answer_question_with_llm(question, evidence):

#     evidence_text = ""

#     for i, item in enumerate(evidence, 1):

#         title = item.get("title", "")
#         snippet = item.get("snippet", "")
#         url = item.get("url", "")

#         evidence_text += (
#             f"\nSOURCE {i}\n"
#             f"TITLE: {title}\n"
#             f"URL: {url}\n"
#             f"EVIDENCE: {snippet[:5000]}\n"
#         )

#     prompt = f"""
# You are a factual question answering system.

# Answer the user's question using ONLY the supplied web evidence.

# Do not use your own prior knowledge.

# Extract the specific answer requested by the question.

# Do not return the entire source article.

# If the evidence does not clearly answer the question, say:
# "Insufficient evidence."

# QUESTION:
# {question}

# WEB EVIDENCE:
# {evidence_text}

# Return ONLY valid JSON:

# {{
#     "answer": "",
#     "confidence": 0,
#     "source_indices": []
# }}
# """

#     try:

#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a concise evidence-based question answering system."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0,
#             max_tokens=300
#         )

#         content = response.choices[0].message.content.strip()

#         if content.startswith("```"):
#             content = content.replace("```json", "")
#             content = content.replace("```", "")
#             content = content.strip()

#         return json.loads(content)

#     except Exception as e:

#         print("LLM QUESTION ERROR:", repr(e))

#         return {
#             "answer": "Insufficient evidence.",
#             "confidence": 0,
#             "source_indices": []
#         }

import os
import json

from openai import OpenAI
from dotenv import load_dotenv


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing from .env"
    )


# ---------------------------------------------------------
# OPENROUTER CLIENT
# ---------------------------------------------------------

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


MODEL = "openai/gpt-4o-mini"


# ---------------------------------------------------------
# CLEAN LLM JSON
# ---------------------------------------------------------

def clean_json_response(content):

    content = content.strip()

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

    return content


# ---------------------------------------------------------
# VERIFY CLAIM
# ---------------------------------------------------------

def verify_claim_with_llm(claim, evidence):

    evidence_text = ""

    for i, item in enumerate(evidence, 1):

        title = item.get(
            "title",
            ""
        )

        snippet = item.get(
            "snippet",
            ""
        )

        url = item.get(
            "url",
            ""
        )

        evidence_text += (
            f"\nSOURCE {i}\n"
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"EVIDENCE: {snippet[:5000]}\n"
        )

    prompt = f"""
You are a strict factual verification system.

Your task is to verify the CLAIM using ONLY the supplied WEB EVIDENCE.

Do NOT use your own prior knowledge.

Do NOT decide based only on keyword overlap.

A source SUPPORTS the claim only if the source evidence directly
establishes that the claim is true.

A source CONTRADICTS the claim only if the source evidence directly
establishes that the claim is false.

If the evidence is insufficient, ambiguous, unrelated, or does not
directly establish whether the claim is true or false, return UNCERTAIN.

IMPORTANT SOURCE RULES:

1. source_indices MUST contain the source numbers that directly
   support or contradict the claim.

2. If verdict is SUPPORTS, identify the source(s) that support it.

3. If verdict is CONTRADICTS, identify the source(s) that contradict it.

4. Do NOT include sources merely because they contain similar words.

5. If no supplied source directly supports or contradicts the claim,
   return UNCERTAIN with an empty source_indices list.

6. Use only source numbers that actually exist.

7. The reason must explain WHY the selected evidence supports or
   contradicts the claim.

8. Do not invent facts that are not present in the supplied evidence.

Pay special attention to:

- dates
- locations
- names
- numbers
- historical versus current information
- negations
- similar but different claims
- exaggerated scientific claims
- whether the source actually establishes the proposition

CLAIM:
{claim}

WEB EVIDENCE:
{evidence_text}

Return ONLY valid JSON.

Required format:

{{
    "verdict": "SUPPORTS",
    "confidence": 0,
    "reason": "Explain specifically why the evidence supports or contradicts the claim.",
    "source_indices": [1]
}}

verdict must be exactly one of:

SUPPORTS
CONTRADICTS
UNCERTAIN

confidence must be a number from 0 to 100.

source_indices must contain valid source numbers from the supplied
WEB EVIDENCE.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evidence-based fact checker. "
                        "Never invent evidence or source numbers."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=500
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise ValueError(
                "Empty response from LLM"
            )

        content = clean_json_response(
            content
        )

        result = json.loads(
            content
        )

        # -------------------------------------------------
        # VERDICT
        # -------------------------------------------------

        verdict = result.get(
            "verdict",
            "UNCERTAIN"
        )

        if verdict not in {
            "SUPPORTS",
            "CONTRADICTS",
            "UNCERTAIN"
        }:

            verdict = "UNCERTAIN"

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        try:

            confidence = float(
                result.get(
                    "confidence",
                    0
                )
            )

        except Exception:

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                100.0,
                confidence
            )
        )

        # -------------------------------------------------
        # REASON
        # -------------------------------------------------

        reason = str(
            result.get(
                "reason",
                ""
            )
        ).strip()

        # -------------------------------------------------
        # SOURCE INDICES
        # -------------------------------------------------

        source_indices = result.get(
            "source_indices",
            []
        )

        if not isinstance(
            source_indices,
            list
        ):

            source_indices = []

        valid_indices = []

        for index in source_indices:

            if isinstance(index, int):

                if 1 <= index <= len(evidence):

                    valid_indices.append(
                        index
                    )

        source_indices = list(
            dict.fromkeys(
                valid_indices
            )
        )

        # -------------------------------------------------
        # SAFETY VALIDATION
        # -------------------------------------------------

        if (
            verdict in {
                "SUPPORTS",
                "CONTRADICTS"
            }
            and not source_indices
        ):

            verdict = "UNCERTAIN"

            confidence = min(
                confidence,
                50.0
            )

            reason = (
                "The supplied web evidence was not sufficient "
                "to identify a specific source that directly "
                "supports or contradicts the claim."
            )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
            "source_indices": source_indices
        }

    except Exception as e:

        print(
            "LLM VERIFIER ERROR:",
            repr(e)
        )

        return {
            "verdict": "UNCERTAIN",
            "confidence": 0,
            "reason": "LLM verification failed.",
            "source_indices": []
        }


# ---------------------------------------------------------
# ANSWER FACTUAL QUESTION
# ---------------------------------------------------------

def answer_question_with_llm(
    question,
    evidence
):

    evidence_text = ""

    for i, item in enumerate(
        evidence,
        1
    ):

        title = item.get(
            "title",
            ""
        )

        snippet = item.get(
            "snippet",
            ""
        )

        url = item.get(
            "url",
            ""
        )

        evidence_text += (
            f"\nSOURCE {i}\n"
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"EVIDENCE: {snippet[:5000]}\n"
        )

    prompt = f"""
You are a factual question answering system.

Answer the user's question using ONLY the supplied web evidence.

Do not use your own prior knowledge.

Extract the specific answer requested by the question.

Do not return the entire source article.

If the evidence does not clearly answer the question, say:
"Insufficient evidence."

QUESTION:
{question}

WEB EVIDENCE:
{evidence_text}

Return ONLY valid JSON:

{{
    "answer": "",
    "confidence": 0,
    "source_indices": []
}}

confidence must be a number from 0 to 100.

source_indices must contain the source numbers used to produce
the answer.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise evidence-based "
                        "question answering system."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=300
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise ValueError(
                "Empty response from LLM"
            )

        content = clean_json_response(
            content
        )

        result = json.loads(
            content
        )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        try:

            confidence = float(
                result.get(
                    "confidence",
                    0
                )
            )

        except Exception:

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                100.0,
                confidence
            )
        )

        # -------------------------------------------------
        # SOURCE INDICES
        # -------------------------------------------------

        source_indices = result.get(
            "source_indices",
            []
        )

        if not isinstance(
            source_indices,
            list
        ):

            source_indices = []

        valid_indices = []

        for index in source_indices:

            if isinstance(index, int):

                if 1 <= index <= len(evidence):

                    valid_indices.append(
                        index
                    )

        result["confidence"] = confidence

        result["source_indices"] = list(
            dict.fromkeys(
                valid_indices
            )
        )

        return result

    except Exception as e:

        print(
            "LLM QUESTION ERROR:",
            repr(e)
        )

        return {
            "answer": "Insufficient evidence.",
            "confidence": 0,
            "source_indices": []
        }