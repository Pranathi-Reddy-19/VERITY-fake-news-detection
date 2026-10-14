import re
from urllib.parse import urlparse

from src.rag import retrieve_articles
from src.web_search import search_web
from src.llm_verifier import (
    verify_claim_with_llm,
    answer_question_with_llm,
)
from src.gat_predict import predict_gat


# =========================================================
# SOURCE QUALITY
# =========================================================

HIGH_QUALITY_DOMAINS = {
    "gov.in",
    "nic.in",
    "gov",
    "edu",
    "org",
    "who.int",
    "un.org",
    "nasa.gov",
    "isro.gov.in",
    "rbi.org.in",
    "eci.gov.in",
    "pib.gov.in",
    "supremecourt.gov.in",
}

TRUSTED_NEWS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "ndtv.com",
    "timesofindia.indiatimes.com",
    "usatoday.com",
}

LOW_QUALITY_DOMAINS = {
    "facebook.com",
    "youtube.com",
    "reddit.com",
    "quora.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
}


# =========================================================
# HELPERS
# =========================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_confidence(value):
    value = safe_float(value)

    # Convert 0-1 confidence to percentage.
    if 0 < value <= 1:
        value *= 100

    return max(0.0, min(100.0, value))


def normalize_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_domain(url):
    if not url:
        return ""

    try:
        domain = urlparse(str(url)).netloc.lower()
        return domain.replace("www.", "")
    except Exception:
        return ""


def source_quality(url):
    domain = get_domain(url)

    if not domain:
        return 0.5

    if domain in LOW_QUALITY_DOMAINS:
        return 0.1

    for trusted in TRUSTED_NEWS:
        if domain == trusted or domain.endswith("." + trusted):
            return 0.9

    for trusted in HIGH_QUALITY_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return 1.0

    if domain.endswith(".gov") or domain.endswith(".gov.in"):
        return 1.0

    if domain.endswith(".edu"):
        return 0.9

    return 0.6


# =========================================================
# QUERY TYPE
# =========================================================

def detect_query_type(text):
    text = str(text).strip().lower()

    question_words = [
        "what ",
        "who ",
        "where ",
        "when ",
        "why ",
        "how ",
        "which ",
        "is ",
        "are ",
        "was ",
        "were ",
        "does ",
        "did ",
        "can ",
        "capital of",
    ]

    if text.endswith("?"):
        return "FACTUAL_QUESTION"

    for word in question_words:
        if text.startswith(word):
            return "FACTUAL_QUESTION"

    return "CLAIM"


# =========================================================
# WEB FILTERING
# =========================================================

def filter_web_results(results):
    filtered = []

    if not results:
        return filtered

    for result in results:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url", ""))
        title = str(result.get("title", ""))
        snippet = str(result.get("snippet", ""))

        quality = source_quality(url)

        if quality <= 0.1:
            continue

        if not title and not snippet:
            continue

        item = dict(result)

        item["url"] = url
        item["title"] = title
        item["snippet"] = snippet
        item["quality"] = quality

        filtered.append(item)

    return filtered


# =========================================================
# KEYWORDS
# =========================================================

STOPWORDS = {
    "the", "is", "a", "an", "of", "in", "on", "to",
    "for", "and", "or", "was", "were", "are", "this",
    "that", "with", "by", "from", "as", "at", "it",
    "its", "be", "has", "have", "had", "been",
}


def important_words(text):
    words = normalize_text(text).split()

    return {
        word
        for word in words
        if len(word) >= 3 and word not in STOPWORDS
    }


def overlap_score(query, evidence):
    q_words = important_words(query)
    e_words = important_words(evidence)

    if not q_words:
        return 0.0

    return len(q_words & e_words) / len(q_words)


# =========================================================
# HEURISTIC EVIDENCE RELATION
# =========================================================

def evidence_relation(claim, evidence):
    claim_words = important_words(claim)
    evidence_words = important_words(evidence)

    if not claim_words:
        return "UNCERTAIN", 0.0

    overlap = len(claim_words & evidence_words) / len(claim_words)

    text = normalize_text(evidence)

    contradiction_patterns = [
        "not the capital",
        "is not",
        "was not",
        "are not",
        "were not",
        "false claim",
        "false",
        "incorrect",
        "wrong",
        "no evidence",
        "debunked",
        "fact check",
        "factcheck",
        "misleading",
        "fabricated",
        "fake news",
        "untrue",
        "not true",
        "hoax",
    ]

    contradiction = any(
        pattern in text
        for pattern in contradiction_patterns
    )

    if contradiction and overlap >= 0.15:
        return "CONTRADICTS", overlap

    if overlap >= 0.35:
        return "SUPPORTS", overlap

    return "UNCERTAIN", overlap


# =========================================================
# ANALYZE WEB EVIDENCE
# =========================================================

def analyze_evidence(claim, results):
    supporting = []
    contradicting = []
    uncertain = []

    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        url = result.get("url", "")

        combined = f"{title} {snippet}"

        relation, overlap = evidence_relation(
            claim,
            combined,
        )

        item = dict(result)

        item["quality"] = safe_float(
            result.get("quality", source_quality(url))
        )

        item["overlap"] = overlap
        item["relation"] = relation
        item["source"] = "WEB"

        if relation == "SUPPORTS":
            supporting.append(item)

        elif relation == "CONTRADICTS":
            contradicting.append(item)

        else:
            uncertain.append(item)

    for group in (
        supporting,
        contradicting,
        uncertain,
    ):
        group.sort(
            key=lambda x: (
                x.get("quality", 0),
                x.get("overlap", 0),
            ),
            reverse=True,
        )

    return supporting, contradicting, uncertain


# =========================================================
# STRUCTURE RAG EVIDENCE
# =========================================================

def structure_rag_evidence(rag_results):
    evidence = []

    if not rag_results:
        return evidence

    for index, item in enumerate(rag_results, 1):
        if not isinstance(item, dict):
            continue

        similarity = safe_float(
            item.get(
                "similarity",
                item.get("score", 0.0),
            )
        )

        if similarity > 1:
            similarity /= 100.0

        similarity = max(0.0, min(1.0, similarity))

        title = item.get(
            "title",
            f"Retrieved Article {index}",
        )

        snippet = item.get(
            "snippet",
            item.get(
                "text",
                item.get("content", ""),
            ),
        )

        url = item.get("url", "")

        label = item.get(
            "label_name",
            item.get("label", "UNKNOWN"),
        )

        evidence.append({
            "id": index,
            "title": str(title),
            "url": str(url),
            "snippet": str(snippet),
            "text": str(
                item.get(
                    "text",
                    snippet,
                )
            ),
            "similarity": similarity,
            "similarity_percent": similarity * 100,
            "label": str(label),
            "label_name": str(label),
            "source": "RAG",
        })

    return evidence


# =========================================================
# STRUCTURE WEB EVIDENCE
# =========================================================

def structure_web_evidence(results):
    evidence = []

    if not results:
        return evidence

    for index, item in enumerate(results, 1):
        if not isinstance(item, dict):
            continue

        url = item.get("url", "")

        quality = safe_float(
            item.get(
                "quality",
                source_quality(url),
            )
        )

        overlap = safe_float(
            item.get("overlap", 0.0)
        )

        evidence.append({
            "id": index,
            "title": str(item.get("title", "")),
            "url": str(url),
            "snippet": str(item.get("snippet", "")),
            "quality": quality,
            "quality_percent": quality * 100,
            "overlap": overlap,
            "overlap_percent": overlap * 100,
            "relation": item.get(
                "relation",
                "UNCERTAIN",
            ),
            "source": "WEB",
        })

    return evidence


# =========================================================
# LLM VERIFICATION
# =========================================================

def llm_verify(claim, results):
    """
    IMPORTANT:
    The LLM sees a sorted list of evidence.

    Each evidence item keeps `original_index`,
    so LLM source indices can later be mapped
    back to the correct original web result.
    """

    evidence = []

    for original_index, result in enumerate(results):
        if not result.get("url"):
            continue

        evidence.append({
            "original_index": original_index,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
            "quality": result.get(
                "quality",
                source_quality(result.get("url", "")),
            ),
        })

    evidence.sort(
        key=lambda x: x.get("quality", 0),
        reverse=True,
    )

    evidence = evidence[:6]

    if not evidence:
        return {
            "result": {
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "reason": "No usable web evidence was found.",
                "source_indices": [],
            },
            "evidence_used": [],
        }

    print(
        "LLM: Verifying",
        len(evidence),
        "sources...",
    )

    llm_input = [
        {
            "title": item["title"],
            "url": item["url"],
            "snippet": item["snippet"],
        }
        for item in evidence
    ]

    try:
        result = verify_claim_with_llm(
            claim,
            llm_input,
        )

    except Exception as e:
        print("LLM verification failed:", e)

        result = {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reason": "The LLM verifier could not complete verification.",
            "source_indices": [],
        }

    if not isinstance(result, dict):
        result = {
            "verdict": "UNCERTAIN",
            "confidence": 0.0,
            "reason": "Invalid LLM verifier response.",
            "source_indices": [],
        }

    result["verdict"] = str(
        result.get("verdict", "UNCERTAIN")
    ).upper().strip()

    result["confidence"] = normalize_confidence(
        result.get("confidence", 0)
    )

    result["reason"] = str(
        result.get("reason", "")
    )

    valid_indices = []

    for index in result.get("source_indices", []):
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue

        if (
            1 <= index <= len(evidence)
            and index not in valid_indices
        ):
            valid_indices.append(index)

    result["source_indices"] = valid_indices

    print(
        "LLM VERDICT:",
        result["verdict"],
    )

    print(
        "LLM CONFIDENCE:",
        result["confidence"],
    )

    print(
        "LLM SOURCES:",
        result["source_indices"],
    )

    return {
        "result": result,
        "evidence_used": evidence,
    }


# =========================================================
# MAIN CLASSIFIER
# =========================================================

def classify_article(article):
    article = str(article).strip()

    if not article:
        return {
            "prediction": "INSUFFICIENT EVIDENCE",
            "confidence": 0.0,
            "query_type": "CLAIM",
            "evidence_status": "UNCERTAIN",
            "gat_prediction": "UNKNOWN",
            "gat_confidence": 0.0,
            "rag_results": [],
            "rag_evidence": [],
            "web_results": [],
            "web_evidence": [],
            "structured_web_evidence": [],
            "structured_evidence": [],
            "supporting": [],
            "contradicting": [],
            "uncertain": [],
            "llm_verdict": "UNCERTAIN",
            "llm_confidence": 0.0,
            "llm_reason": "No claim was submitted.",
            "llm_sources": [],
            "llm_selected_sources": [],
        }

    # =====================================================
    # QUERY TYPE
    # =====================================================

    query_type = detect_query_type(article)

    print("\n============================================================")
    print("VERITY ANALYSIS")
    print("============================================================")
    print("CLAIM:", article)
    print("TYPE:", query_type)

    # =====================================================
    # RAG
    # =====================================================

    try:
        rag_results = retrieve_articles(
            article,
            top_k=5,
        )
    except Exception as e:
        print("RAG ERROR:", e)
        rag_results = []

    if rag_results is None:
        rag_results = []

    print(
        f"RAG: Retrieved {len(rag_results)} related articles"
    )

    rag_evidence = structure_rag_evidence(
        rag_results
    )

    similarity_scores = [
        safe_float(item.get("similarity", 0))
        for item in rag_evidence
    ]

    similarity_scores_percent = [
        safe_float(item.get("similarity_percent", 0))
        for item in rag_evidence
    ]

    # =====================================================
    # WEB
    # =====================================================

    print("Searching live web evidence...")

    try:
        raw_results = search_web(
            article,
            max_results=10,
        )
    except Exception as e:
        print("WEB SEARCH ERROR:", e)
        raw_results = []

    if raw_results is None:
        raw_results = []

    print("WEB RESULTS:", len(raw_results))

    results = filter_web_results(
        raw_results
    )

    print(
        "FILTERED WEB RESULTS:",
        len(results),
    )

    # Add heuristic relation.
    for result in results:
        relation, overlap = evidence_relation(
            article,
            f"{result.get('title', '')} "
            f"{result.get('snippet', '')}",
        )

        result["relation"] = relation
        result["overlap"] = overlap

    # =====================================================
    # FACTUAL QUESTION
    # =====================================================

    if query_type == "FACTUAL_QUESTION":

        usable_sources = [
            result
            for result in results
            if result.get("url")
        ]

        usable_sources.sort(
            key=lambda x: source_quality(
                x.get("url", "")
            ),
            reverse=True,
        )

        usable_sources = usable_sources[:6]

        print(
            "LLM: Answering from",
            len(usable_sources),
            "sources...",
        )

        try:
            answer_result = answer_question_with_llm(
                article,
                usable_sources,
            )
        except Exception as e:
            print("QUESTION LLM ERROR:", e)

            answer_result = {
                "answer": "Insufficient evidence.",
                "confidence": 0.0,
                "source_indices": [],
            }

        answer = str(
            answer_result.get(
                "answer",
                "Insufficient evidence.",
            )
        ).strip()

        answer_confidence = normalize_confidence(
            answer_result.get("confidence", 0)
        )

        source_indices = answer_result.get(
            "source_indices",
            [],
        )

        answer_sources = []

        for index in source_indices:
            try:
                index = int(index)
            except (TypeError, ValueError):
                continue

            if 1 <= index <= len(usable_sources):
                answer_sources.append(
                    usable_sources[index - 1]
                )

        if not answer_sources and usable_sources:
            answer_sources = [
                usable_sources[0]
            ]

        structured_web = structure_web_evidence(
            usable_sources
        )

        answer_status = (
            "SUPPORTED"
            if answer_sources
            and answer.lower() != "insufficient evidence."
            else "UNCERTAIN"
        )

        return {
            "prediction": (
                "ANSWERED"
                if answer_status == "SUPPORTED"
                else "INSUFFICIENT EVIDENCE"
            ),

            "confidence": answer_confidence,

            "query_type": query_type,

            "evidence_status": answer_status,

            "answer": answer,

            "answer_source": (
                answer_sources[0]
                if answer_sources
                else {}
            ),

            "answer_sources": answer_sources,

            "supporting": answer_sources,

            "contradicting": [],

            "uncertain": [],

            "rag_results": rag_results,

            "retrieved_articles": rag_results,

            "retrieved": rag_results,

            "rag_evidence": rag_evidence,

            "rag_prediction": (
                rag_results[0].get(
                    "label_name",
                    "UNCERTAIN",
                )
                if rag_results
                else "UNCERTAIN"
            ),

            "rag_confidence": (
                similarity_scores[0] * 100
                if similarity_scores
                else 0.0
            ),

            "similarity_scores": similarity_scores,

            "similarity_scores_percent":
                similarity_scores_percent,

            "web_results": results,

            "web_evidence": structured_web,

            "structured_web_evidence":
                structured_web,

            "structured_evidence":
                rag_evidence + structured_web,

            "evidence":
                rag_evidence + structured_web,

            "llm_verdict": "ANSWER",

            "llm_confidence": answer_confidence,

            "llm_reason":
                "The answer was generated from retrieved web evidence.",

            "llm_sources": source_indices,

            "llm_selected_sources":
                answer_sources,

            "llm_result": answer_result,
        }

    # =====================================================
    # GAT
    # =====================================================

    try:
        gat_result = predict_gat(article)
    except Exception as e:
        print("GAT ERROR:", e)

        gat_result = {
            "prediction": "UNCERTAIN",
            "confidence": 0.0,
        }

    gat_prediction = str(
        gat_result.get(
            "prediction",
            "UNCERTAIN",
        )
    )

    gat_confidence = normalize_confidence(
        gat_result.get("confidence", 0)
    )

    print(
        f"GAT: {gat_prediction} "
        f"({gat_confidence:.2f}%)"
    )

    # =====================================================
    # HEURISTIC WEB GROUPS
    # =====================================================

    supporting_web, contradicting_web, uncertain_web = (
        analyze_evidence(
            article,
            results,
        )
    )

    # =====================================================
    # LLM
    # =====================================================

    llm_package = llm_verify(
        article,
        results,
    )

    llm_result = llm_package["result"]
    llm_evidence_used = llm_package["evidence_used"]

    llm_verdict = str(
        llm_result.get(
            "verdict",
            "UNCERTAIN",
        )
    ).upper().strip()

    llm_confidence = normalize_confidence(
        llm_result.get("confidence", 0)
    )

    llm_sources = llm_result.get(
        "source_indices",
        [],
    )

    # =====================================================
    # MAP LLM SOURCES CORRECTLY
    # =====================================================

    llm_selected_sources = []

    for llm_index in llm_sources:

        try:
            llm_index = int(llm_index)
        except (TypeError, ValueError):
            continue

        if not (
            1 <= llm_index <= len(llm_evidence_used)
        ):
            continue

        selected = dict(
            llm_evidence_used[llm_index - 1]
        )

        original_index = selected.get(
            "original_index"
        )

        if (
            isinstance(original_index, int)
            and 0 <= original_index < len(results)
        ):
            original = dict(
                results[original_index]
            )
        else:
            original = {}

        selected.update(original)

        selected["llm_source_index"] = llm_index
        selected["llm_selected"] = True
        selected["source"] = "WEB"

        if llm_verdict == "SUPPORTS":
            selected["relation"] = "SUPPORTS"

        elif llm_verdict == "CONTRADICTS":
            selected["relation"] = "CONTRADICTS"

        else:
            selected["relation"] = "UNCERTAIN"

        llm_selected_sources.append(
            selected
        )

    # =====================================================
    # FINAL EVIDENCE GROUPS
    # =====================================================

    if llm_verdict == "SUPPORTS":

        supporting = (
            llm_selected_sources
            if llm_selected_sources
            else supporting_web
        )

        contradicting = []
        uncertain = uncertain_web

    elif llm_verdict == "CONTRADICTS":

        supporting = []
        contradicting = (
            llm_selected_sources
            if llm_selected_sources
            else contradicting_web
        )
        uncertain = uncertain_web

    else:

        supporting = supporting_web
        contradicting = contradicting_web
        uncertain = uncertain_web

    # =====================================================
    # FINAL VERDICT
    # =====================================================

    if (
        llm_verdict == "SUPPORTS"
        and llm_selected_sources
    ):

        prediction = "REAL"
        confidence = llm_confidence
        evidence_status = "SUPPORTED"

    elif (
        llm_verdict == "CONTRADICTS"
        and llm_selected_sources
    ):

        prediction = "FAKE"
        confidence = llm_confidence
        evidence_status = "CONTRADICTED"

    else:

        prediction = "INSUFFICIENT EVIDENCE"
        confidence = llm_confidence
        evidence_status = "UNCERTAIN"

    # =====================================================
    # STRUCTURED WEB
    # =====================================================

    structured_web = structure_web_evidence(
        results
    )

    # Mark LLM-selected sources.
    for item in structured_web:
        for selected in llm_selected_sources:
            if (
                item.get("url")
                and item.get("url") == selected.get("url")
            ):
                item["llm_selected"] = True
                item["llm_source_index"] = selected.get(
                    "llm_source_index"
                )

                item["relation"] = selected.get(
                    "relation",
                    item.get("relation", "UNCERTAIN"),
                )

    structured_evidence = (
        rag_evidence
        + structured_web
    )

    # =====================================================
    # LOG
    # =====================================================

    print("\n------------------------------------------------------------")
    print("FINAL:", prediction)
    print("CONFIDENCE:", round(confidence, 2), "%")
    print("EVIDENCE:", evidence_status)
    print("RAG EVIDENCE:", len(rag_evidence))
    print("WEB EVIDENCE:", len(structured_web))
    print("LLM SELECTED:", len(llm_selected_sources))
    print("------------------------------------------------------------")

    # =====================================================
    # FINAL RETURN
    # =====================================================

    return {

        # FINAL
        "prediction": prediction,
        "confidence": confidence,
        "query_type": query_type,
        "evidence_status": evidence_status,

        # GAT
        "gat_prediction": gat_prediction,
        "gat_confidence": gat_confidence,
        "gat_result": gat_result,

        # RAG
        "rag_results": rag_results,
        "rag_evidence": rag_evidence,

        "rag_prediction": (
            rag_results[0].get(
                "label_name",
                "UNCERTAIN",
            )
            if rag_results
            else "UNCERTAIN"
        ),

        "rag_confidence": (
            similarity_scores[0] * 100
            if similarity_scores
            else 0.0
        ),

        "retrieved_articles": rag_results,
        "retrieved": rag_results,

        "similarity_scores": similarity_scores,

        "similarity_scores_percent":
            similarity_scores_percent,

        # WEB
        "web_results": results,

        "web_evidence": structured_web,

        "structured_web_evidence":
            structured_web,

        # EVIDENCE
        "structured_evidence":
            structured_evidence,

        "evidence":
            structured_evidence,

        "supporting": supporting,
        "contradicting": contradicting,
        "uncertain": uncertain,

        "supporting_count": len(supporting),
        "contradicting_count": len(contradicting),
        "uncertain_count": len(uncertain),

        # LLM
        "llm_verdict": llm_verdict,
        "llm_confidence": llm_confidence,

        "llm_reason": str(
            llm_result.get(
                "reason",
                "",
            )
        ),

        "llm_sources": llm_sources,

        "llm_selected_sources":
            llm_selected_sources,

        "llm_result": llm_result,

        # DEBUG
        "llm_evidence_used":
            llm_evidence_used,
    }


# =========================================================
# EXPLANATION
# =========================================================

def generate_explanation(article, result):

    prediction = result.get(
        "prediction",
        "INSUFFICIENT EVIDENCE",
    )

    confidence = normalize_confidence(
        result.get("confidence", 0)
    )

    query_type = result.get(
        "query_type",
        "CLAIM",
    )

    supporting = result.get(
        "supporting",
        [],
    )

    contradicting = result.get(
        "contradicting",
        [],
    )

    evidence_status = result.get(
        "evidence_status",
        "UNKNOWN",
    )

    # =====================================================
    # FACTUAL QUESTION
    # =====================================================

    if query_type == "FACTUAL_QUESTION":

        answer = str(
            result.get(
                "answer",
                "",
            )
        ).strip()

        sources = result.get(
            "answer_sources",
            [],
        )

        if answer and answer.lower() != "insufficient evidence.":

            return {
                "verdict": "ANSWER",
                "answer": answer,
                "confidence": confidence,
                "evidence_quality": "LIVE WEB",
                "reasoning": (
                    "The answer was generated using "
                    "retrieved live web evidence."
                ),
                "supporting_sources": sources,
                "contradicting_sources": [],
            }

        return {
            "verdict": "INSUFFICIENT EVIDENCE",
            "answer": (
                "I could not find sufficiently reliable "
                "evidence to answer this question."
            ),
            "confidence": 0.0,
            "evidence_quality": "INSUFFICIENT",
            "reasoning": (
                "The available retrieved evidence was "
                "not sufficient to establish an answer."
            ),
            "supporting_sources": [],
            "contradicting_sources": [],
        }

    # =====================================================
    # LLM REASONING
    # =====================================================

    llm_reason = str(
        result.get(
            "llm_reason",
            "",
        )
    ).strip()

    if llm_reason:
        reasoning = llm_reason

    elif prediction == "REAL":

        reasoning = (
            "The available verification evidence "
            "supports the submitted claim."
        )

    elif prediction == "FAKE":

        reasoning = (
            "The available verification evidence "
            "contradicts the submitted claim."
        )

    else:

        reasoning = (
            "The available evidence was insufficient "
            "to establish a definitive result."
        )

    return {
        "verdict": prediction,
        "confidence": confidence,
        "evidence_quality": evidence_status,
        "reasoning": reasoning,
        "supporting_sources": supporting,
        "contradicting_sources": contradicting,
        "evidence_count": (
            len(supporting)
            + len(contradicting)
        ),
    }