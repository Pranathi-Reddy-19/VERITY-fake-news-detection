import os
import requests
from dotenv import load_dotenv

load_dotenv()

def search_web(query, max_results=8):
    query = str(query).strip()

    if not query:
        return []

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        print("ERROR: TAVILY_API_KEY is missing")
        return []

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        results = []

        answer = data.get("answer")

        if answer:
            results.append({
                "title": "Tavily Search Answer",
                "url": "",
                "snippet": answer
            })

        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")
            })

        print(f"WEB SEARCH: {len(results)} results")

        return results

    except Exception as e:
        print("WEB SEARCH ERROR:", repr(e))
        return []