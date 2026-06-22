import os
import re
import ssl
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse, quote

from Bio import Entrez
from dotenv import load_dotenv
import pandas as pd
import requests


load_dotenv()
Entrez.email = os.getenv("ENTREZ_EMAIL", "")
ssl._create_default_https_context = ssl._create_unverified_context

if not Entrez.email:
    raise ValueError("Missing ENTREZ_EMAIL. Set it in .env or environment variables.")

SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY", "").strip()
SCOPUS_MAX_RESULTS = int(os.getenv("SCOPUS_MAX_RESULTS", "40"))

# PUBMED_QUERY = """("machine learning" OR "supervised learning")
# AND (benchmarking OR "model comparison")
# AND (biomedical OR clinical OR bioinformatics OR biological)
# AND (dataset OR database OR "public dataset")"""
# PUBMED_QUERY = """("machine learning" OR "supervised learning")
# AND (benchmarking OR "model comparison")
# AND (biomedical OR clinical OR bioinformatics OR biological)
# AND (dataset OR database OR "public dataset")"""
PUBMED_QUERY = """("machine learning" OR "supervised learning")
AND (benchmarking OR "model comparison" OR evaluation)
AND (classification OR prediction)
AND (biological OR biomedical OR bioinformatics OR genomics)
AND (dataset OR database)"""
SCOPUS_QUERY = (
    'TITLE-ABS-KEY(("large language models" OR LLM OR "AI agents") '
    'AND (bioinformatics OR "RNA-seq" OR genomics) '
    'AND (workflow OR pipeline OR automation OR agent OR orchestration))'
)

ENABLE_SCOPUS = False


def _first_dict(value):
    if isinstance(value, list) and value:
        return value[0] if isinstance(value[0], dict) else {}
    return value if isinstance(value, dict) else {}


def _pubmed_publication_date(article_data):
    pub_date = (
        _first_dict(article_data.get("Journal", {}))
        .get("JournalIssue", {})
        .get("PubDate", {})
    )
    if not pub_date:
        pub_date = _first_dict(article_data.get("ArticleDate", {}))

    if not pub_date:
        return "", ""

    medline_date = pub_date.get("MedlineDate")
    if medline_date:
        text = str(medline_date).strip()
        year = text[:4] if len(text) >= 4 and text[:4].isdigit() else ""
        return text, year

    year = str(pub_date.get("Year", "") or "").strip()
    month = str(pub_date.get("Month", "") or "").strip()
    day = str(pub_date.get("Day", "") or "").strip()
    parts = [p for p in (year, month, day) if p]
    if not parts:
        return "", ""

    display = " ".join(parts)
    year_out = year if year.isdigit() else (display[:4] if len(display) >= 4 and display[:4].isdigit() else "")
    return display, year_out


def search_pubmed_pmids(query, max_results=40):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
    except URLError as exc:
        raise RuntimeError("Could not connect to PubMed due to SSL/network settings.") from exc
    return record.get("IdList", [])


def fetch_pubmed_details(pmids):
    if not pmids:
        return []

    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    rows = []
    for article in records.get("PubmedArticle", []):
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})
        pmid = str(medline.get("PMID", ""))
        title = str(article_data.get("ArticleTitle", ""))
        abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(part) for part in abstract_parts).strip()
        pub_date_text, pub_year = _pubmed_publication_date(article_data)
        rows.append(
            {
                "source": "pubmed",
                "id": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "title": title,
                "abstract": abstract,
                "publication_date": pub_date_text,
                "year": pub_year,
                "doi": "",
            }
        )
    return rows


def _extract_scopus_eid(entry):
    eid = entry.get("eid")
    if eid:
        return str(eid)

    prism_url = (entry.get("prism:url") or "").strip()
    if prism_url:
        qs = urlparse(prism_url).query
        eid_vals = parse_qs(qs).get("eid") or []
        if eid_vals:
            return eid_vals[0]
        match = re.search(r"[eE]id=([^&]+)", prism_url)
        if match:
            return match.group(1)

    for link in entry.get("link", []):
        href = (link.get("@href") or "").strip()
        if not href:
            continue
        api_match = re.search(r"/abstract/eid/([^/?#]+)", href, re.I)
        if api_match:
            return api_match.group(1)

    identifier = entry.get("dc:identifier", "")
    if identifier.startswith("SCOPUS_ID:"):
        nid = identifier.replace("SCOPUS_ID:", "").strip()
        return f"2-s2.0-{nid}" if nid else ""

    return ""


def _scopus_record_web_url(eid):
    if not eid:
        return ""
    return (
        "https://www.scopus.com/record/display.uri"
        f"?eid={quote(eid, safe='')}&origin=resultslist"
    )


def search_scopus(query, max_results=SCOPUS_MAX_RESULTS):
    if not SCOPUS_API_KEY:
        print("Skipping Scopus search: missing SCOPUS_API_KEY in .env")
        return []

    url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY, "Accept": "application/json"}
    normalized_query = " ".join(query.split())
    params = {"query": normalized_query, "count": max_results}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        details = ""
        if "response" in locals() and response is not None:
            details = f" | API response: {response.text[:300]}"
        print(f"Skipping Scopus search due to API/network issue: {exc}{details}")
        return []

    entries = response.json().get("search-results", {}).get("entry", [])

    rows = []
    for entry in entries:
        identifier = entry.get("dc:identifier", "")
        scopus_id = identifier.replace("SCOPUS_ID:", "") if identifier else ""
        eid = _extract_scopus_eid(entry)
        if not eid and scopus_id:
            eid = f"2-s2.0-{scopus_id}"
        title = entry.get("dc:title", "")
        abstract = entry.get("dc:description", "")
        cover_date = (entry.get("prism:coverDate") or "").strip()
        year = cover_date[:4] if cover_date and cover_date[:4].isdigit() else ""
        doi = entry.get("prism:doi", "")
        record_url = _scopus_record_web_url(eid)

        rows.append(
            {
                "source": "scopus",
                "id": eid or scopus_id,
                "url": record_url,
                "title": title,
                "abstract": abstract,
                "publication_date": cover_date,
                "year": year,
                "doi": doi,
            }
        )
        if len(rows) >= max_results:
            break

    return rows


def save_results(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        rows,
        columns=["source", "id", "url", "title", "abstract", "publication_date", "year", "doi"],
    )
    dataframe.to_csv(output_path, index=False, encoding="utf-8")


def main():
    pubmed_pmids = search_pubmed_pmids(PUBMED_QUERY)
    pubmed_rows = fetch_pubmed_details(pubmed_pmids)
    if ENABLE_SCOPUS:
        scopus_rows = search_scopus(SCOPUS_QUERY, max_results=SCOPUS_MAX_RESULTS)
    else:
        scopus_rows = []
        print("Scopus search disabled (ENABLE_SCOPUS=False).")

    all_rows = pubmed_rows + scopus_rows

    output_file = Path("outputs/literature_results.csv")
    save_results(all_rows, output_file)

    print(f"PubMed records: {len(pubmed_rows)}")
    print(f"Scopus records: {len(scopus_rows)}")
    print(f"Total records saved: {len(all_rows)}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
