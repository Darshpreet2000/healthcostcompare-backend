import os
import json
import logging
from functools import lru_cache
from fastapi import FastAPI, Query
from pydantic import BaseModel
from google.cloud import bigquery
import google.generativeai as genai
import httpx

# ----------------------------
# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------
# FastAPI app
app = FastAPI(title="CMS Healthcare AI API")

# ----------------------------
# GCP BigQuery auth from env var
sa_json_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not sa_json_str:
    raise ValueError("Missing GCP service account JSON in env var")
try:
    sa_info = json.loads(sa_json_str)
    # If sa_info is still a string, it means the JSON was double-encoded
    if isinstance(sa_info, str):
        sa_info = json.loads(sa_info)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid GCP service account JSON format: {e}") from e

bq_client = bigquery.Client.from_service_account_info(sa_info)

# ----------------------------
# Gemini AI setup
google_api_key = os.environ.get("GEMINI_API_TOKEN")
if not google_api_key:
    raise ValueError("Missing GOOGLE_API_KEY in env vars")
genai.configure(api_key=google_api_key)

# ----------------------------
# Table names
CMS_HEALTHCARE_TABLE = "spry-sensor-475217-k0.medical_data_connector.cms_healthcare_data" # Placeholder, needs actual value
HOSPITAL_DATA_TABLE = "spry-sensor-475217-k0.medical_data_connector.hospital_data" # Placeholder, needs actual value

# ----------------------------
# Response schemas
class Hospital(BaseModel):
    name: str
    rating: float
    avg_total_payment: float
    medicare_payment: float
    location: str
    distance_miles: float = None

class AIResponse(BaseModel):
    query: str
    insights: str
    why_costs_vary: str
    questions_to_ask: list[str]
    hospital_comparison: list[Hospital]

# ----------------------------
# Cache DRG_Desc list
@lru_cache(maxsize=1)
def get_drg_list():
    sql = f"SELECT DISTINCT DRG_Desc FROM `{CMS_HEALTHCARE_TABLE}`"
    query_job = bq_client.query(sql)
    drgs = [row['DRG_Desc'] for row in query_job.result()]
    logger.info(f"Loaded {len(drgs)} unique DRG_Desc values")
    return drgs

# ----------------------------
# Gemini AI call helper
def call_gemini_ai(prompt: str, max_tokens: int = 300):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # Using gemini-pro as a common model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini AI call failed: {e}")
        return "AI insights unavailable at this time."

# ----------------------------
# Match user query to DRG
def get_similar_drg(user_query: str, drg_list: list[str]) -> list[str]:
    prompt = f"""
    User query: "{user_query}"
    CMS DRG list: {drg_list}

    Return 3-5 DRG_Desc values from the list that best match the query.
    Return the names in XML tags like this: <similar_drgs_list><similar_drg>DRG_Desc_1</similar_drg><similar_drg>DRG_Desc_2</similar_drg></similar_drgs_list>
    """
    logger.info("Calling Gemini AI to match DRG")
    matches_text = call_gemini_ai(prompt)
    matches = []
    try:
        # Attempt to parse as XML
        from xml.etree import ElementTree as ET
        root = ET.fromstring(matches_text)
        if root.tag == "similar_drgs_list":
            for drg_element in root.findall("similar_drg"):
                drg_text = drg_element.text
                if drg_text:
                    matches.append(drg_text.strip())
        else:
            raise ValueError("Not a valid XML format")
    except Exception as e:
        logger.error(f"Failed to parse AI response as XML: {e}. Raw response: {matches_text}")
        # Fallback to newline splitting if XML parsing fails
        matches = [m.strip("- ").strip() for m in matches_text.split("\n") if m.strip()]

    # Further clean each match to remove potential extra quotes or brackets (if fallback was used)
    cleaned_matches = []
    for m in matches:
        m = m.strip()
        # Remove leading asterisks and spaces
        while m.startswith("*") or m.startswith(" "):
            m = m.lstrip("* ").lstrip()
        if m.startswith("'") and m.endswith("'"):
            m = m[1:-1]
        if m.startswith('"') and m.endswith('"'):
            m = m[1:-1]
        if m.startswith("[") and m.endswith("]"):
            m = m[1:-1]
        cleaned_matches.append(m.strip())

    logger.info(f"Matched DRG_Desc: {cleaned_matches}")
    return cleaned_matches

# ----------------------------
# Build SQL queries
def build_cms_sql_from_matches(matches: list[str]) -> str:
    drg_conditions = ",".join([f"'{m}'" for m in matches])
    sql = f"""
    SELECT *
    FROM `{CMS_HEALTHCARE_TABLE}`
    WHERE DRG_Desc IN ({drg_conditions})
    ORDER BY Avg_Tot_Pymt_Amt ASC
    LIMIT 20
    """
    return sql

def build_hospital_sql(provider_ids: list[str]) -> str:
    ids_str = ",".join([f"'{pid}'" for pid in provider_ids])
    sql = f"""
    SELECT *
    FROM `{HOSPITAL_DATA_TABLE}`
    WHERE facility_id IN ({ids_str})
    """
    return sql

# ----------------------------
# Query BigQuery helper
def query_bigquery(sql: str):
    logger.info(f"Running BigQuery SQL: {sql.strip()[:200]}...")
    query_job = bq_client.query(sql)
    rows = [dict(row) for row in query_job.result()]
    logger.info(f"Fetched {len(rows)} rows from BigQuery")
    return rows

# ----------------------------
# Generate AI insights
def generate_insights(user_query: str, cms_results: list[dict], hospital_list: list[dict]) -> str:
    prompt = f"""
    Generate AI insights for the following query: {user_query}.
    CMS results: {cms_results}
    Hospital data: {hospital_list}
    Format insights like: average costs, why costs vary, questions patients should ask.
    """
    logger.info("Calling Gemini AI to generate insights")
    return call_gemini_ai(prompt)

# ----------------------------
# API endpoint
@app.get("/search", response_model=AIResponse)
def search(query: str = Query(..., description="Search term like 'knee replacement'")):
    logger.info(f"Received query: {query}")

    drg_list = get_drg_list()
    matched_drg = get_similar_drg(query, drg_list)
    if not matched_drg:
        return AIResponse(
            query=query,
            insights="No matching DRG found.",
            why_costs_vary="",
            questions_to_ask=[],
            hospital_comparison=[]
        )

    cms_sql = build_cms_sql_from_matches(matched_drg)
    cms_results = query_bigquery(cms_sql)
    if not cms_results:
        return AIResponse(
            query=query,
            insights="No CMS data found for matched DRG.",
            why_costs_vary="",
            questions_to_ask=[],
            hospital_comparison=[]
        )

    provider_ids = list({row['rndrng_prvdr_ccn'] for row in cms_results})
    hospital_sql = build_hospital_sql(provider_ids)
    hospital_results = query_bigquery(hospital_sql)

    insights_text = generate_insights(query, cms_results, hospital_results)

    hospitals = []
    for h in hospital_results:
        hospitals.append(Hospital(
            name=h.get('facility_name', ""),
            rating=float(h.get('hospital_overall_rating') or 0),
            avg_total_payment=float(h.get('Avg_Tot_Pymt_Amt') or 0),
            medicare_payment=float(h.get('Avg_Mdcr_Pymt_Amt') or 0),
            location=f"{h.get('citytown', '')}, {h.get('state', '')}",
        ))

    return AIResponse(
        query=query,
        insights=insights_text,
        why_costs_vary="Costs vary due to hospital size, location, technology, and insurance negotiations.",
        questions_to_ask=[
            "What is the total estimated cost including all fees?",
            "Are there alternative procedures or treatments?",
            "What is the expected recovery time and follow-up care?",
            "Does my insurance cover this procedure at this facility?",
            "What are the credentials and experience of the medical team?"
        ],
        hospital_comparison=hospitals
    )
