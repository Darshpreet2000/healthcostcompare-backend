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
class ProcedureCostDetail(BaseModel):
    drg_description: str
    avg_total_payment: float
    medicare_payment: float
    total_discharges: int
    avg_submitted_covered_charge: float

class Hospital(BaseModel):
    name: str
    rating: float
    location: str
    hospital_type: str = ""
    emergency_services: str = ""
    distance_miles: float = None
    procedure_cost_details: list[ProcedureCostDetail] = []

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
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
import re

def generate_insights(user_query: str, cms_results: list[dict], hospital_list: list[dict]) -> tuple[str, str, list[str]]:
    """
    Calls Gemini to generate structured XML insights about hospital cost data.
    Returns a tuple: (insights_markdown, why_costs_vary_text, questions_list)
    """

    prompt = f"""
You are a healthcare data analyst helping patients interpret hospital medical procedure cost data to guide users to take better healthcare decisions. 

<instructions>
1. Analyze the provided hospital chargemaster data and hospital info.
2. Summarize key insights about procedure costs, trends, and variation across hospitals.
3. Explain in 2–3 sentences why costs vary.
4. Suggest 3–5 questions a patient should ask before choosing a hospital.
5. Respond ONLY in this exact XML format provided in the <ai_response> XML tags.
6. Make sure you XML response is well-formed and parsable.
</instructions>

<ai_response>
  <insights>...(Write in best way in Markdown-formatted, provide the Key Insights with trends, Tables, and recommended options)...</insights>
  <why_costs_vary>...(2–3 sentence explanation)...</why_costs_vary>
  <questions_to_ask>
    <question>Question 1</question>
    <question>Question 2</question>
    <question>Question 3</question>
  </questions_to_ask>
</ai_response>


User query: "{user_query}"

Hospital chargemaster sample data:
{cms_results}

Hospital general info sample data:
{hospital_list}

Formatting notes:
- Use **Markdown** only inside <insights>.
- Do not include anything outside <ai_response>.
- Keep total response under 350 words.
"""

    logger.info("Calling Gemini AI for XML-structured insight generation")
    ai_response_text = call_gemini_ai(prompt)

    insights_text = ""
    why_costs_text = ""
    questions_list = []

    # --- XML Parsing with fallback ---
    try:
        # Try to isolate the <ai_response> block even if extra text is added
        match = re.search(r"<ai_response>.*</ai_response>", ai_response_text, re.DOTALL)
        if match:
            xml_content = match.group(0)
        else:
            xml_content = ai_response_text  # fallback to whole string

        root = ET.fromstring(xml_content)

        # Extract insights (Markdown)
        insights_el = root.find("insights")
        if insights_el is not None:
            insights_text = (insights_el.text or "").strip()

        # Extract why_costs_vary
        why_el = root.find("why_costs_vary")
        if why_el is not None:
            why_costs_text = (why_el.text or "").strip()

        # Extract questions (nested <question> tags or text)
        questions_el = root.find("questions_to_ask")
        if questions_el is not None:
            for q_el in questions_el.findall("question"):
                q_text = (q_el.text or "").strip()
                if q_text:
                    questions_list.append(q_text)

            # fallback: if only plain text was returned inside <questions_to_ask>
            if not questions_list and questions_el.text:
                for line in questions_el.text.split("\n"):
                    line = line.strip("-• \t").strip()
                    if line:
                        questions_list.append(line)

    except Exception as e:
        logger.error(f"XML parsing failed: {e}. Raw AI output: {ai_response_text[:500]}")
        # fallback: treat entire AI output as markdown insights
        insights_text = ai_response_text

    return insights_text, why_costs_text, questions_list


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

    insights_text, why_cost_vary, questions = generate_insights(query, cms_results, hospital_results)

    # Group CMS results by provider_ccn
    cms_data_by_hospital = {}
    for cms_row in cms_results:
        provider_ccn = cms_row.get('rndrng_prvdr_ccn')
        if provider_ccn not in cms_data_by_hospital:
            cms_data_by_hospital[provider_ccn] = []
        cms_data_by_hospital[provider_ccn].append(cms_row)

    # Create a lookup for hospital data for efficient merging
    hospital_lookup = {h['facility_id']: h for h in hospital_results}

    hospitals = []
    for provider_ccn, cms_rows_for_hospital in cms_data_by_hospital.items():
        hospital_detail = hospital_lookup.get(provider_ccn)

        if hospital_detail:
            procedure_cost_details = []
            for cms_row in cms_rows_for_hospital:
                procedure_cost_details.append(ProcedureCostDetail(
                    drg_description=cms_row.get('DRG_Desc', ""),
                    avg_total_payment=float(cms_row.get('avg_tot_pymt_amt') or 0),
                    medicare_payment=float(cms_row.get('avg_mdcr_pymt_amt') or 0),
                    total_discharges=int(cms_row.get('tot_dschrgs') or 0),
                    avg_submitted_covered_charge=float(cms_row.get('avg_submtd_cvrd_chrg') or 0.0),
                ))

            hospitals.append(Hospital(
                name=hospital_detail.get('facility_name', ""),
                rating=float(hospital_detail.get('hospital_overall_rating') or 0),
                location=f"{hospital_detail.get('citytown', '')}, {hospital_detail.get('state', '')}",
                hospital_type=hospital_detail.get('hospital_type', ""),
                emergency_services=hospital_detail.get('emergency_services', ""),
                procedure_cost_details=procedure_cost_details,
            ))

    return AIResponse(
        query=query,
        insights=insights_text,
        why_costs_vary=why_cost_vary,
        questions_to_ask=questions,
        hospital_comparison=hospitals
    )
