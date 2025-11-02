# CMS Healthcare AI API

## Overview

This project provides a FastAPI backend that serves as an AI-powered API for healthcare cost comparison. It integrates with Google BigQuery to access comprehensive CMS healthcare and hospital data, and leverages Google's Gemini AI for natural language processing, DRG (Diagnosis Related Group) matching, and the generation of patient-friendly insights. This API is a core component of the broader MediCompare AI ecosystem, enabling intelligent and data-driven healthcare cost transparency.

## Features

*   **Natural Language Query Processing:** Interprets user queries (e.g., "Compare knee replacement costs in Boston") to extract relevant medical procedures and locations.
*   **BigQuery Integration:** Efficiently queries large datasets of CMS healthcare and hospital information stored in Google BigQuery.
*   **AI-Powered DRG Matching:** Uses Gemini AI to match natural language procedure descriptions to standardized DRG codes.
*   **AI-Generated Insights:** Generates structured insights, explanations for cost variations, and patient-specific questions using Gemini AI, based on retrieved data.
*   **Structured API Responses:** Provides well-defined JSON responses for easy integration with frontend applications.

## Architecture

The API is built with the following key components:

*   **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
*   **Google BigQuery:** A fully-managed, serverless data warehouse that enables super-fast SQL queries using the processing power of Google's infrastructure.
*   **Google Gemini AI Studio:** Utilized for advanced natural language understanding and generation capabilities.

## Setup and Installation

To get this API running locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Darshpreet2000/healthcostcompare-backend.git
    cd healthcostcompare
    ```
2.  **Install dependencies:**
    Ensure you have Python installed. Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables:**
    You need to set the following environment variables:
    *   `GOOGLE_APPLICATION_CREDENTIALS_JSON`: A JSON string containing your Google Cloud service account key. This is required for BigQuery authentication.
    *   `GEMINI_API_TOKEN`: Your API key for accessing Google's Gemini AI.

    Example (for local development, do not commit sensitive keys):
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type": "service_account", ...}'
    export GEMINI_API_TOKEN='YOUR_GEMINI_API_KEY'
    ```
    For deployment, use your platform's secure environment variable management.

4.  **Update BigQuery Table Names (if necessary):**
    In `api/search.py`, ensure `CMS_HEALTHCARE_TABLE` and `HOSPITAL_DATA_TABLE` variables point to your actual BigQuery table paths.

5.  **Run the API:**
    ```bash
    uvicorn api.search:app --reload
    ```
    The API will be accessible at `http://127.0.0.1:8000`.

## API Endpoints

### `GET /search`

**Description:** Processes a natural language query to find matching medical procedures, retrieve hospital cost data, and generate AI-powered insights.

**Parameters:**

*   `query` (string, **required**): The natural language search term (e.g., "Compare knee replacement costs in Boston").

**Example Request:**

```
GET http://127.0.0.1:8000/search?query=Compare%20appendectomy%20costs%20in%20California
```

**Example Response (simplified):**

```json
{
  "query": "Compare appendectomy costs in California",
  "insights": "...",
  "why_costs_vary": "...",
  "questions_to_ask": [
    "Question 1",
    "Question 2"
  ],
  "hospital_comparison": [
    {
      "name": "Hospital A",
      "rating": 4.5,
      "location": "Los Angeles, CA",
      "procedure_cost_details": [
        {
          "drg_description": "APPENDICECTOMY W CC",
          "avg_total_payment": 15000.0,
          "medicare_payment": 12000.0,
          "total_discharges": 100,
          "avg_submitted_covered_charge": 25000.0
        }
      ]
    }
  ]
}
```

## Technologies Used

*   **Language:** Python
*   **Web Framework:** FastAPI
*   **Data Warehousing:** Google Cloud BigQuery
*   **AI/NLP:** Google Generative AI (Gemini)
*   **Data Validation:** Pydantic
*   **HTTP Client:** httpx

## Contributing

Contributions are welcome! Please refer to the contribution guidelines (if available) for more information.
