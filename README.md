![logo_ironhack_blue](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Automated Customer Reviews — Business Case

This repository contains the code and notebooks used to build an automated customer-review pipeline: data collection & preprocessing, review classification, product clustering, and review summarization.

**What's included**
- **Notebooks** (analysis and experiments):
  - [1_data_collection_preprocessing.ipynb](project-dsai-business-case-automated-customer-reviews/1_data_collection_preprocessing.ipynb)
  - [2_review_classification.ipynb](project-dsai-business-case-automated-customer-reviews/2_review_classification.ipynb)
  - [3_product_clustering.ipynb](project-dsai-business-case-automated-customer-reviews/3_product_clustering.ipynb)
  - [4_review_summarization.ipynb](project-dsai-business-case-automated-customer-reviews/4_review_summarization.ipynb)
- **Simple API/demo**: [app.py](project-dsai-business-case-automated-customer-reviews/app.py)
- **Dependencies**: [requirements.txt](project-dsai-business-case-automated-customer-reviews/requirements.txt)

GUI / Frontend
- The GUI for this project is an e-commerce demo split into two components:
  - `ecommerce-backend` — FastAPI backend serving product summaries, reviews, meta-categories, recommendation articles, and a `/classify` endpoint.
  - `ecommerce-frontend` — TypeScript + Bootstrap frontend (Create React App) for browsing products, viewing reviews and basic cart interactions.

Backend (FastAPI) — key endpoints
- `GET /products` — list products (optional `?category=` filter)
- `GET /products/{product_id}` — product details with reviews
- `GET /categories` — list meta-categories and counts
- `GET /categories/{category_name}/article` — recommendation article for a category
- `POST /classify` — classify a review text (fallback rule-based classifier; optional fine-tuned model support)

Deployment
- The frontend will be deployed to a static-hosting service (Vercel or GitHub Pages) and the backend will be deployed to an appropriate hosting service. The public URL(s) will be added here once available.

Quick start (ML repo)
1. In this repository, create and activate a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Optionally run the local demo API in this repo (if using `app.py`):

```powershell
python app.py
```

Notes
- The `ecommerce-backend` FastAPI service loads preprocessed CSVs from its `data/` folder and exposes the endpoints above; it includes a simple rule-based classifier as a fallback and can load a local fine-tuned model from `models/sentiment_classifier` if present.
- The frontend is a Create React App TypeScript project using Bootstrap; check the frontend repo's README for details.

Contact / Next steps
- I can add API examples, run the backend here to verify endpoints, or add Dockerfiles to both frontend and backend for easier deployment. Once you deploy, provide the public URL and I'll add it to this README.
Notes
- The `ecommerce-backend` FastAPI service loads preprocessed CSVs from its `data/` folder and exposes the endpoints above; it includes a simple rule-based classifier as a fallback and can load a local fine-tuned model from `models/sentiment_classifier` if present.
- The frontend is a Create React App TypeScript project using Bootstrap for layout; its README in `ecommerce-frontend` contains frontend-specific dev scripts.

Contact / Next steps
- I can integrate detailed API examples into this README, run the backend here to verify endpoints, or add Dockerfiles to both frontend and backend for easier local deployment.
