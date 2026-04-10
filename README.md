# Salary Prediction Application

## 🎯 AI-Powered Compensation Intelligence for Data Science Professionals

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.0-red.svg)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Supabase-Cloud-brightgreen.svg)](https://supabase.com/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange.svg)](https://ollama.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-yellow.svg)](https://scikit-learn.org/)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Model Performance](#model-performance)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [API Reference](#api-reference)
10. [Key Findings](#key-findings)
11. [Technologies Used](#technologies-used)
12. [License](#license)

---

## 📖 Overview

The **Salary Prediction Application** is an end-to-end machine learning system that predicts annual compensation for data science professionals based on job characteristics including title, experience level, location, company size, and remote work arrangement.

Built as part of an advanced ML engineering curriculum, this project demonstrates a complete production-ready pipeline: from exploratory data analysis and feature engineering through model training, API development, LLM integration, and interactive dashboard creation.

### Why This Matters

Salary uncertainty costs professionals thousands annually. Job seekers accept below-market offers, hiring managers rely on intuition, and career decisions are made without data. This platform solves that by providing:

- **Accurate predictions** based on 3,755 real salary records
- **AI-powered analysis** explaining the "why" behind each prediction
- **Market intelligence** revealing trends across locations and roles
- **Persistent storage** for tracking and comparison

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Real-time Predictions** | Instant salary estimates via REST API |
| **AI Narrative Analysis** | Llama 3.2 generates contextual insights |
| **Interactive Dashboard** | Streamlit interface with four specialized tabs |
| **Batch Processing** | Generate predictions for hundreds of scenarios |
| **Visual Analytics** | Market trends, distributions, and comparisons |
| **Persistent Storage** | All predictions saved to cloud database |

### Dashboard Tabs

| Tab | Purpose |
|-----|---------|
| **All Predictions** | View and filter complete prediction history |
| **Market Analytics** | Explore salary distributions, location impact, and trends |
| **AI Salary Analyst** | Chat with AI about compensation strategy |
| **About** | Model details and key findings |

### AI-Powered Analysis

The local LLM (Ollama + Llama 3.2) generates:
- Three-paragraph narrative analysis for each prediction
- Market context and benchmarking
- Strategic career recommendations
- Visual comparison charts (matplotlib)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────────┐
                              │   Kaggle Dataset    │
                              │  (3,755 records)    │
                              └──────────┬──────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               LOCAL PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │   Data       │───▶│  Feature     │───▶│   Decision   │                 │
│   │   Loader     │    │  Engineering │    │   Tree v4    │                 │
│   └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                   │                         │
│                                                   ▼                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │   Ollama     │◀───│   Batch      │◀───│   Model      │                 │
│   │   LLM        │    │   Predictor  │    │   Inference  │                 │
│   └──────────────┘    └──────────────┘    └──────────────┘                 │
│          │                    │                                              │
│          └────────┬───────────┘                                             │
│                   ▼                                                          │
│          ┌──────────────┐                                                   │
│          │   Supabase   │                                                   │
│          │   (Cloud)    │                                                   │
│          └──────────────┘                                                   │
│                   ▲                                                          │
└───────────────────┼─────────────────────────────────────────────────────────┘
                    │
┌───────────────────┼─────────────────────────────────────────────────────────┐
│                   │                  SERVICES                                │
├───────────────────┼─────────────────────────────────────────────────────────┤
│                   │                                                          │
│   ┌──────────────┐│              ┌──────────────┐                           │
│   │   FastAPI    ││              │  Streamlit   │                           │
│   │   (REST)     ││              │  Dashboard   │                           │
│   └──────────────┘│              └──────────────┘                           │
│          │         │                    │                                    │
│          ▼         │                    ▼                                    │
│   ┌──────────────┐ │             ┌──────────────┐                           │
│   │  Prediction  │ │             │   Supabase   │                           │
│   │  Response    │◀┴─────────────│   Read-Only  │                           │
│   └──────────────┘               └──────────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of Concerns** | Each module has a single responsibility |
| **Single Source of Truth** | Encoding logic in one file used by all components |
| **Reproducibility** | Fixed random seeds, saved artifacts |
| **Graceful Degradation** | Services handle failures without crashing |

---

## 📊 Dataset

### Source
[Kaggle Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries) (2020-2026)

### Statistics
| Metric | Value |
|--------|-------|
| Total Records | 3,755 |
| Features (raw) | 11 |
| Features (engineered) | 19 |
| Target | `salary_in_usd` |
| Salary Range | $5,132 - $450,000 |
| Mean Salary | $137,570 |
| Median Salary | $135,000 |

### Feature Distribution
| Experience Level | Count | Percentage |
|------------------|-------|------------|
| Senior (SE) | 2,029 | 67.7% |
| Mid (MI) | 626 | 20.9% |
| Entry (EN) | 250 | 8.3% |
| Executive (EX) | 91 | 3.0% |

---

## 🎯 Model Performance

### Algorithm
**Decision Tree Regressor** with GridSearchCV hyperparameter tuning

### Hyperparameters
```python
{
    'criterion': 'squared_error',
    'max_depth': 10,
    'max_features': None,
    'min_samples_leaf': 10,
    'min_samples_split': 30
}
```

### Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² (CV)** | 0.5594 | Cross-validation performance |
| **R² (Test)** | 0.4561 | Generalization to unseen data |
| **MAE** | $35,740 | Average prediction error |
| **MAE %** | 26.7% | Error relative to mean salary |
| **RMSE** | $46,441 | Penalizes large errors |

### Feature Importance
| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | `region_x_exp` | 62.1% | Location × Experience dominates |
| 2 | `dev_index` | 19.6% | Country development level |
| 3 | `size_x_title` | 5.6% | Company size × Job title |
| 4 | `job_title_encoded` | 5.0% | Job title frequency |

**Key Insight:** Where you live (combined with experience) accounts for **62%** of salary variance — more than job title, company size, and remote status combined.

---

## 📁 Project Structure

```
salary-prediction-app/
│
├── data/
│   ├── raw/                          # Raw Kaggle data (cached)
│   └── processed/                    # Cleaned data (optional)
│
├── salary_src/                       # Core ML modules
│   ├── __init__.py
│   ├── data_loader.py                # Load dataset from cache
│   ├── preprocess.py                 # All encoding + feature engineering
│   └── train_model.py                # Complete training pipeline
│
├── local_pipeline/                   # Batch processing + LLM
│   ├── __init__.py
│   ├── llm_analyzer.py               # Ollama integration
│   ├── run_pipeline.py               # Main orchestrator
│   ├── test_different_people.py      # Diversity testing
│   └── test_predictions.py           # Validation suite
│
├── fastapi_app/                      # REST API
│   ├── __init__.py
│   ├── main.py                       # FastAPI endpoints
│   ├── schemas.py                    # Pydantic validation
│   ├── utils.py                      # Preprocessing for API
│   └── model_loader.py               # Singleton model loader
│
├── streamlit_dashboard/              # Web interface
│   └── dashboard.py                  # Streamlit application
│
├── scripts/                          # Utilities
│   ├── download_dataset.py           # One-time data download
│   ├── setup_supabase.py             # Database configuration
│   └── full_validation.py            # End-to-end testing
│
├── models/                           # Saved artifacts
│   ├── decision_tree_v4.pkl          # Trained model
│   ├── transformer_v4.pkl            # PowerTransformer
│   ├── encoding_maps.json            # All encodings
│   ├── metrics_v4.json               # Performance metrics
│   └── feature_importance_v4.json    # Feature importance
│
├── notebooks/
│   └── 01_data_exploration.ipynb     # EDA notebook
│
├── .env                              # Environment variables (not in git)
├── .gitignore                        # Git ignore rules
├── pyproject.toml                    # Dependencies (uv)
├── uv.lock                           # Locked dependencies
└── README.md                         # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai/) (for LLM features)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Step 1: Clone Repository
```bash
git clone https://github.com/Jawad-Mansour/salary-prediction-app.git
cd salary-prediction-app
```

### Step 2: Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 3: Download Dataset
```bash
python scripts/download_dataset.py
```

### Step 4: Set Up Supabase
1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Run the SQL schema (see `scripts/setup_supabase.py`)
4. Copy credentials to `.env`:
```env
SUPABASE_URL=your-project-url
SUPABASE_KEY=your-anon-key
```

### Step 5: Install Ollama & Pull Model
```bash
# Install Ollama from https://ollama.ai
# Then pull the model:
ollama pull llama3.2
```

---

## 🚀 Usage

### 1. Train the Model (Optional - Pre-trained Included)
```bash
python -m salary_src.train_model
```

### 2. Start FastAPI Server
```bash
uvicorn fastapi_app.main:app --reload --port 8000
```
API available at: `http://127.0.0.1:8000`
Documentation: `http://127.0.0.1:8000/docs`

### 3. Run Local Pipeline (Generate Predictions + LLM Analysis)
```bash
# Start Ollama first (separate terminal)
ollama serve

# Run pipeline
python local_pipeline/run_pipeline.py

# Options:
# --limit 5           Test with 5 predictions
# --skip-llm          Skip LLM (faster)
# --model phi3:mini   Use different model
```

### 4. Launch Dashboard
```bash
streamlit run streamlit_dashboard/dashboard.py
```
Dashboard available at: `http://localhost:8501`

### 5. Test Individual Predictions
```bash
curl "http://127.0.0.1:8000/predict?job_title=Data%20Scientist&experience_level=SE&employment_type=FT&company_size=L&remote_ratio=100&work_year=2026&employee_residence=US&company_location=US"
```

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Service health check |
| GET | `/predict` | Predict salary |

### Request Parameters (`/predict`)

| Parameter | Type | Valid Values | Default |
|-----------|------|--------------|---------|
| `job_title` | string | Any valid title | Required |
| `experience_level` | string | EN, MI, SE, EX | Required |
| `employment_type` | string | FT, PT, CT, FL | Required |
| `company_size` | string | S, M, L | Required |
| `remote_ratio` | integer | 0, 50, 100 | Required |
| `work_year` | integer | 2020-2026 | Required |
| `employee_residence` | string | ISO country code | "US" |
| `company_location` | string | ISO country code | "US" |

### Example Response
```json
{
  "job_title": "Data Scientist",
  "experience_level": "SE",
  "employment_type": "FT",
  "company_size": "L",
  "remote_ratio": 100,
  "work_year": 2026,
  "employee_residence": "US",
  "company_location": "US",
  "predicted_salary_usd": 161485.48,
  "status": "success"
}
```

### Error Responses

| Status | Meaning | Example |
|--------|---------|---------|
| 400 | Bad request | Invalid parameter format |
| 422 | Validation error | Invalid enum value |
| 503 | Service unavailable | Model not loaded |

---

## 💡 Key Findings

### 1. Location is the Dominant Factor
`region_x_exp` accounts for **62.1%** of feature importance. Where you live, combined with your experience level, is the single biggest determinant of salary.

### 2. Experience Level Impact
| Level | Average Salary | Premium |
|-------|----------------|---------|
| Entry (EN) | $78,546 | Baseline |
| Mid (MI) | $104,526 | +33% |
| Senior (SE) | $153,051 | +95% |
| Executive (EX) | $194,931 | +148% |

### 3. Company Size Effect
| Size | Average Salary |
|------|----------------|
| Small | $78,227 |
| Medium | $143,131 |
| Large | $118,301 |

Medium companies surprisingly pay the highest averages.

### 4. Remote Work Impact
| Arrangement | Average Salary |
|-------------|----------------|
| On-site (0%) | $144,316 |
| Hybrid (50%) | $78,401 |
| Fully Remote (100%) | $136,481 |

Remote work shows no salary penalty for experienced professionals.

### 5. Top Paying Roles
| Role | Average Salary |
|------|----------------|
| Data Science Tech Lead | $375,000 |
| Cloud Data Architect | $250,000 |
| Data Lead | $212,500 |
| Principal Data Scientist | $198,171 |
| Director of Data Science | $195,141 |

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|----------|------------|---------|
| **ML Framework** | scikit-learn 1.5.1 | Decision Tree, preprocessing |
| **API Framework** | FastAPI 0.112.0 | REST endpoints |
| **Dashboard** | Streamlit 1.37.0 | Web interface |
| **Visualization** | Plotly 5.22.0 | Interactive charts |
| **Database** | Supabase | Cloud PostgreSQL |
| **LLM** | Ollama + Llama 3.2 | AI-powered analysis |
| **Data Processing** | pandas 2.2.2, numpy 1.26.4 | Data manipulation |
| **Package Manager** | uv | Fast dependency management |
| **Version Control** | Git | Source control |

---

## 📄 License

This project is created for educational purposes as part of an advanced machine learning engineering curriculum.

---

## 👤 Author

**Jawad Mansour**

---

## 🙏 Acknowledgments

- Kaggle for the [Data Science Salaries Dataset](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
- Ollama team for local LLM infrastructure
- Supabase for the generous free tier
- Streamlit for the incredible dashboard framework

---

*Built with precision. Deployed with confidence.* 🚀