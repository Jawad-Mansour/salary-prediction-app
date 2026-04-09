"""
Local Pipeline - Batch Predictions with LLM Analysis
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
from dotenv import load_dotenv
from supabase import create_client

# Import our modules
from salary_src.preprocess import load_encoding_maps, preprocess_single_row
from local_pipeline.llm_analyzer import LLMAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(project_root / ".env")


class PipelineConfig:
    MODEL_PATH = project_root / "models" / "decision_tree.pkl"
    TRANSFORMER_PATH = project_root / "models" / "transformer.pkl"
    ENCODING_MAPS_PATH = project_root / "models" / "encoding_maps.json"
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    LLM_MODEL = "llama3.2"
    
    EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
    EMPLOYMENT_TYPES = ["FT"]
    COMPANY_SIZES = ["S", "M", "L"]
    REMOTE_RATIOS = [0, 50, 100]
    WORK_YEARS = [2024]
    
    JOB_TITLES = [
        "Data Scientist",
        "Data Engineer", 
        "Data Analyst"
    ]


def load_model_and_transformer():
    """Load model and transformer."""
    logger.info("Loading model and transformer...")
    
    # Load model
    model = joblib.load(PipelineConfig.MODEL_PATH)
    logger.info(f"✅ Model loaded: {type(model).__name__}")
    
    # Load transformer
    transformer = joblib.load(PipelineConfig.TRANSFORMER_PATH)
    logger.info(f"✅ Transformer loaded")
    
    # Load encoding maps
    encoding_maps = load_encoding_maps(PipelineConfig.ENCODING_MAPS_PATH)
    
    return model, transformer, encoding_maps


def predict_salary(model, transformer, input_data: Dict, encoding_maps: Dict) -> float:
    """Predict salary and inverse transform back to original scale."""
    try:
        # Preprocess the input
        X = preprocess_single_row(input_data, encoding_maps)
        
        # Predict on transformed scale
        prediction_transformed = model.predict(X)[0]
        
        # Inverse transform to get actual salary
        # Transformer returns a 1D array, so use [0] not [0][0]
        prediction_array = transformer.inverse_transform(np.array([[prediction_transformed]]))
        prediction = float(prediction_array[0])
        
        # Ensure prediction is reasonable (between 20k and 500k)
        if prediction < 20000:
            logger.warning(f"Prediction too low (${prediction:,.2f}), capping at $20,000")
            prediction = 20000.0
        elif prediction > 500000:
            logger.warning(f"Prediction too high (${prediction:,.2f}), capping at $500,000")
            prediction = 500000.0
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def generate_input_grid() -> List[Dict]:
    combinations = []
    for job_title in PipelineConfig.JOB_TITLES:
        for exp_level in PipelineConfig.EXPERIENCE_LEVELS:
            for emp_type in PipelineConfig.EMPLOYMENT_TYPES:
                for company_size in PipelineConfig.COMPANY_SIZES:
                    for remote_ratio in PipelineConfig.REMOTE_RATIOS:
                        for work_year in PipelineConfig.WORK_YEARS:
                            combinations.append({
                                'job_title': job_title,
                                'experience_level': exp_level,
                                'employment_type': emp_type,
                                'company_size': company_size,
                                'remote_ratio': remote_ratio,
                                'work_year': work_year,
                                'employee_residence': 'US',
                                'company_location': 'US'
                            })
    logger.info(f"Generated {len(combinations)} combinations")
    return combinations


def run_pipeline(limit: Optional[int] = None, skip_llm: bool = False):
    logger.info("=" * 60)
    logger.info("🚀 STARTING LOCAL PIPELINE")
    logger.info("=" * 60)
    
    # Load model, transformer, and maps
    model, transformer, encoding_maps = load_model_and_transformer()
    
    # Initialize LLM
    llm = None
    if not skip_llm:
        try:
            llm = LLMAnalyzer(model_name=PipelineConfig.LLM_MODEL)
            logger.info(f"✅ LLM initialized: {PipelineConfig.LLM_MODEL}")
        except Exception as e:
            logger.warning(f"LLM init failed: {e}")
    
    # Initialize Supabase
    supabase = create_client(PipelineConfig.SUPABASE_URL, PipelineConfig.SUPABASE_KEY)
    logger.info("✅ Supabase client ready")
    
    # Generate input grid
    input_grid = generate_input_grid()
    if limit:
        input_grid = input_grid[:limit]
        logger.info(f"LIMIT: Processing only {limit} combinations")
    
    successful = 0
    failed = 0
    
    for idx, input_data in enumerate(input_grid):
        logger.info(f"\n[{idx+1}/{len(input_grid)}] Processing: {input_data['job_title']} - {input_data['experience_level']}")
        
        # Predict salary
        predicted_salary = predict_salary(model, transformer, input_data, encoding_maps)
        
        if predicted_salary is None or predicted_salary <= 0:
            logger.error(f"   Invalid prediction: {predicted_salary}")
            failed += 1
            continue
        
        logger.info(f"   Predicted: ${predicted_salary:,.2f}")
        
        # Generate LLM analysis
        narrative = None
        chart_base64 = None
        
        if llm and not skip_llm:
            try:
                narrative, chart_base64 = llm.generate_analysis(input_data, predicted_salary)
                logger.info(f"   ✅ LLM analysis generated")
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"   LLM failed: {e}")
        
        # Insert into Supabase
        record = {
            'job_title': input_data['job_title'],
            'experience_level': input_data['experience_level'],
            'employment_type': input_data['employment_type'],
            'company_size': input_data['company_size'],
            'remote_ratio': input_data['remote_ratio'],
            'work_year': input_data['work_year'],
            'employee_residence': input_data.get('employee_residence', 'US'),
            'company_location': input_data.get('company_location', 'US'),
            'predicted_salary_usd': predicted_salary,
            'llm_narrative': narrative,
            'chart_base64': chart_base64,
            'prediction_version': 'v3'
        }
        
        try:
            supabase.table('predictions').insert(record).execute()
            successful += 1
            logger.info(f"   ✅ Inserted into Supabase")
        except Exception as e:
            logger.error(f"   Insert failed: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 PIPELINE SUMMARY")
    logger.info(f"   Total: {len(input_grid)}")
    logger.info(f"   Successful: {successful}")
    logger.info(f"   Failed: {failed}")
    logger.info("=" * 60)
    logger.info("✅ PIPELINE COMPLETE!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit number of predictions')
    parser.add_argument('--skip-llm', action='store_true', help='Skip LLM calls')
    parser.add_argument('--model', type=str, default='llama3.2', help='Ollama model to use')
    args = parser.parse_args()
    
    if args.model:
        PipelineConfig.LLM_MODEL = args.model
        print(f"Using LLM model: {PipelineConfig.LLM_MODEL}")
    
    run_pipeline(limit=args.limit, skip_llm=args.skip_llm)