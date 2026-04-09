"""
LLM Analyzer Module - Handles Ollama integration for narrative and chart generation

Author: Salary Prediction App
Version: 1.0.1 (Fixed variable name)
"""

import ollama
import base64
import io
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    Handles communication with Ollama LLM for salary analysis.
    Supports model switching (llama3.2, phi3:mini, etc.)
    """
    
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize the LLM analyzer.
        
        Args:
            model_name: Ollama model to use (llama3.2, phi3:mini, mistral, etc.)
        """
        self.model_name = model_name
        self._check_model_available()
    
    def _check_model_available(self):
        """Check if the specified model is available in Ollama."""
        try:
            # List available models
            models = ollama.list()
            available_models = [m['name'] for m in models.get('models', [])]
            
            # Check if our model is available (exact or partial match)
            model_found = any(self.model_name in m for m in available_models)
            
            if not model_found:
                logger.warning(f"Model '{self.model_name}' not found in Ollama")
                logger.warning(f"Available models: {available_models}")
                logger.warning(f"Please run: ollama pull {self.model_name}")
            else:
                logger.info(f"✅ LLM Model ready: {self.model_name}")
        except Exception as e:
            logger.error(f"Could not connect to Ollama: {e}")
            logger.error("Make sure Ollama is running (ollama serve)")
    
    def switch_model(self, new_model: str):
        """Switch to a different Ollama model."""
        self.model_name = new_model
        self._check_model_available()
        logger.info(f"Switched to model: {self.model_name}")
    
    def generate_analysis(self, job_data: Dict, predicted_salary: float) -> Tuple[str, str]:
        """
        Generate narrative and chart using LLM.
        
        Args:
            job_data: Dictionary with job details
            predicted_salary: Model's salary prediction
        
        Returns:
            Tuple of (narrative_text, chart_base64_string)
        """
        # Build the prompt
        prompt = self._build_prompt(job_data, predicted_salary)
        
        try:
            # Call Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.7,  # Slight creativity
                    'num_predict': 1024,  # Max tokens to generate
                }
            )
            
            # Extract response text
            full_response = response['message']['content']
            
            # Parse narrative and chart
            narrative, chart_code = self._parse_response(full_response)
            
            # Generate chart from code
            chart_base64 = self._execute_chart_code(chart_code, job_data, predicted_salary)
            
            return narrative, chart_base64
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._get_fallback_narrative(job_data, predicted_salary), ""
    
    def _build_prompt(self, job_data: Dict, predicted_salary: float) -> str:
        """Build the LLM prompt with all job details."""
        
        # Map encoded/abbreviated values to readable text
        experience_map = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior', 'EX': 'Executive'}
        employment_map = {'FT': 'Full-Time', 'PT': 'Part-Time', 'CT': 'Contract', 'FL': 'Freelance'}
        company_size_map = {'S': 'Small (1-50 employees)', 'M': 'Medium (51-250 employees)', 'L': 'Large (250+ employees)'}
        
        experience = experience_map.get(job_data.get('experience_level', 'Unknown'), job_data.get('experience_level', 'Unknown'))
        employment = employment_map.get(job_data.get('employment_type', 'Unknown'), job_data.get('employment_type', 'Unknown'))
        company_size = company_size_map.get(job_data.get('company_size', 'Unknown'), job_data.get('company_size', 'Unknown'))
        remote_ratio = job_data.get('remote_ratio', 0)
        job_title = job_data.get('job_title', 'Unknown')
        
        prompt = f"""You are a compensation analyst. Given the following job profile and predicted salary, write a 3-paragraph narrative analysis.

Job Title: {job_title}
Experience Level: {experience}
Employment Type: {employment}
Company Size: {company_size}
Remote Ratio: {remote_ratio}%
Predicted Salary: ${predicted_salary:,.2f}

Paragraph 1: Interpret this salary in context — is it high or low for this profile? Why?
Paragraph 2: What factors in this profile most likely drove the prediction up or down?
Paragraph 3: What would this person need to change (role, experience, company size, location) to meaningfully increase their compensation?

Then output Python matplotlib code (between <chart> and </chart> tags) that produces a bar chart comparing this predicted salary against the typical range for this experience level. Use only matplotlib. No external data needed — use reasonable benchmark values based on your knowledge.

Example chart code format:
<chart>
import matplotlib.pyplot as plt
import numpy as np

categories = ['Entry', 'Mid', 'Senior', 'Executive']
typical_ranges = [65000, 95000, 130000, 180000]
predicted_salary = {predicted_salary}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, typical_ranges, color='steelblue', alpha=0.7, label='Typical Range')
ax.axhline(y=predicted_salary, color='red', linewidth=2, linestyle='--', label=f'Predicted: ${predicted_salary:,.0f}')
ax.set_xlabel('Experience Level')
ax.set_ylabel('Salary (USD)')
ax.set_title(f'Salary Comparison: {job_title}')
ax.legend()
plt.tight_layout()
</chart>
"""
        return prompt
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse LLM response to extract narrative and chart code.
        
        Returns:
            Tuple of (narrative_text, chart_code)
        """
        narrative = response
        chart_code = ""
        
        # Extract chart code between <chart> and </chart> tags
        chart_match = re.search(r'<chart>(.*?)</chart>', response, re.DOTALL)
        if chart_match:
            chart_code = chart_match.group(1).strip()
            # Remove chart code from narrative
            narrative = re.sub(r'<chart>.*?</chart>', '', response, flags=re.DOTALL).strip()
        
        # Clean up narrative
        narrative = narrative.strip()
        
        return narrative, chart_code
    
    def _execute_chart_code(self, chart_code: str, job_data: Dict, predicted_salary: float) -> str:
        """
        Execute the chart code and return base64 encoded image.
        
        Args:
            chart_code: Python code that generates a matplotlib chart
            job_data: Job details for context
            predicted_salary: Predicted salary value
        
        Returns:
            Base64 encoded PNG string
        """
        if not chart_code:
            # Generate default chart
            return self._generate_default_chart(job_data, predicted_salary)
        
        try:
            # Create a local namespace for code execution
            namespace = {
                'plt': plt,
                'np': __import__('numpy'),
                'predicted_salary': predicted_salary,
                'job_data': job_data,
                'fig': None,
                'ax': None
            }
            
            # Execute the chart code
            exec(chart_code, namespace)
            
            # Get the figure (either from namespace or use current figure)
            fig = namespace.get('fig', plt.gcf())
            
            # Convert to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Clear the figure to free memory
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return self._generate_default_chart(job_data, predicted_salary)
    
    def _generate_default_chart(self, job_data: Dict, predicted_salary: float) -> str:
        """Generate a default chart when LLM chart generation fails."""
        try:
            experience_map = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
            exp_level = job_data.get('experience_level', 'MI')
            exp_name = experience_map.get(exp_level, 'Mid')
            
            # Typical salary ranges by experience
            typical_ranges = {
                'Entry': 65000,
                'Mid': 95000,
                'Senior': 130000,
                'Executive': 180000
            }
            
            typical = typical_ranges.get(exp_name, 95000)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            categories = ['Typical Range', 'Predicted']
            values = [typical, predicted_salary]
            colors = ['steelblue', 'coral']
            
            bars = ax.bar(categories, values, color=colors, edgecolor='black')
            ax.set_ylabel('Salary (USD)')
            ax.set_title(f'Salary Comparison: {job_data.get("job_title", "Unknown")}')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                       f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
            
            ax.axhline(y=typical, color='gray', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Default chart generation failed: {e}")
            return ""
    
    def _get_fallback_narrative(self, job_data: Dict, predicted_salary: float) -> str:
        """Generate fallback narrative when LLM fails."""
        experience_map = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior', 'EX': 'Executive'}
        exp = experience_map.get(job_data.get('experience_level', 'MI'), 'Mid Level')
        
        return f"""Based on the analysis, a {job_data.get('job_title', 'professional')} at {exp} level 
        with {job_data.get('remote_ratio', 0)}% remote work is predicted to earn ${predicted_salary:,.2f}.
        
        This prediction takes into account the company size ({job_data.get('company_size', 'M')}) 
        and employment type ({job_data.get('employment_type', 'FT')}).
        
        For a higher salary, consider advancing to Senior or Executive levels, or moving to a larger company."""