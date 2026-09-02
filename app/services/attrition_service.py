"""
Attrition service managing ML prediction workflows and risk profiling.
"""

from typing import Dict, Any, Union
from app.ml.predictor import predict_attrition
from app.validation.employee_schema import EmployeeAttritionInput, AttritionPredictionResponse

class AttritionService:
    """Service handling employee attrition risk assessment."""
    
    @staticmethod
    def predict_risk(employee_data: Union[EmployeeAttritionInput, Dict[str, Any]]) -> AttritionPredictionResponse:
        """Evaluates employee attrition risk using trained ML pipeline."""
        return predict_attrition(employee_data)

attrition_service = AttritionService()
