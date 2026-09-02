"""
Attrition Prediction API endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from app.validation.employee_schema import EmployeeAttritionInput, AttritionPredictionResponse
from app.services.attrition_service import attrition_service
from app.utils.logger import logger

router = APIRouter(tags=["Attrition Prediction"])

@router.post(
    "/predict/attrition",
    response_model=AttritionPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Employee Attrition Risk",
    description="Calculates flight risk probability and assigns risk category (Low, Medium, High) using the trained XGBoost model."
)
async def predict_employee_attrition(payload: EmployeeAttritionInput):
    try:
        response = attrition_service.predict_risk(payload)
        return response
    except Exception as e:
        logger.error(f"Error during attrition prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
