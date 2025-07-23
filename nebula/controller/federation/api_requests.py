from pydantic import BaseModel
from typing import Dict, Any

class InitFederationRequest(BaseModel):
    experiment_type: str

class RunScenarioRequest(BaseModel):
    scenario_data: Dict[str, Any]
    user: str
    federation_id: str
    
class StopScenarioRequest(BaseModel):
    federation_id: str
    