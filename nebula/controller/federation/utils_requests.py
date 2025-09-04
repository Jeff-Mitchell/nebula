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
    
def factory_requests_path(resource: str, scenario_name: str = "", federation_id: str = "") -> str:
    if resource == "init":
        return "/init"
    elif resource == "run":
        return "/scenarios/run"
    elif resource == "stop":
        return "/scenarios/stop"
    elif resource == "update":
        return f"/nodes/{scenario_name}/update"
    elif resource == "done":
        return f"/nodes/{scenario_name}/done"
    elif resource == "finish":
        return f"/scenarios/{federation_id}/finish"
    else:
        raise Exception(f"resource not found: {resource}")
    
    