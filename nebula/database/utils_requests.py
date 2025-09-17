from typing import Any, Dict, List

from pydantic import BaseModel, conint, confloat


class Routes:
    # Scenarios
    INIT = "/"
    STATUS = "/status"
    RESOURCES = "/resources"
    LEAST_MEMORY_GPU = "/least_memory_gpu"
    AVAILABLE_GPUS = "/available_gpus/"
    DISCOVER_VPN = "/discover-vpn"

    RUN = "/scenarios/run"
    UPDATE = "/scenarios/update"
    STOP = "/scenarios/stop"
    REMOVE = "/scenarios/remove"
    FINISH = "/scenarios/set_status_to_finished"
    RUNNING = "/scenarios/running"
    CHECK_SCENARIO = "/scenarios/check/{role}/{scenario_name}"
    GET_SCENARIOS_BY_USER = "/scenarios/{user}/{role}"
    GET_SCENARIOS_BY_SCENARIO_NAME = "/scenarios/{scenario_name}"

    # Nodes
    NODES_BY_SCENARIO_NAME = "/nodes/{scenario_name}"
    NODES_UPDATE = "/nodes/update"
    NODES_REMOVE = "/nodes/remove"
    NODES_UPDATE_BY_SCENARIO = "/nodes/{scenario_name}/update"
    NODES_DONE_BY_SCENARIO = "/nodes/{scenario_name}/done"

    # Notes
    NOTES_BY_SCENARIO_NAME = "/notes/{scenario_name}"
    NOTES_UPDATE = "/notes/update"
    NOTES_REMOVE = "/notes/remove"

    # Users
    USER_LIST = "/user/list"
    USER_BY_SCENARIO_NAME = "/user/{scenario_name}"
    USER_ADD = "/user/add"
    USER_DELETE = "/user/delete"
    USER_UPDATE = "/user/update"
    USER_VERIFY = "/user/verify"

    # Physical
    PHYSICAL_RUN = "/physical/run/{ip}"
    PHYSICAL_STOP = "/physical/stop/{ip}"
    PHYSICAL_SETUP = "/physical/setup/{ip}"
    PHYSICAL_STATE = "/physical/state/{ip}"
    PHYSICAL_SCENARIO_STATE = "/physical/scenario-state/{scenario_name}"


class RunScenarioRequest(BaseModel):
    """Request model to trigger a scenario run on the controller.

    - Only requires scenario_data and user.
    - Extra fields (e.g., role, federation_id) are ignored.
    """
    scenario_data: Dict[str, Any]
    user: str


class ScenarioUpdateRequest(BaseModel):
    scenario_name: str
    start_time: str
    end_time: str
    scenario: Dict[str, Any]
    status: str
    username: str


class ScenarioStopRequest(BaseModel):
    scenario_name: str
    all: bool = False


class ScenarioRemoveRequest(BaseModel):
    scenario_name: str


class ScenarioFinishRequest(BaseModel):
    scenario_name: str
    all: bool = False


class NotesUpdateRequest(BaseModel):
    scenario_name: str
    notes: str


class NotesRemoveRequest(BaseModel):
    scenario_name: str


class NodesRemoveRequest(BaseModel):
    scenario_name: str


class UserAddRequest(BaseModel):
    user: str
    password: str
    role: str


class UserDeleteRequest(BaseModel):
    user: str


class UserUpdateRequest(BaseModel):
    user: str
    password: str
    role: str


class UserVerifyRequest(BaseModel):
    user: str
    password: str


# Nodes update payload
class DeviceArgs(BaseModel):
    uid: str
    idx: int
    role: str
    malicious: bool


class NetworkArgs(BaseModel):
    ip: str
    port: conint(ge=1, le=65535)  # type: ignore[valid-type]
    neighbors: List[Any]


class MobilityArgs(BaseModel):
    latitude: confloat(ge=-90, le=90)  # type: ignore[valid-type]
    longitude: confloat(ge=-180, le=180)  # type: ignore[valid-type]


class TrackingArgs(BaseModel):
    run_hash: str


class FederationArgs(BaseModel):
    round: int


class ScenarioArgs(BaseModel):
    federation: str
    name: str


class NodesUpdateRequest(BaseModel):
    device_args: DeviceArgs
    network_args: NetworkArgs
    mobility_args: MobilityArgs
    tracking_args: TrackingArgs
    federation_args: FederationArgs
    scenario_args: ScenarioArgs
    timestamp: str


def factory_requests_path(resource: str, user: str = "", role: str = "", scenario_name: str = "") -> str:
    if resource == "init":
        return Routes.INIT
    elif resource == "update":
        return Routes.UPDATE
    elif resource == "stop":
        return Routes.STOP
    elif resource == "remove":
        return Routes.REMOVE
    elif resource == "finish":
        return Routes.FINISH
    elif resource == "running":
        return Routes.RUNNING
    elif resource == "check_scenario":
        return Routes.CHECK_SCENARIO.format(role=role, scenario_name=scenario_name)
    elif resource == "get_scenarios_by_user":
        return Routes.GET_SCENARIOS_BY_USER.format(user=user, role=role)
    elif resource == "get_scenarios_by_scenario_name":
        return Routes.GET_SCENARIOS_BY_SCENARIO_NAME.format(scenario_name=scenario_name)
    # Nodes
    elif resource == "get_nodes_by_scenario_name":
        return Routes.NODES_BY_SCENARIO_NAME.format(scenario_name=scenario_name)
    elif resource == "update_nodes":
        return Routes.NODES_UPDATE
    elif resource == "remove_nodes":
        return Routes.NODES_REMOVE
    # Notes
    elif resource == "get_notes_by_scenario_name":
        return Routes.NOTES_BY_SCENARIO_NAME.format(scenario_name=scenario_name)
    elif resource == "update_notes":
        return Routes.NOTES_UPDATE
    elif resource == "remove_notes":
        return Routes.NOTES_REMOVE
    # Users
    elif resource == "list_users":
        return Routes.USER_LIST
    elif resource == "get_user_by_scenario_name":
        return Routes.USER_BY_SCENARIO_NAME.format(scenario_name=scenario_name)
    elif resource == "add_user":
        return Routes.USER_ADD
    elif resource == "delete_user":
        return Routes.USER_DELETE
    elif resource == "update_user":
        return Routes.USER_UPDATE
    elif resource == "verify_user":
        return Routes.USER_VERIFY
    else:
        raise Exception(f"resource not found: {resource}")
