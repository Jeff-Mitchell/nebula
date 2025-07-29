
import argparse
import logging
import os
import sys
from typing import Annotated

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from fastapi import Body, FastAPI, HTTPException, Path, status
from fastapi.concurrency import asynccontextmanager

from nebula.database.database_adapter_factory import factory_database_adapter

# Get a database instance
db = factory_database_adapter("PostgresDB")


# Setup logger
def configure_logger(log_file):
    """
    Configures the logging system for the database API.
    """
    log_console_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_console_format))
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            console_handler,
            file_handler,
        ],
    )
    uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access"]
    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = False
        handler = logging.FileHandler(log_file, mode="a")
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
        logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for the database API.
    """
    # Code to run on startup
    db_log = os.environ.get("NEBULA_DATABASE_LOG", "database.log")
    configure_logger(db_log)

    # Initialize the database connection pool
    await db.init_db_pool()
    await db.insert_default_admin()

    yield

    # Code to run on shutdown
    await db.close_db_pool()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the NEBULA Database API"}


# Scenarios
@app.post("/scenarios/update")
async def update_scenario(
    scenario_name: str = Body(..., embed=True),
    start_time: str = Body(..., embed=True),
    end_time: str = Body(..., embed=True),
    scenario: dict = Body(..., embed=True),
    status: str = Body(..., embed=True),
    username: str = Body(..., embed=True),
):
    try:
        await db.scenario_update_record(scenario_name, start_time, end_time, scenario, status, username)
        return {"message": f"Scenario {scenario_name} updated successfully"}
    except Exception as e:
        logging.exception(f"Error updating scenario {scenario_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/scenarios/stop")
async def stop_scenario(
    scenario_name: str = Body(..., embed=True),
    all: bool = Body(False, embed=True),
):
    try:
        if all:
            await db.scenario_set_all_status_to_finished()
        else:
            await db.scenario_set_status_to_finished(scenario_name)
        return {"message": "Scenario status updated successfully"}
    except Exception as e:
        logging.exception(f"Error stopping scenario {scenario_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/scenarios/remove")
async def remove_scenario(
    scenario_name: str = Body(..., embed=True),
):
    try:
        await db.remove_scenario_by_name(scenario_name)
        return {"message": f"Scenario {scenario_name} removed successfully"}
    except Exception as e:
        logging.exception(f"Error removing scenario {scenario_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/scenarios/{user}/{role}")
async def get_scenarios(
    user: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
    role: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
):
    try:
        scenarios = await db.get_all_scenarios_and_check_completed(username=user, role=role)
        if role == "admin":
            scenario_running = await db.get_running_scenario()
        else:
            scenario_running = await db.get_running_scenario(username=user)
        return {"scenarios": scenarios, "scenario_running": scenario_running}
    except Exception as e:
        logging.exception(f"Error obtaining scenarios: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/scenarios/set_status_to_finished")
async def set_scenario_status_to_finished(
    scenario_name: str = Body(..., embed=True), all: bool = Body(False, embed=True)
):
    try:
        if all:
            await db.scenario_set_all_status_to_finished()
        else:
            await db.scenario_set_status_to_finished(scenario_name)
        return {"message": f"Scenario {scenario_name} status set to finished successfully"}
    except Exception as e:
        logging.exception(f"Error setting scenario {scenario_name} to finished: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/scenarios/running")
async def get_running_scenario_endpoint(get_all: bool = False):
    try:
        return await db.get_running_scenario(get_all=get_all)
    except Exception as e:
        logging.exception(f"Error obtaining running scenario: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/scenarios/check/{role}/{scenario_name}")
async def check_scenario(
    role: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
    scenario_name: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
):
    try:
        allowed = await db.check_scenario_with_role(role, scenario_name)
        return {"allowed": allowed}
    except Exception as e:
        logging.exception(f"Error checking scenario with role: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/scenarios/{scenario_name}")
async def get_scenario_by_name_endpoint(
    scenario_name: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
):
    try:
        scenario = await db.get_scenario_by_name(scenario_name)
        return scenario
    except Exception as e:
        logging.exception(f"Error obtaining scenario {scenario_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Nodes
@app.get("/nodes/{scenario_name}")
async def list_nodes_by_scenario_name_endpoint(
    scenario_name: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
):
    try:
        nodes = await db.list_nodes_by_scenario_name(scenario_name)
        return nodes
    except Exception as e:
        logging.exception(f"Error obtaining nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/nodes/update")
async def update_node_record(data: dict):
    try:
        await db.update_node_record(
            str(data["device_args"]["uid"]),
            str(data["device_args"]["idx"]),
            str(data["network_args"]["ip"]),
            str(data["network_args"]["port"]),
            str(data["device_args"]["role"]),
            data["network_args"]["neighbors"],
            str(data["mobility_args"]["latitude"]),
            str(data["mobility_args"]["longitude"]),
            str(data["timestamp"]),
            str(data["scenario_args"]["federation"]),
            str(data["federation_args"]["round"]),
            str(data["scenario_args"]["name"]),
            str(data["tracking_args"]["run_hash"]),
            str(data["device_args"]["malicious"]),
        )
        return {"message": "Node updated successfully"}
    except Exception as e:
        logging.exception(f"Error updating node: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/nodes/remove")
async def remove_nodes_by_scenario_name_endpoint(scenario_name: str = Body(..., embed=True)):
    try:
        await db.remove_nodes_by_scenario_name(scenario_name)
        return {"message": f"Nodes for scenario {scenario_name} removed successfully"}
    except Exception as e:
        logging.exception(f"Error removing nodes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Notes
@app.get("/notes/{scenario_name}")
async def get_notes_by_scenario_name(
    scenario_name: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
):
    try:
        notes_record = await db.get_notes(scenario_name)
        if notes_record is not None:
            notes_record = dict(notes_record.items())
        return notes_record
    except Exception as e:
        logging.exception(f"Error obtaining notes for scenario {scenario_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/notes/update")
async def update_notes_by_scenario_name(scenario_name: str = Body(..., embed=True), notes: str = Body(..., embed=True)):
    try:
        await db.save_notes(scenario_name, notes)
        return {"message": f"Notes for scenario {scenario_name} updated successfully"}
    except Exception as e:
        logging.exception(f"Error updating notes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/notes/remove")
async def remove_notes_by_scenario_name_endpoint(scenario_name: str = Body(..., embed=True)):
    try:
        await db.remove_note(scenario_name)
        return {"message": f"Notes for scenario {scenario_name} removed successfully"}
    except Exception as e:
        logging.exception(f"Error removing notes: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Users
@app.get("/user/list")
async def list_users_controller(all_info: bool = False):
    try:
        user_list = await db.list_users(all_info)
        if all_info:
            user_list = [dict(user) for user in user_list]
        return {"users": user_list}
    except Exception as e:
        logging.exception(f"Error retrieving users: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving users: {e}")


@app.get("/user/{scenario_name}")
async def get_user_by_scenario_name_endpoint(
    scenario_name: Annotated[str, Path(pattern="^[a-zA-Z0-9_-]+$", min_length=1, max_length=50)],
):
    try:
        user = await db.get_user_by_scenario_name(scenario_name)
        return user
    except Exception as e:
        logging.exception(f"Error obtaining user for scenario {scenario_name}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/user/add")
async def add_user_controller(user: str = Body(...), password: str = Body(...), role: str = Body(...)):
    try:
        await db.add_user(user, password, role)
        return {"detail": "User added successfully"}
    except Exception as e:
        logging.exception(f"Error adding user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error adding user: {e}")


@app.post("/user/delete")
async def remove_user_controller(user: str = Body(..., embed=True)):
    try:
        await db.delete_user_from_db(user)
        return {"detail": "User deleted successfully"}
    except Exception as e:
        logging.exception(f"Error deleting user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error deleting user: {e}")


@app.post("/user/update")
async def update_user_controller(user: str = Body(...), password: str = Body(...), role: str = Body(...)):
    try:
        await db.update_user(user, password, role)
        return {"detail": "User updated successfully"}
    except Exception as e:
        logging.exception(f"Error updating user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error updating user: {e}")


@app.post("/user/verify")
async def verify_user_controller(user: str = Body(...), password: str = Body(...)):
    try:
        user_submitted = user.upper()
        users = await db.list_users()
        if users and await db.verify(user_submitted, password):
            user_info = await db.get_user_info(user_submitted)
            return {"user": user_submitted, "role": user_info[2]}
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        logging.exception(f"Error verifying user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error verifying user: {e}")
