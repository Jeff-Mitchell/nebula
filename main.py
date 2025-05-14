import os
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

import nebula.node as node

import subprocess, json, re

import asyncio
import signal

app = FastAPI()

CERTS_FOLDER = "./app/certs/"
CONFIG_FOLDER = "./app/config/"
LOGS_FOLDER = "./app/logs/"
METRICS_FOLDER = "./app/logs/metrics/"

CONFIG_FILE_COUNT = 1
DATASET_FILE_COUNT = 2

TRAINING_PROC: asyncio.subprocess.Process | None = None

def _find_x_files(folder: str, extension: str = ".json"):
    """
    Find all json files in a folder

    Args:
        folder (str): Path to the folder to be checked
    """
    archivos_json = []
    for file_name in os.listdir(folder):
        if file_name.endswith(extension):
            archivos_json.append(os.path.join(folder, file_name))
    return archivos_json


def _LFI_sentry(path: str):
    """
    Basic anti path traversal sentry. TODO: improve, it shouldn't be necessary to check for all these characters manually
    It also checks if the folder exists

    Args:
        path (str): Path to be checked

    Returns:
        bool: True if the path is malicious, False if it's safe and exists
    """
    return (
        not os.path.exists(CONFIG_FOLDER + path)
        or path == ""
        or ".." in path
        or "/" in path
        or "\\" in path
        or "~" in path
        or "*" in path
        or "?" in path
        or ":" in path
        or "<" in path
        or ">" in path
        or "|" in path
        or '"' in path
        or "'" in path
        or "`" in path
        or "$" in path
        or "%" in path
        or "&" in path
        or "!" in path
        or "{" in path
        or "}" in path
        or "[" in path
        or "]" in path
        or "@" in path
        or "#" in path
        or "+" in path
        or "=" in path
        or ";" in path
        or "," in path
        or " " in path
        or "\t" in path
        or "\n" in path
        or "\r" in path
        or "\f" in path
        or "\v" in path
    )


# Config
@app.get("/config/", tags=["config"])
def get_config(
    path: str,
):
    """
    Get the config file

    Args:
        path (str): Name of the folder (nebula+DFL+timestamp) where the config file is located

    Returns:
        FileResponse: The config file"""
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        json_files = _find_x_files(CONFIG_FOLDER + path + "/")
        if len(json_files) != CONFIG_FILE_COUNT:
            # raise Exception("There should be only one json file in the folder")
            raise HTTPException(status_code=404, detail="Item not found")
        else:
            file_name = json_files.pop()
            with open(file_name) as file:
                return FileResponse(file)


@app.put("/config/", status_code=status.HTTP_201_CREATED, tags=["config"], response_model=dict)
def set_config(
    config: Annotated[UploadFile, File()],
    path: str,
) -> dict:
    """
    Set the config file

    Args:
        config (UploadFile): File to be written
        path (str): Name of the folder where the config file is located. Path should be $scenraio_args.name

    Returns:
        dict: Name of the written file

    """
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")

    else:
        os.makedirs(CONFIG_FOLDER + path, exist_ok=True)
        with open(CONFIG_FOLDER + path + "/" + config.filename, "wb") as file:
            file.write(config.file.read())
            return {"filename": config.filename}


@app.delete("/config/", tags=["config"], response_model=dict)
def delete_config(
    path: str,
) -> dict:
    """
    Delete the config file

    Args:
        path (str): Name of the folder (nebula+DFL+timestamp) where the config file is located

    Returns:
        dict: Name of the deleted file
    """
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")

    # check if there is a json file in the folder
    json_files = _find_x_files(CONFIG_FOLDER + path + "/")
    if len(json_files) != CONFIG_FILE_COUNT:
        # raise Exception("There should be only one json file in the folder")
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        file_name = json_files.pop()
        os.remove(file_name)
        return {"filename": file_name}


# Dataset
@app.get("/dataset/", tags=["dataset"], response_model=list)
def get_dataset(
    path: str,
) -> list:
    """
    Get the dataset file

    Args:
        path (str): Name of the folder (nebula+DFL+timestamp) where the dataset file is located

    Returns:
        list[FileResponse]: List of the two dataset files in the folder
    """

    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")

    h5_files = _find_x_files(CONFIG_FOLDER + path + "/", ".h5")
    if len(h5_files) != DATASET_FILE_COUNT:
        # raise Exception("There should be only two h5 file in the folder")
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        file_response = []
        for file_name in h5_files:
            with open(file_name) as file:
                file_response.append(FileResponse(file))
            return file_response


@app.put("/dataset/", status_code=status.HTTP_201_CREATED, tags=["dataset"])
def set_dataset(
    dataset: Annotated[UploadFile, File()],
    dataset_p: Annotated[UploadFile, File()],
    path: str,
) -> list:
    """
    Set the dataset file

    Args:
        dataset (UploadFile): File to be written
        path (str): Name of the folder (nebula+DFL+timestamp) where the dataset file is located

    Returns:
        dict: Name of the written file
    """
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")

    else:
        try:
            os.makedirs(CONFIG_FOLDER + path, exist_ok=True)
            return_files = []
            with open(CONFIG_FOLDER + path + "/" + dataset.filename, "wb") as file:
                file.write(dataset.file.read())
                return_files.append(dataset.filename)
            with open(CONFIG_FOLDER + path + "/" + dataset_p.filename, "wb") as file:
                file.write(dataset_p.file.read())
                return_files.append(dataset_p.filename)
        except OSError:
            pass
        return return_files


@app.delete("/dataset/", tags=["dataset"])
def delete_dataset(
    path: str,
):
    """
    Delete the dataset file

    Args:
        path (str): Name of the folder (nebula+DFL+timestamp) where the dataset file is located

    Returns:
        dict: Name of the deleted file"""
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")

    # check if there is a json file in the folder
    data_files = _find_x_files(CONFIG_FOLDER + path + "/", ".h5")
    if len(data_files) != DATASET_FILE_COUNT:
        # raise Exception("There should be only one json file in the folder")
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        removed_files = {}
        for file_name in data_files:
            os.remove(file_name)
            removed_files[file_name] = "deleted"
        return removed_files


# certs
@app.get("/certs/", tags=["certs"])
def get_certs():
    """
    Get the certs file

    Returns:
        FileResponse: The certs file
    """
    certs_files = _find_x_files(CERTS_FOLDER + "/", ".cert")
    return_files = []
    for file_name in certs_files:
        with open(file_name) as file:
            return_files.append(FileResponse(file))
    return return_files


@app.put("/certs/", status_code=status.HTTP_201_CREATED, tags=["certs"])
def set_cert(
    cert: Annotated[UploadFile, File()],
) -> dict:
    """
    Set the certs file

    Args:
        cert (UploadFile): File to be written

    Returns:
        dict: Name of the written file"""
    with open(CERTS_FOLDER + cert.filename, "wb") as file:
        file.write(cert.file.read())
        return {"filename": cert.filename}


@app.delete("/certs/", tags=["certs"])
def delete_certs():
    """
    Delete the ALL certs file

    Returns:
        dict: Name of the deleted file
    """
    certs_files = _find_x_files(CERTS_FOLDER + "/", ".cert")
    removed_files = {}
    for file_name in certs_files:
        os.remove(file_name)
        removed_files[file_name] = "deleted"
    return removed_files


# Logs
@app.get("/get_logs/", tags=["logs"])
def get_logs(
    path: str,
):
    """
    Get the log file

    Args:
        path (str): Name of the folder (nebula+DFL+timestamp) where the log file is located

    Returns:
        FileResponse: The log file
    """
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", ".log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        log_file = min(log_files, key=lambda x: len(os.path.basename(x)))
        with open(log_file) as file:
            return FileResponse(file)


@app.delete("/get_logs/", tags=["logs"])
def delete_logs(
    path: str,
) -> dict:
    """
    Delete the log file

    Args:
        path (str): Name of the folder (nebula+DFL+timestamp) where the log file is located

    Returns:
        dict: Name of the deleted file
    """
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", ".log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        log_file = min(log_files, key=lambda x: len(os.path.basename(x)))
        os.remove(log_file)
        return {"filename": log_file}


@app.get("/get_logs/debug/", tags=["logs"])
def get_debug_logs(
    path: str,
):
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", "debug.log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        with open(log_files.pop()) as file:
            return FileResponse(file)


@app.delete("/get_logs/debug/", tags=["logs"])
def delete_debug_logs(
    path: str,
) -> dict:
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", "debug.log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        log_file = log_files.pop()
        os.remove(log_file)
        return {"filename": log_file}


@app.get("/get_logs/error/", tags=["logs"])
def get_error_logs(
    path: str,
):
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", "error.log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        with open(log_files.pop()) as file:
            return FileResponse(file)


@app.delete("/get_logs/error/", tags=["logs"])
def delete_error_logs(
    path: str,
) -> dict:
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", "error.log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        log_file = log_files.pop()
        os.remove(log_file)
        return {"filename": log_file}


@app.get("/get_logs/training/", tags=["logs"])
def get_train_logs(
    path: str,
):
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", "training.log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        with open(log_files.pop()) as file:
            return FileResponse(file)


@app.delete("/get_logs/training/", tags=["logs"])
def delete_train_logs(
    path: str,
) -> dict:
    # check for path traversal
    if _LFI_sentry(path):
        raise HTTPException(status_code=404, detail="Item not found")
    else:
        log_files = _find_x_files(LOGS_FOLDER + path + "/", "training.log")
        if not log_files:
            raise HTTPException(status_code=404, detail="Log file not found")
        log_file = log_files.pop()
        os.remove(log_file)
        return {"filename": log_file}


# Metrics
@app.get("/metrics/", tags=["metrics"])
def get_metrics():
    # check for path traversal

    log_files = _find_x_files(METRICS_FOLDER, "")
    if not log_files:
        raise HTTPException(status_code=404, detail="Log file not found")
    return_files = []
    for file_name in log_files:
        with open(file_name) as file:
            return_files.append(FileResponse(file))


# Actions
@app.get("/run/", tags=["actions"])
async def run():
    """
    Lanza el entrenamiento con la configuración indicada en `path`
    """

    json_files = _find_x_files(CONFIG_FOLDER)

    # Check if there is a json file in the folder
    if len(json_files) != CONFIG_FILE_COUNT:
        raise HTTPException(status_code=404, detail="Config file not found")

    # Avoids running multiple training processes at the same time
    global TRAINING_PROC
    if TRAINING_PROC and TRAINING_PROC.returncode is None:
        raise HTTPException(status_code=409, detail="Training already running")

    # Creates the subprocess in a non-blocking way
    cmd = [
        "python",
        "/home/dietpi/prueba/nebula/nebula/node.py",
        json_files[0],
    ]
    TRAINING_PROC = await asyncio.create_subprocess_exec(*cmd)

    return {"pid": TRAINING_PROC.pid, "state": "running"}


# @app.get("/pause/", tags=["actions"])
# async def pause():
#     """
#     Send SIGSTOP to the training process to pause it.
#     """
#     global TRAINING_PROC
#     if not TRAINING_PROC or TRAINING_PROC.returncode is not None:
#         raise HTTPException(status_code=404, detail="No training running")

#     TRAINING_PROC.send_signal(signal.SIGSTOP)
#     return {"pid": TRAINING_PROC.pid, "state": "paused"}

@app.get("/stop/", tags=["actions"])
async def stop():
    """
    Send SIGTERM to the training process and wait for it to finish.
    """
    global TRAINING_PROC
    if not TRAINING_PROC or TRAINING_PROC.returncode is not None:
        raise HTTPException(status_code=404, detail="No training running")

    TRAINING_PROC.send_signal(signal.SIGTERM)
    await TRAINING_PROC.wait()
    pid = TRAINING_PROC.pid 
    TRAINING_PROC = None
    return {"pid": pid, "state": "stopped"}


@app.put(
    "/setup/",
    status_code=status.HTTP_201_CREATED,
    tags=["setup"],
    response_model=list,
)
def setup_new_run(
    config: Annotated[UploadFile, File()],
    global_test: Annotated[UploadFile, File()],
    train_set: Annotated[UploadFile, File()],
) -> list:
    """
    Upload three files (1 × *.json* + 2 × *.h5*), rewrite paths inside the JSON
    to match this node, validate neighbour IPs via Tailscale, then clear old
    configs/logs and save the new files.

    Errors
    ------
    • **409 Conflict** – training already running  
    • **400 Bad Request** – wrong extensions, neighbour IP mismatch, or JSON
      parse error.

    Returns
    -------
    list
        Names of the stored files (JSON first, then the two datasets).
    """

    # Concurrency guard
    global TRAINING_PROC
    if TRAINING_PROC and TRAINING_PROC.returncode is None:
        raise HTTPException(
            status_code=409,
            detail="Training already running; pause or stop it before uploading.",
        )

    # Extension validation
    if not config.filename.endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail=f"`{config.filename}` must have a .json extension.",
        )
    for ds in (global_test, train_set):
        if not ds.filename.endswith(".h5"):
            raise HTTPException(
                status_code=400,
                detail=f"`{ds.filename}` must have a .h5 extension.",
            )

    # Read & patch the JSON
    try:
        original_cfg = json.load(config.file)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON file: {exc}",
        )

    # Replace tracking paths 
    tracking = original_cfg.get("tracking_args", {})
    tracking["log_dir"] = LOGS_FOLDER.rstrip("/")
    tracking["config_dir"] = CONFIG_FOLDER.rstrip("/")
    original_cfg["tracking_args"] = tracking

    # Replace security paths
    sec = original_cfg.get("security_args", {})
    for key in ("certfile", "keyfile", "cafile"):
        if key in sec and sec[key]:
            basename = os.path.basename(sec[key])
            sec[key] = os.path.join(CERTS_FOLDER.rstrip("/"), basename)
    original_cfg["security_args"] = sec

    # Validate neighbour IPs
    neigh_str: str = (
        original_cfg.get("network_args", {})
        .get("neighbors", "")
        .strip()
    )
    # Extract plain IPs (ignore :port if present)
    requested_ips = {
        re.split(r":", entry)[0]
        for entry in neigh_str.split()
        if entry
    }

    if requested_ips:
        try:
            ts_out = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            ts_status = json.loads(ts_out.stdout)
            reachable_ips = set(ts_status.get("Self", {}).get("TailscaleIPs", []))
            for peer in ts_status.get("Peer", {}).values():
                reachable_ips.update(peer.get("TailscaleIPs", []))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not verify neighbours via Tailscale: {exc}",
            )

        missing = sorted(ip for ip in requested_ips if ip not in reachable_ips)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Neighbour IP(s) not reachable via Tailscale: {', '.join(missing)}",
            )

    # Remove old .json / .h5 files
    for fname in os.listdir(CONFIG_FOLDER):
        if fname.endswith((".json", ".h5")):
            try:
                os.remove(os.path.join(CONFIG_FOLDER, fname))
            except OSError:
                pass
            
    # Check if files were removed
    for fname in os.listdir(CONFIG_FOLDER):
        if fname.endswith((".json", ".h5")):
            raise HTTPException(
                status_code=400,
                detail=f"Could not delete old file: {fname}",
            )

    # Save the patched JSON 
    json_dest = os.path.join(CONFIG_FOLDER, config.filename)
    with open(json_dest, "wb") as dst:
        dst.write(json.dumps(original_cfg, indent=2).encode("utf-8"))

    # Save the datasets
    saved_files: list[str] = [config.filename]
    for uploaded in (global_test, train_set):
        dst_path = os.path.join(CONFIG_FOLDER, uploaded.filename)
        with open(dst_path, "wb") as dst:
            dst.write(uploaded.file.read())
        saved_files.append(uploaded.filename)
        print (f"Saved {uploaded.filename} to {dst_path}")

    # Purge all *.log files 
    for root, _, files in os.walk(LOGS_FOLDER):
        for fname in files:
            if fname.endswith(".log"):
                try:
                    os.remove(os.path.join(root, fname))
                except OSError:
                    pass
    
    # Check if files were removed
    for root, _, files in os.walk(LOGS_FOLDER):
        for fname in files:
            if fname.endswith(".log"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not delete old file: {fname}",
                )

    return saved_files
