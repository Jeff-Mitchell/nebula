import asyncio
import importlib
import json
import logging
import os
import sys
from typing import TYPE_CHECKING
from pathlib import Path


import aiohttp
import psutil

_TEXT_EXTS = {".txt", ".log", ".json", ".csv", ".yaml", ".yml"}

if TYPE_CHECKING:
    pass


class Reporter:

    _LOGS_DIR = (Path(__file__).resolve().parent / ".." / "app" / "logs").resolve()
    _MAX_JSON_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB

    def __init__(self, config, trainer):
        """
        Initializes the reporter module for sending periodic updates to a dashboard controller.

        This initializer sets up the configuration parameters required to report metrics and statistics
        about the network, participant, and trainer. It connects to a specified URL endpoint where
        these metrics will be logged, and it initializes values used for tracking network traffic.

        Args:
            config (dict): The configuration dictionary containing all setup parameters.
            trainer (Trainer): The trainer object responsible for managing training sessions.
            cm (CommunicationsManager): The communications manager handling network connections
                                        and interactions.

        Attributes:
            frequency (int): The frequency at which the reporter sends updates.
            grace_time (int): Grace period before starting the reporting.
            data_queue (Queue): An asyncio queue for managing data to be reported.
            url (str): The endpoint URL for reporting updates.
            counter (int): Counter for tracking the number of reports sent.
            first_net_metrics (bool): Flag indicating if this is the first collection of network metrics.
            prev_bytes_sent (int), prev_bytes_recv (int), prev_packets_sent (int), prev_packets_recv (int):
                Metrics for tracking network data sent and received.
            acc_bytes_sent (int), acc_bytes_recv (int), acc_packets_sent (int), acc_packets_recv (int):
                Accumulators for network traffic.

        Raises:
            None

        Notes:
            - Logs the start of the reporter module.
            - Initializes both current and accumulated metrics for traffic monitoring.
        """
        logging.info("Starting reporter module")
        self._cm = None
        self.config = config
        cfg_logs_dir = Path(self.config.participant["tracking_args"]["log_dir"]).expanduser().resolve()
        self.__class__._LOGS_DIR = cfg_logs_dir
        self.trainer = trainer
        self.frequency = self.config.participant["reporter_args"]["report_frequency"]
        self.grace_time = self.config.participant["reporter_args"]["grace_time_reporter"]
        self.data_queue = asyncio.Queue()
        self.url = f"http://{self.config.participant['scenario_args']['controller']}/nodes/{self.config.participant['scenario_args']['name']}/update"
        self.counter = 0

        self.first_net_metrics = True
        self.prev_bytes_sent = 0
        self.prev_bytes_recv = 0
        self.prev_packets_sent = 0
        self.prev_packets_recv = 0

        self.acc_bytes_sent = 0
        self.acc_bytes_recv = 0
        self.acc_packets_sent = 0
        self.acc_packets_recv = 0
        self._running = asyncio.Event()
        self._final_metrics_sent = False
        self._reporter_task = None  # Track the background task

    @property
    def cm(self):
        if not self._cm:
            from nebula.core.network.communications import CommunicationsManager

            self._cm = CommunicationsManager.get_instance()
            return self._cm
        else:
            return self._cm

    async def enqueue_data(self, name, value):
        """
        Asynchronously enqueues data for reporting.

        This function adds a named data value pair to the data queue, which will later be processed
        and sent to the designated reporting endpoint. The queue enables handling of reporting tasks
        independently of other processes.

        Args:
            name (str): The name or identifier for the data item.
            value (Any): The value of the data item to be reported.

        Returns:
            None

        Notes:
            - This function is asynchronous to allow non-blocking data enqueueing.
            - Uses asyncio's queue to manage data, ensuring concurrency.
        """
        await self.data_queue.put((name, value))

    async def start(self):
        """
        Starts the reporter module after a grace period.

        This asynchronous function initiates the reporting process following a designated grace period.
        It creates a background task to run the reporting loop, allowing data to be reported at defined intervals.

        Returns:
            asyncio.Task: The task for the reporter loop, which handles the data reporting asynchronously.

        Notes:
            - The grace period allows for a delay before the first reporting cycle.
            - The reporter loop runs in the background, ensuring continuous data updates.
        """
        self._running.set()
        await asyncio.sleep(self.grace_time)
        self._reporter_task = asyncio.create_task(self.run_reporter(), name="Reporter_run_reporter")
        return self._reporter_task

    async def run_reporter(self):
        """
        Runs the continuous reporting loop.

        This asynchronous function performs periodic reporting tasks such as reporting resource usage,
        data queue contents, and, optionally, status updates to the controller. The loop runs indefinitely,
        updating the counter with each cycle to track the frequency of specific tasks.

        Key Actions:
            - Regularly reports the resource status.
            - Reloads the configuration file every 50 cycles to reflect any updates.

        Notes:
            - The reporting frequency is determined by the 'report_frequency' setting in the config file.
        """
        while self._running.is_set():
            if self.config.participant["reporter_args"]["report_status_data_queue"]:
                if self.config.participant["scenario_args"]["controller"] != "nebula-test":
                    await self.__report_status_to_controller()
                await self.__report_data_queue()
            await self.__report_resources()
            self.counter += 1
            if self.counter % 50 == 0:
                logging.info("Reloading config file...")
                self.cm.engine.config.reload_config_file()
            await asyncio.sleep(self.frequency)

    async def report_scenario_finished(self):
        """
        Reports the scenario completion status to the controller.

        This asynchronous function notifies the scenario controller that the participant has finished
        its tasks. It sends a POST request to the designated controller URL, including the participant's
        ID in the JSON payload.

        URL Construction:
            - The URL is dynamically built using the controller address and scenario name
              from the configuration settings.

        Parameters:
            - idx (int): The unique identifier for this participant, sent in the request data.

        Returns:
            - bool: True if the report was successful (status 200), False otherwise.

        Error Handling:
            - Logs an error if the response status is not 200, indicating that the controller
              might be temporarily overloaded.
            - Logs exceptions if the connection attempt to the controller fails.
        """

        # ───── Send final metrics once ─────
        if not self._final_metrics_sent:
            await self.__send_final_metrics()
            self._final_metrics_sent = True

        url = f"http://{self.config.participant['scenario_args']['controller']}/nodes/{self.config.participant['scenario_args']['name']}/done"
        data = json.dumps({"idx": self.config.participant["device_args"]["idx"]})
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"NEBULA Participant {self.config.participant['device_args']['idx']}",
        }
        try:
            await self.__report_status_to_controller()
        except Exception as e:
            logging.exception(f"Error reporting status before scenario finished: {e}")
        try:
            async with aiohttp.ClientSession() as session, session.post(url, data=data, headers=headers) as response:
                if response.status != 200:
                    logging.error(
                        f"Error received from controller: {response.status} (probably there is overhead in the controller, trying again in the next round)"
                    )
                    text = await response.text()
                    logging.debug(text)
                else:
                    logging.info(
                        f"Participant {self.config.participant['device_args']['idx']} reported scenario finished"
                    )
                    return True
        except aiohttp.ClientError:
            logging.exception(f"Error connecting to the controller at {url}")
        return False

    async def stop(self):
        logging.info("🔍  Stopping reporter module...")
        self._running.clear()

        # Cancel the background task
        if self._reporter_task and not self._reporter_task.done():
            logging.info("🛑  Cancelling Reporter background task...")
            self._reporter_task.cancel()
            try:
                await self._reporter_task
            except asyncio.CancelledError:
                pass
            self._reporter_task = None
            logging.info("🛑  Reporter background task cancelled")
        await self.report_scenario_finished()

    # Final metrics
    async def __send_final_metrics(self) -> bool:
        latest_dir = await asyncio.to_thread(self.__get_latest_metrics_dir)
        if latest_dir is None:
            logging.warning("No metrics directory found in %s", self._LOGS_DIR)
            return False

        metrics = await asyncio.to_thread(self.__collect_metrics_from_dir, latest_dir)

        body = json.dumps(metrics, ensure_ascii=False)
        compressed = False
        if len(body.encode()) > self._MAX_JSON_SIZE_BYTES:
            body = await asyncio.to_thread(self.__gzip_b64, body.encode())
            compressed = True

        payload = {
            "idx": self.config.participant["device_args"]["idx"],
            "ip":  self.config.participant["network_args"]["ip"],
            "compressed": compressed,
            "metrics": body,
        }

        url = f"http://{self.config.participant['scenario_args']['controller']}/nodes/{self.config.participant['scenario_args']['name']}/metrics"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"NEBULA Participant {self.config.participant['device_args']['idx']}",
        }

        logging.info("Sending final metrics (compressed=%s)…", compressed)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=json.dumps(payload), headers=headers) as resp:
                    if resp.status != 200:
                        logging.error("Controller did not respond %s; metrics discarded", resp.status)
                        logging.debug(await resp.text())
                        return False
                    logging.info("Final metrics have been sended correctly")
                    return True
        except aiohttp.ClientError:
            logging.exception("Could not contact with the controller for final metrics sending")
            return False

    # Directory utilities 
    @classmethod
    def __get_latest_metrics_dir(cls):
        """
        Returns the folder containing the metrics from the most recent run.

        Expected structure:
            <log_dir>/
                nebula_DFL_YYYY_mm_dd_hh_mm_ss/       <- most recent run
                    metrics/                          <- metrics are stored here
                        participant_<n>/...
        """
        try:
            runs = [d for d in cls._LOGS_DIR.iterdir() if d.is_dir()]
            if not runs:
                return None

            latest_run   = max(runs, key=lambda p: p.stat().st_mtime)
            metrics_root = latest_run / "metrics"

            # if for any reason the "metrics" subdirectory does not exist,
            # return the full run directory to avoid breaking the logic.
            return metrics_root if metrics_root.is_dir() else latest_run

        except Exception as exc:
            logging.exception("Error while searching for metrics directory: %s", exc)
            return None

    @staticmethod
    def __collect_metrics_from_dir(base_dir: Path) -> dict:
        """
        relative_path -> {
            "binary": bool,
            "data": str,          # UTF-8 text or base64 if binary
            "mime": str | None    # optional hint for binaries
        }
        """
        import base64, mimetypes

        metrics: dict[str, dict] = {}

        for file in base_dir.rglob("*"):
            if not file.is_file():
                continue

            rel_path = str(file.relative_to(base_dir))
            ext = file.suffix.lower()

            try:
                if ext in _TEXT_EXTS:
                    metrics[rel_path] = {
                        "binary": False,
                        "data": file.read_text(errors="replace"),
                        "mime": "text/plain",
                    }
                else:
                    metrics[rel_path] = {
                        "binary": True,
                        "data": base64.b64encode(file.read_bytes()).decode(),
                        "mime": mimetypes.guess_type(file.name)[0] or "application/octet-stream",
                    }
            except Exception as exc:           # noqa: BLE001
                logging.warning("Could not read %s — %s", file, exc)

        return metrics

    @staticmethod
    def __gzip_b64(data: bytes) -> str:
        import base64, gzip, io
        with io.BytesIO() as bio:
            with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
                gz.write(data)
            return base64.b64encode(bio.getvalue()).decode()

    async def __report_data_queue(self):
        """
        Processes and reports queued data entries.

        This asynchronous function iterates over the data queue, retrieving each name-value pair
        and sending it to the trainer's logging mechanism. Once logged, each item is marked as done.

        Functionality:
            - Retrieves and logs all entries in the data queue until it is empty.
            - Assumes that `log_data` can handle asynchronous execution for optimal performance.

        Parameters:
            - name (str): The identifier for the data entry (e.g., metric name).
            - value (Any): The value of the data entry to be logged.

        Returns:
            - None

        Notes:
            - Each processed item is marked as done in the queue.
        """

        while not self.data_queue.empty():
            name, value = await self.data_queue.get()
            await self.trainer.logger.log_data({name: value})  # Assuming log_data can be made async
            self.data_queue.task_done()

    async def __report_status_to_controller(self):
        """
        Sends the participant's status to the controller.

        This asynchronous function transmits the current participant configuration to the controller's
        URL endpoint. It handles both client and general exceptions to ensure robust communication
        with the controller, retrying in case of errors.

        Functionality:
            - Initiates a session to post participant data to the controller.
            - Logs the response status, indicating issues when status is non-200.
            - Retries after a short delay in case of connection errors or unhandled exceptions.

        Parameters:
            - None (uses internal `self.config.participant` data to build the payload).

        Returns:
            - None

        Notes:
            - Uses the participant index to specify the User-Agent in headers.
            - Delays for 5 seconds upon general exceptions to avoid rapid retry loops.
        """
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    self.url,
                    data=json.dumps(self.config.participant),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": f"NEBULA Participant {self.config.participant['device_args']['idx']}",
                    },
                ) as response,
            ):
                if response.status != 200:
                    logging.error(
                        f"Error received from controller: {response.status} (probably there is overhead in the controller, trying again in the next round)"
                    )
                    text = await response.text()
                    logging.debug(text)
        except aiohttp.ClientError:
            logging.exception(f"Error connecting to the controller at {self.url}")
        except Exception:
            logging.exception("Error sending status to controller, will try again in a few seconds")
            await asyncio.sleep(5)

    async def __report_resources(self):
        """
        Reports system resource usage metrics.

        This asynchronous function gathers and logs CPU usage data for the participant's device,
        and attempts to retrieve the CPU temperature (Linux systems only). Additionally, it measures
        CPU usage specifically for the current process.

        Functionality:
            - Gathers total CPU usage (percentage) and attempts to retrieve CPU temperature.
            - Uses `psutil` for non-blocking access to system data on Linux.
            - Records CPU usage of the current process for finer monitoring.

        Parameters:
            - None

        Notes:
            - On non-Linux platforms, CPU temperature will default to 0.
            - Uses `asyncio.to_thread` to call CPU and sensor readings without blocking the event loop.
        """
        cpu_percent = psutil.cpu_percent()
        cpu_temp = 0
        try:
            if sys.platform == "linux":
                sensors = await asyncio.to_thread(psutil.sensors_temperatures)
                cpu_temp = sensors.get("coretemp")[0].current if sensors.get("coretemp") else 0
        except Exception:  # noqa: S110
            pass

        pid = os.getpid()
        cpu_percent_process = await asyncio.to_thread(psutil.Process(pid).cpu_percent, interval=None)

        process = psutil.Process(pid)
        memory_process = await asyncio.to_thread(lambda: process.memory_info().rss / (1024**2))
        memory_percent_process = process.memory_percent()
        memory_info = await asyncio.to_thread(psutil.virtual_memory)
        memory_percent = memory_info.percent
        memory_used = memory_info.used / (1024**2)

        disk_percent = psutil.disk_usage("/").percent

        net_io_counters = await asyncio.to_thread(psutil.net_io_counters)
        bytes_sent = net_io_counters.bytes_sent
        bytes_recv = net_io_counters.bytes_recv
        packets_sent = net_io_counters.packets_sent
        packets_recv = net_io_counters.packets_recv

        if self.first_net_metrics:
            bytes_sent_diff = 0
            bytes_recv_diff = 0
            packets_sent_diff = 0
            packets_recv_diff = 0
            self.first_net_metrics = False
        else:
            bytes_sent_diff = bytes_sent - self.prev_bytes_sent
            bytes_recv_diff = bytes_recv - self.prev_bytes_recv
            packets_sent_diff = packets_sent - self.prev_packets_sent
            packets_recv_diff = packets_recv - self.prev_packets_recv

        self.prev_bytes_sent = bytes_sent
        self.prev_bytes_recv = bytes_recv
        self.prev_packets_sent = packets_sent
        self.prev_packets_recv = packets_recv

        self.acc_bytes_sent += bytes_sent_diff
        self.acc_bytes_recv += bytes_recv_diff
        self.acc_packets_sent += packets_sent_diff
        self.acc_packets_recv += packets_recv_diff

        current_connections = await self.cm.get_addrs_current_connections(only_direct=True)

        resources = {
            "W-CPU/CPU global (%)": cpu_percent,
            "W-CPU/CPU process (%)": cpu_percent_process,
            "W-CPU/CPU temperature (°)": cpu_temp,
            "Z-RAM/RAM global (%)": memory_percent,
            "Z-RAM/RAM global (MB)": memory_used,
            "Z-RAM/RAM process (%)": memory_percent_process,
            "Z-RAM/RAM process (MB)": memory_process,
            "Y-Disk/Disk (%)": disk_percent,
            "X-Network/Network (MB sent)": round(self.acc_bytes_sent / (1024**2), 3),
            "X-Network/Network (MB received)": round(self.acc_bytes_recv / (1024**2), 3),
            "X-Network/Network (packets sent)": self.acc_packets_sent,
            "X-Network/Network (packets received)": self.acc_packets_recv,
            "X-Network/Connections": len(current_connections),
        }
        self.trainer.logger.log_data(resources)

        if importlib.util.find_spec("pynvml") is not None:
            try:
                import pynvml

                await asyncio.to_thread(pynvml.nvmlInit)
                devices = await asyncio.to_thread(pynvml.nvmlDeviceGetCount)
                for i in range(devices):
                    handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, i)
                    gpu_percent = (await asyncio.to_thread(pynvml.nvmlDeviceGetUtilizationRates, handle)).gpu
                    gpu_temp = await asyncio.to_thread(
                        pynvml.nvmlDeviceGetTemperature,
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                    gpu_mem = await asyncio.to_thread(pynvml.nvmlDeviceGetMemoryInfo, handle)
                    gpu_mem_percent = round(gpu_mem.used / gpu_mem.total * 100, 3)
                    gpu_power = await asyncio.to_thread(pynvml.nvmlDeviceGetPowerUsage, handle) / 1000.0
                    gpu_clocks = await asyncio.to_thread(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM)
                    gpu_memory_clocks = await asyncio.to_thread(
                        pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM
                    )
                    gpu_fan_speed = await asyncio.to_thread(pynvml.nvmlDeviceGetFanSpeed, handle)
                    gpu_info = {
                        f"W-GPU/GPU{i} (%)": gpu_percent,
                        f"W-GPU/GPU{i} temperature (°)": gpu_temp,
                        f"W-GPU/GPU{i} memory (%)": gpu_mem_percent,
                        f"W-GPU/GPU{i} power": gpu_power,
                        f"W-GPU/GPU{i} clocks": gpu_clocks,
                        f"W-GPU/GPU{i} memory clocks": gpu_memory_clocks,
                        f"W-GPU/GPU{i} fan speed": gpu_fan_speed,
                    }
                    self.trainer.logger.log_data(gpu_info)
            except Exception:  # noqa: S110
                pass