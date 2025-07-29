import asyncio
import glob
import json
import os
import shutil
from nebula.utils import APIUtils
import docker
from nebula.controller.federation.federation_controller import FederationController
from nebula.controller.federation.utils_requests import factory_requests_path
from typing import Dict
from fastapi import Request
from nebula.config.config import Config
from nebula.core.utils.certificate import generate_ca_certificate
from nebula.core.utils.locker import Locker

class ProcessesFederationController(FederationController):
    def __init__(self, hub_url, logger):
        super().__init__(hub_url, logger)
        self._user = ""
        self.root_path = ""
        self.host_platform = ""
        self.config_dir = ""
        self.log_dir = ""
        self.cert_dir = ""
        self.advanced_analytics = ""
        self.url = ""
        self.config = Config(entity="FederationController")
        self.round_per_node = {}
        self.additionals: dict = {}
        self._last_file_index = 0
        self._deployment_lock = Locker("deployment_lock", async_lock=True)
        self._federation_round = 0

    """                                             ###############################
                                                    #      ENDPOINT CALLBACKS     #
                                                    ###############################
    """

    async def run_scenario(self, id: str, scenario_data: Dict, user: str):
        #TODO maintain files on memory, not read them again
        self._user = user
        await self._initialize_scenario(scenario_data)
        generate_ca_certificate(dir_path=self.cert_dir)
        await self._load_configuration_and_start_nodes()
        self._start_initial_nodes()

        return self.sb.get_scenario_name()
         
    async def stop_scenario(self, id: str = ""):
        """
        Stop running participant nodes by removing the scenario command files.

        This method deletes the 'current_scenario_commands.sh' (or '.ps1' on Windows)
        file associated with a scenario. Removing this file signals the nodes to stop
        by terminating their processes.

        Args:
            scenario_name (str, optional): The name of the scenario to stop. If None,
                all scenarios' command files will be removed.

        Notes:
            - If the environment variable NEBULA_CONFIG_DIR is not set, a default
              configuration directory path is used.
            - Supports both Linux/macOS ('.sh') and Windows ('.ps1') script files.
            - Any errors during file removal are logged with the traceback.
        """
        # When stopping the nodes, we need to remove the current_scenario_commands.sh file -> it will cause the nodes to stop using PIDs
        try:
            nebula_config_dir = os.environ.get("NEBULA_CONFIG_DIR")
            if not nebula_config_dir:
                current_dir = os.path.dirname(__file__)
                nebula_base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
                nebula_config_dir = os.path.join(nebula_base_dir, "app", "config")
                self.logger.info(f"NEBULA_CONFIG_DIR not found. Using default path: {nebula_config_dir}")

            if id:
                if os.environ.get("NEBULA_HOST_PLATFORM") == "windows":
                    scenario_commands_file = os.path.join(
                        nebula_config_dir, self.sb.get_scenario_name(), "current_scenario_commands.ps1"
                    )
                else:
                    scenario_commands_file = os.path.join(
                        nebula_config_dir, self.sb.get_scenario_name(), "current_scenario_commands.sh"
                    )
                if os.path.exists(scenario_commands_file):
                    os.remove(scenario_commands_file)
            else:
                if os.environ.get("NEBULA_HOST_PLATFORM") == "windows":
                    files = glob.glob(
                        os.path.join(nebula_config_dir, "**/current_scenario_commands.ps1"), recursive=True
                    )
                else:
                    files = glob.glob(
                        os.path.join(nebula_config_dir, "**/current_scenario_commands.sh"), recursive=True
                    )
                for file in files:
                    os.remove(file)
        except Exception as e:
            self.logger.exception(f"Error while removing current_scenario_commands.sh file: {e}")

    async def update_nodes(self, scenario_name: str, request: Request):
        config = await request.json()
        participant_idx = int(config["device_args"]["idx"])
        participant_round = int(config["federation_args"]["round"])
        self.round_per_node[participant_idx] = participant_round
        last_fed_round = self._federation_round
        self._federation_round = min(self.round_per_node.values())
        
        additionals_deployables = [
            idx
            for idx, round in self.additionals.items() 
            if self._federation_round >= round
        ]
        
        adds_deployed = set()
        # Only verify when federation round is updated
        if self._federation_round != last_fed_round:
            self.logger.info(f"Federation Round updating, current value: {self._federation_round}")
            # Ensure concurrency
            for index in additionals_deployables:
                if index in adds_deployed:
                    continue
                
                for idx, node in enumerate(self.config.participants):
                    if index == idx:
                        async with self._deployment_lock:
                            if index in self.additionals.keys():
                                self.logger.info(f"Deploying additional participant: {index}")
                                self._start_node(node, self._network_name, self._base_network_name, self._base, self._ip_last_index, additional=True)
                                self._ip_last_index += 1
                                self.additionals.pop(index)
                                adds_deployed.add(index)
                                
        request_body = await request.json()
        payload = {"scenario_name": scenario_name, "data": request_body}
        
        asyncio.create_task(self._send_to_hub("update", payload, scenario_name))
        
        return {"message": "Node updated successfully in Federation Controller"}

    async def node_done(self, scenario_name: str, request: Request):
        request_body = await request.json()
        payload = {"scenario_name": scenario_name, "data": request_body}
        asyncio.create_task(self._send_to_hub("done", payload, scenario_name))
        return {"message": "Nodes done"}

    """                                             ###############################
                                                    #       FUNCTIONALITIES       #
                                                    ###############################
    """
    
    async def _send_to_hub(self, path, payload, scenario_name=""):
        try:
            url_request = self._hub_url + factory_requests_path(path, scenario_name)
            await APIUtils.post(url_request, payload)
        except Exception as e:
            self.logger.info(f"Failed to send update to Hub: {e}")

    async def _initialize_scenario(self, scenario_data):
        # Initialize Scenario builder using scenario_data from user
        self.logger.info("🔧  Initializing Scenario Builder using scenario data")
        self.sb.set_scenario_data(scenario_data)
        scenario_name = self.sb.get_scenario_name()
        
        self.root_path = os.environ.get("NEBULA_ROOT_HOST")
        self.host_platform = os.environ.get("NEBULA_HOST_PLATFORM")
        self.config_dir = os.path.join(os.environ.get("NEBULA_CONFIG_DIR"), scenario_name)
        self.log_dir = os.environ.get("NEBULA_LOGS_DIR")
        self.cert_dir = os.environ.get("NEBULA_CERTS_DIR")
        self.advanced_analytics = os.environ.get("NEBULA_ADVANCED_ANALYTICS", "False") == "True"
        self.config = Config(entity="scenarioManagement")
        self.env_tag = os.environ.get("NEBULA_ENV_TAG", "dev")
        self.prefix_tag = os.environ.get("NEBULA_PREFIX_TAG", "dev")
        self.user_tag = os.environ.get("NEBULA_USER_TAG", os.environ.get("USER", "unknown"))
        
        self.url = f"127.0.0.1:{os.environ.get('NEBULA_FEDERATION_CONTROLLER_PORT')}"
        
        # Create Scenario management dirs
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, scenario_name), exist_ok=True)
        os.makedirs(self.cert_dir, exist_ok=True)

        # Give permissions to the directories
        os.chmod(self.config_dir, 0o777)
        os.chmod(os.path.join(self.log_dir, scenario_name), 0o777)
        os.chmod(self.cert_dir, 0o777)

        # Save the scenario configuration
        scenario_file = os.path.join(self.config_dir, "scenario.json")
        with open(scenario_file, "w") as f:
            json.dump(scenario_data, f, sort_keys=False, indent=2)

        os.chmod(scenario_file, 0o777)

        # Save management settings
        settings = {
            "scenario_name": scenario_name,
            "root_path": self.root_path,
            "config_dir": self.config_dir,
            "log_dir": self.log_dir,
            "cert_dir": self.cert_dir,
            "env": None,
        }

        settings_file = os.path.join(self.config_dir, "settings.json")
        with open(settings_file, "w") as f:
            json.dump(settings, f, sort_keys=False, indent=2)

        os.chmod(settings_file, 0o777)
        
        # Attacks assigment and mobility
        self.logger.info("🔧  Building general configuration")
        self.sb.build_general_configuration()
        self.logger.info("✅  Building general configuration done")
        
        # Create participant configs and .json
        for index, (_, node) in enumerate(self.sb.get_federation_nodes().items()):
            self.logger.info(f"Creating .json file for participant: {index}, Configuration: {node}")
            node_config = node
            try:
                participant_file = os.path.join(self.config_dir, f"participant_{node_config['id']}.json")
                self.logger.info(f"Filename: {participant_file}")
                os.makedirs(os.path.dirname(participant_file), exist_ok=True)
            except Exception as e:
                 self.logger.info(f"ERROR while creating files: {e}")
                 
            try:         
                participant_config = self.sb.build_scenario_config_for_node(index, node)
                #self.logger.info(f"dictionary: {participant_config}")
            except Exception as e:
                 self.logger.info(f"ERROR while building configuration for node: {e}")

            try:
                with open(participant_file, "w") as f:
                    json.dump(participant_config, f, sort_keys=False, indent=2)
                os.chmod(participant_file, 0o777)
            except Exception as e:
                 self.logger.info(f"ERROR while dumping configuration into files: {e}")

        self.logger.info("✅  Initializing Scenario Builder done")
                
    async def _load_configuration_and_start_nodes(self):
        self.logger.info("🔧  Loading Scenario configuration...")
        # Get participants configurations
        participant_files = glob.glob(f"{self.config_dir}/participant_*.json")
        participant_files.sort()
        if len(participant_files) == 0:
            raise ValueError("No participant files found in config folder")

        self.config.set_participants_config(participant_files)
        self.n_nodes = len(participant_files)
        self.logger.info(f"Number of nodes: {self.n_nodes}")
        
        self.sb.create_topology_manager(self.config)
        
        # Update participants configuration
        is_start_node = False
        config_participants = []
        
        additional_participants = self.sb.get_additional_nodes()
        additional_nodes = len(additional_participants) if additional_participants else 0
        self.logger.info(f"######## nodes: {self.n_nodes} + additionals: {additional_nodes} ######")
        
        participant_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # Initial participants
        self.logger.info("🔧  Building preload configuration for initial nodes...")
        for i in range(self.n_nodes):
            try:
                with open(f"{self.config_dir}/participant_" + str(i) + ".json") as f:
                    participant_config = json.load(f)
            except Exception as e:
                self.logger.info(f"ERROR: open/load participant .json")

            self.logger.info(f"Building preload conf for participant {i}")
            try:
                self.sb.build_preload_initial_node_configuration(i, participant_config, self.log_dir, self.config_dir, self.cert_dir, self.advanced_analytics)
            except Exception as e:
                self.logger.info(f"ERROR: cannot build preload configuration")

            try:
                with open(f"{self.config_dir}/participant_" + str(i) + ".json", "w") as f:
                    json.dump(participant_config, f, sort_keys=False, indent=2)
            except Exception as e:
                self.logger.info(f"ERROR: cannot dump preload configuration into participant .json file")

            config_participants.append((
                participant_config["network_args"]["ip"],
                participant_config["network_args"]["port"],
                participant_config["device_args"]["role"],
            ))
            if participant_config["device_args"]["start"]:
                if not is_start_node:
                    is_start_node = True
                else:
                    raise ValueError("Only one node can be start node")

        self.logger.info("✅  Building preload configuration for initial nodes done")
            
        self.config.set_participants_config(participant_files)
        
        # Add role to the topology (visualization purposes)
        self.sb.visualize_topology(config_participants, path=f"{self.config_dir}/topology.png", plot=False)
        
        # Additional participants
        self.logger.info("🔧  Building preload configuration for additional nodes...")
        additional_participants_files = []
        if additional_participants:
            last_participant_file = participant_files[-1]
            last_participant_index = len(participant_files)

            for i, _ in enumerate(additional_participants):
                additional_participant_file = f"{self.config_dir}/participant_{last_participant_index + i}.json"
                shutil.copy(last_participant_file, additional_participant_file)

                with open(additional_participant_file) as f:
                    participant_config = json.load(f)

                self.logger.info(f"Configuration | additional nodes |  participant: {self.n_nodes + i}")
                self.sb.build_preload_additional_node_configuration(last_participant_index, i, participant_config)
            
                with open(additional_participant_file, "w") as f:
                    json.dump(participant_config, f, sort_keys=False, indent=2)

                additional_participants_files.append(additional_participant_file)

        if additional_participants_files:
            self.config.add_participants_config(additional_participants_files)

        if additional_participants:
            self.n_nodes += len(additional_participants)

        self.logger.info("✅  Building preload configuration for additional nodes done")
        self.logger.info("✅  Loading Scenario configuration done")
        
        # Build dataset    
        dataset = self.sb.configure_dataset(self.config_dir)
        self.logger.info(f"🔧  Splitting {self.sb.get_dataset_name()} dataset...")
        dataset.initialize_dataset()
        self.logger.info(f"✅  Splitting {self.sb.get_dataset_name()} dataset... Done")

    def _start_initial_nodes(self):
        self.logger.info("Starting nodes as processes...")

        self.config.participants.sort(key=lambda x: x["device_args"]["idx"])
        self._ip_last_index = 2
        for idx, node in enumerate(self.config.participants):
            
            if node["deployment_args"]["additional"]:
                self.additionals[idx] = int(node["deployment_args"]["deployment_round"])
                self.logger.info(f"Participant {idx} is additional. Round of deployment: {int(node['deployment_args']['deployment_round'])}")
            else:
                # deploy initial nodes
                self.logger.info(f"Deployment starting for participant {idx}")
                self.round_per_node[idx] = 0
                self._start_node(node, self._network_name, self._base_network_name, self._base, self._ip_last_index)
                self._ip_last_index += 1
                        
    def _start_node(self, node, network_name, base_network_name, base, i, additional=False):
        self.processes_root_path = os.path.join(os.path.dirname(__file__), "..", "..")
        
        self.logger.info(f"env path: {self.env_path}")

        # Include additional config to the participants
        for idx, node in enumerate(self.config.participants):
            node["tracking_args"]["log_dir"] = os.path.join(self.root_path, "app", "logs")
            node["tracking_args"]["config_dir"] = os.path.join(self.root_path, "app", "config", self.sb.get_scenario_name())
            node["scenario_args"]["controller"] = self.url
            node["scenario_args"]["deployment"] = self.sb.get_deployment()
            node["security_args"]["certfile"] = os.path.join(
                self.root_path, "app", "certs", f"participant_{node['device_args']['idx']}_cert.pem"
            )
            node["security_args"]["keyfile"] = os.path.join(
                self.root_path, "app", "certs", f"participant_{node['device_args']['idx']}_key.pem"
            )
            node["security_args"]["cafile"] = os.path.join(self.root_path, "app", "certs", "ca_cert.pem")

            # Write the config file in config directory
            with open(f"{self.config_dir}/participant_{node['device_args']['idx']}.json", "w") as f:
                json.dump(node, f, indent=4)

        try:
            if self.host_platform == "windows":
                commands = """
                $ParentDir = Split-Path -Parent $PSScriptRoot
                $PID_FILE = "$PSScriptRoot\\current_scenario_pids.txt"
                New-Item -Path $PID_FILE -Force -ItemType File

                """
                sorted_participants = sorted(
                    self.config.participants,
                    key=lambda node: node["device_args"]["idx"],
                    reverse=True,
                )
                for node in sorted_participants:
                    if node["device_args"]["start"]:
                        commands += "Start-Sleep -Seconds 10\n"
                    else:
                        commands += "Start-Sleep -Seconds 2\n"

                    commands += f'Write-Host "Running node {node["device_args"]["idx"]}..."\n'
                    commands += f'$OUT_FILE = "{self.root_path}\\app\\logs\\{self.sb.get_scenario_name()}\\participant_{node["device_args"]["idx"]}.out"\n'
                    commands += f'$ERROR_FILE = "{self.root_path}\\app\\logs\\{self.sb.get_scenario_name()}\\participant_{node["device_args"]["idx"]}.err"\n'

                    # Use Start-Process for executing Python in background and capture PID
                    commands += f"""$process = Start-Process -FilePath "python" -ArgumentList "{self.root_path}\\nebula\\core\\node.py {self.root_path}\\app\\config\\{self.sb.get_scenario_name()}\\participant_{node["device_args"]["idx"]}.json" -PassThru -NoNewWindow -RedirectStandardOutput $OUT_FILE -RedirectStandardError $ERROR_FILE
                Add-Content -Path $PID_FILE -Value $process.Id
                """

                commands += 'Write-Host "All nodes started. PIDs stored in $PID_FILE"\n'

                with open(f"{self.config_dir}/current_scenario_commands.ps1", "w") as f:
                    f.write(commands)
                os.chmod(f"{self.config_dir}/current_scenario_commands.ps1", 0o755)
            else:
                commands = '#!/bin/bash\n\nPID_FILE="$(dirname "$0")/current_scenario_pids.txt"\n\n> $PID_FILE\n\n'
                sorted_participants = sorted(
                    self.config.participants,
                    key=lambda node: node["device_args"]["idx"],
                    reverse=True,
                )
                for node in sorted_participants:
                    if node["device_args"]["start"]:
                        commands += "sleep 10\n"
                    else:
                        commands += "sleep 2\n"
                    commands += f'echo "Running node {node["device_args"]["idx"]}..."\n'
                    commands += f"OUT_FILE={self.root_path}/app/logs/{self.sb.get_scenario_name()}/participant_{node['device_args']['idx']}.out\n"
                    commands += f"python {self.root_path}/nebula/core/node.py {self.root_path}/app/config/{self.sb.get_scenario_name()}/participant_{node['device_args']['idx']}.json &\n"
                    commands += "echo $! >> $PID_FILE\n\n"

                commands += 'echo "All nodes started. PIDs stored in $PID_FILE"\n'

                with open(f"{self.config_dir}/current_scenario_commands.sh", "w") as f:
                    f.write(commands)
                os.chmod(f"{self.config_dir}/current_scenario_commands.sh", 0o755)

        except Exception as e:
            raise Exception(f"Error starting nodes as processes: {e}")