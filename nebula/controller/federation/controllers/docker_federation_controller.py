import glob
import json
import os
import shutil
from nebula.utils import DockerUtils
import docker
from nebula.controller.federation.federation_controller import FederationController
from typing import Dict
from fastapi import Request, Response
from nebula.config.config import Config
from nebula.core.utils.certificate import generate_ca_certificate
from nebula.core.utils.locker import Locker


class DockerFederationController(FederationController):
    
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
        self.config = Config(entity="scenarioManagement")
        self.round_per_node = {}
        self.additionals: dict = {}
        self._ip_last_index = 0
        self._network_name = ""
        self._base_network_name = ""
        self._base = ""
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
         
    async def stop_scenario(self, id: str):
        """
        Remove all participant containers and the scenario network.
        Reads ALL scenario.metadata and removes all listed containers and the network, then deletes the metadata file.
        Also forcibly stops and removes any containers still attached to the network before removing it.
        """
        # Try multiple possible config directory locations. This depends on where the user called the function from.
        possible_config_dirs = [
            os.environ.get("NEBULA_CONFIG_DIR"),
            "/nebula/app/config",
            "./app/config",
            os.path.join(os.getcwd(), "app", "config"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app", "config"),
        ]

        config_dir = None
        for dir_path in possible_config_dirs:
            if dir_path and os.path.exists(dir_path):
                config_dir = dir_path
                break

        if not config_dir:
            self.logger.warning("No valid config directory found, skipping cleanup")
            return

        scenario_dirs = []
        self.logger.info(f"Config directory: {config_dir}")
        if os.path.exists(config_dir):
            for item in os.listdir(config_dir):
                scenario_path = os.path.join(config_dir, item)
                if os.path.isdir(scenario_path):
                    metadata_file = os.path.join(scenario_path, "scenario.metadata")
                    if os.path.exists(metadata_file):
                        scenario_dirs.append(scenario_path)

        self.logger.info(f"Removing scenario containers for {scenario_dirs}")
        if not scenario_dirs:
            self.logger.info("No active scenarios found to clean up")
            return

        client = docker.from_env()

        for scenario_dir in scenario_dirs:
            metadata_path = os.path.join(scenario_dir, "scenario.metadata")
            if not os.path.exists(metadata_path):
                self.logger.info(f"Skipping {scenario_dir} - no scenario.metadata found")
                continue

            with open(metadata_path) as f:
                meta = json.load(f)

            # Remove containers listed in metadata
            for name in meta.get("containers", []):
                try:
                    container = client.containers.get(name)
                    container.remove(force=True)
                    self.logger.info(f"Removed scenario container {name}")
                except Exception as e:
                    self.logger.warning(f"Could not remove scenario container {name}: {e}")

            # Remove network, but first forcibly remove any containers still attached
            network_name = meta.get("network")
            if network_name:
                try:
                    network = client.networks.get(network_name)
                    attached_containers = network.attrs.get("Containers") or {}
                    for container_id in attached_containers:
                        try:
                            c = client.containers.get(container_id)
                            c.remove(force=True)
                            self.logger.info(f"Force-removed container {c.name} attached to {network_name}")
                        except Exception as e:
                            self.logger.warning(f"Could not force-remove container {container_id}: {e}")
                    network.remove()
                    self.logger.info(f"Removed scenario network {network_name}")
                except Exception as e:
                    self.logger.warning(f"Could not remove scenario network {network_name}: {e}")

            # Remove metadata file
            try:
                os.remove(metadata_path)
            except Exception as e:
                self.logger.warning(f"Could not remove scenario.metadata: {e}")

    async def update_nodes(self, scenario_name: str, request: Request):
        config = await request.json()
        participant_idx = int(config["device_args"]["idx"])
        participant_round = int(config["federation_args"]["round"])
        self.logger.info(f"Update received from participant: {participant_idx}, round: {participant_round}")
        self.logger.info(f"Update: {self.round_per_node.items()}")
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
            self.logger.info(f"Federation Round updates, current value: {self._federation_round}")
            # Ensure concurrency
            for index in additionals_deployables:
                if index in adds_deployed:
                    continue
                
                for idx, node in enumerate(self.config.participants):
                    if index == idx:
                        async with self._deployment_lock:
                            self.logger.info(f"Deploying additional participant: {index}")
                            self._start_node(node, self._network_name, self._base_network_name, self._base, self._ip_last_index, additional=True)
                            self._ip_last_index += 1
                            adds_deployed.add(index)
        
        #TODO return the others parameters
        return Response(content=json.dumps({"message": "Node updated successfully"}), status_code=200)

    """                                             ###############################
                                                    #       FUNCTIONALITIES       #
                                                    ###############################
    """

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
        
        self.url = f"{os.environ.get('NEBULA_CONTROLLER_HOST')}:{os.environ.get('NEBULA_FEDERATION_CONTROLLER_PORT')}"
        
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

    def get_network_name(self, suffix: str) -> str:
        """
        Generate a standardized network name using tags.
        Args:
            suffix (str): Suffix for the network (default: 'net-base').
        Returns:
            str: The composed network name.
        """
        return f"{self.env_tag}_{self.prefix_tag}_{self.user_tag}_{suffix}"
    
    def get_participant_container_name(self, idx: int) -> str:
        """
        Generate a standardized container name for a participant using tags.
        Args:
            idx (int): The participant index.
        Returns:
            str: The composed container name.
        """
        return f"{self.env_tag}_{self.prefix_tag}_{self.user_tag}_{self.sb.get_scenario_name()}_participant{idx}"

    def _start_initial_nodes(self):
        self.logger.info("Starting nodes using Docker Compose...")
        self._network_name = self.get_network_name(f"{self.sb.get_scenario_name()}-net-scenario")
        self._base_network_name = self.get_network_name("net-base")

        # Create the Docker network
        self._base = DockerUtils.create_docker_network(self._network_name)
  
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
        client = docker.from_env()

        self.config.participants.sort(key=lambda x: x["device_args"]["idx"])
        container_ids = []
        container_names = []  # Track names for metadata

        image = "nebula-core"
        name = self.get_participant_container_name(node["device_args"]["idx"])
        if node["device_args"]["accelerator"] == "gpu":
            environment = {
                "NVIDIA_DISABLE_REQUIRE": True,
                "NEBULA_LOGS_DIR": "/nebula/app/logs/",
                "NEBULA_CONFIG_DIR": "/nebula/app/config/",
            }
            host_config = client.api.create_host_config(
                binds=[f"{self.root_path}:/nebula", "/var/run/docker.sock:/var/run/docker.sock"],
                privileged=True,
                device_requests=[docker.types.DeviceRequest(driver="nvidia", count=-1, capabilities=[["gpu"]])],
                extra_hosts={"host.docker.internal": "host-gateway"},
            )
        else:
            environment = {"NEBULA_LOGS_DIR": "/nebula/app/logs/", "NEBULA_CONFIG_DIR": "/nebula/app/config/"}
            host_config = client.api.create_host_config(
                binds=[f"{self.root_path}:/nebula", "/var/run/docker.sock:/var/run/docker.sock"],
                privileged=True,
                device_requests=[],
                extra_hosts={"host.docker.internal": "host-gateway"},
            )
        volumes = ["/nebula", "/var/run/docker.sock"]
        start_command = "sleep 10" if node["device_args"]["start"] else "sleep 0"
        command = [
            "/bin/bash",
            "-c",
            f"{start_command} && ifconfig && echo '{base}.1 host.docker.internal' >> /etc/hosts && python /nebula/nebula/core/node.py /nebula/app/config/{self.sb.get_scenario_name()}/participant_{node['device_args']['idx']}.json",
        ]
        networking_config = client.api.create_networking_config({
            network_name: client.api.create_endpoint_config(
                ipv4_address=f"{base}.{i}",
            ),
            base_network_name: client.api.create_endpoint_config(),
        })
        node["tracking_args"]["log_dir"] = "/nebula/app/logs"
        node["tracking_args"]["config_dir"] = f"/nebula/app/config/{self.sb.get_scenario_name()}"
        node["scenario_args"]["controller"] = self.url
        node["scenario_args"]["deployment"] = self.sb.get_deployment()
        node["security_args"]["certfile"] = f"/nebula/app/certs/participant_{node['device_args']['idx']}_cert.pem"
        node["security_args"]["keyfile"] = f"/nebula/app/certs/participant_{node['device_args']['idx']}_key.pem"
        node["security_args"]["cafile"] = "/nebula/app/certs/ca_cert.pem"
        node = json.loads(json.dumps(node).replace("192.168.50.", f"{base}."))  # TODO change this
        try:
            existing = client.containers.get(name)
            self.logger.warning(f"Container {name} already exists. Deployment may fail or cause conflicts.")
        except docker.errors.NotFound:
            pass  # No conflict, safe to proceed
        # Write the config file in config directory
        with open(f"{self.config_dir}/participant_{node['device_args']['idx']}.json", "w") as f:
            json.dump(node, f, indent=4)
        try:
            container_id = client.api.create_container(
                image=image,
                name=name,
                detach=True,
                volumes=volumes,
                environment=environment,
                command=command,
                host_config=host_config,
                networking_config=networking_config,
            )
        except Exception as e:
            self.logger.exception(f"Creating container {name}: {e}")
        try:
            client.api.start(container_id)
            container_ids.append(container_id)
            container_names.append(name)
        except Exception as e:
            self.logger.exception(f"Starting participant {name} error: {e}")

        # Write scenario-level metadata for cleanup
        scenario_metadata = {"containers": container_names, "network": network_name}
        if not additional:
            with open(os.path.join(self.config_dir, "scenario.metadata"), "w") as f:
                json.dump(scenario_metadata, f, indent=2)
        else:
            with open(os.path.join(self.config_dir, "scenario.metadata"), "a") as f:
                json.dump(scenario_metadata, f, indent=2)