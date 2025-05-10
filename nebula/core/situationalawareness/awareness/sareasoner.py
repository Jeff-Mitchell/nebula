from __future__ import annotations
import copy
from abc import abstractmethod, ABC
import asyncio
import logging
from nebula.addons.functions import print_msg_box
import importlib.util
import os
from nebula.core.situationalawareness.awareness.suggestionbuffer import SuggestionBuffer
from nebula.core.situationalawareness.awareness.sautils.sacommand import SACommand
from nebula.core.utils.locker import Locker
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import RoundEndEvent, AggregationEvent
from nebula.core.network.communications import CommunicationsManager
from nebula.core.situationalawareness.awareness.sautils.sasystemmonitor import SystemMonitor
from nebula.core.situationalawareness.awareness.arbitatrionpolicies.arbitatrionpolicy import factory_arbitatrion_policy
from nebula.core.situationalawareness.situationalawareness import ISAReasoner, ISADiscovery
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nebula.core.situationalawareness.awareness.sanetwork.sanetwork import SANetwork
    
class SAMComponent(ABC):
    @abstractmethod
    async def init(self):
        raise NotImplementedError
    @abstractmethod
    async def sa_component_actions(self):
        raise NotImplementedError


class SAReasoner(ISAReasoner):
    MODULE_PATH = "nebula/nebula/core/situationalawareness/awareness"

    def __init__(
        self,
        config,
    ):
        print_msg_box(
            msg=f"Starting Situational Awareness Reasoner module...",
            indent=2,
            title="SA Reasoner",
        )
        logging.info("🌐  Initializing SAReasoner")
        self._config = copy.deepcopy(config.participant)
        self._addr = config.participant["network_args"]["addr"]
        self._topology = config.participant["mobility_args"]["topology_type"]
        self._situational_awareness_network = None
        self._situational_awareness_training = None
        self._restructure_process_lock = Locker(name="restructure_process_lock")
        self._restructure_cooldown = 0
        self._arbitrator_notification = asyncio.Event()
        self._suggestion_buffer = SuggestionBuffer(self._arbitrator_notification, verbose=True)
        self._communciation_manager = CommunicationsManager.get_instance()
        self._sys_monitor = SystemMonitor()
        arb_pol = config.participant["situational_awareness"]["sa_reasoner"]["arbitatrion_policy"]
        self._arbitatrion_policy = factory_arbitatrion_policy(arb_pol, True)
        self._sa_components: dict[str, SAMComponent] = {}
        self._sa_discovery: ISADiscovery = None
        self._verbose = config.participant["situational_awareness"]["sa_reasoner"]["verbose"]
        
    @property
    def san(self) -> SANetwork:
        """Situational Awareness Network"""
        return self._situational_awareness_network
    
    @property
    def cm(self):
        return self._communciation_manager
    
    @property
    def sb(self):
        """Suggestion Buffer"""
        return self._suggestion_buffer
    
    @property
    def ab(self):
        """Arbitatrion Policy"""
        return self._arbitatrion_policy
    
    @property
    def sad(self) -> ISADiscovery:
        """SA Discovery"""
        return self._sa_discovery
    
    async def init(self, sa_discovery):
        self._sa_discovery: ISADiscovery = sa_discovery
        await self._loading_sa_components()
        #from nebula.core.situationalawareness.awareness.sanetwork.sanetwork import SANetwork
        #from nebula.core.situationalawareness.awareness.satraining.satraining import SATraining
        #self._situational_awareness_training = SATraining(self, self._addr, "qds", "fastreboot", verbose=True)
        await EventManager.get_instance().subscribe_node_event(RoundEndEvent, self._process_round_end_event)
        await EventManager.get_instance().subscribe_node_event(AggregationEvent, self._process_aggregation_event)
        #self._situational_awareness_network = SANetwork(self, self._addr, self._topology, verbose=True)
        #await self.san.init()
        
    def is_additional_participant(self):
        return self._config["mobility_args"]["additional_node"]["status"]

    """                                                     ###############################
                                                            #    REESTRUCTURE TOPOLOGY    #
                                                            ###############################
    """

    def get_restructure_process_lock(self):
        return self.san.get_restructure_process_lock()

    """                                                     ###############################
                                                            #          SA NETWORK         #
                                                            ###############################
    """

    def get_nodes_known(self, neighbors_too=False, neighbors_only=False):
        return self.san.get_nodes_known(neighbors_too, neighbors_only)

    def accept_connection(self, source, joining=False):
        return self.san.accept_connection(source, joining)

    def get_actions(self):
        return self.san.get_actions()

    """                                                     ###############################
                                                            #         ARBITRATION         #
                                                            ###############################
    """
    
    async def _process_round_end_event(self, ree : RoundEndEvent):
        logging.info("🔄 Arbitration | Round End Event...")
        for sa_comp in self._sa_components.values():
            asyncio.create_task(sa_comp.sa_component_actions())
        valid_commands = await self._arbitatrion_suggestions(RoundEndEvent)

        # Execute SACommand selected
        for cmd in valid_commands:
            if cmd.is_parallelizable():
                if self._verbose: logging.info(f"going to execute parallelizable action: {cmd.get_action()} made by: {await cmd.get_owner()}")
                asyncio.create_task(cmd.execute())
            else:
                if self._verbose: logging.info(f"going to execute action: {cmd.get_action()} made by: {await cmd.get_owner()}")
                await cmd.execute()

    async def _process_aggregation_event(self, age : AggregationEvent):
        logging.info("🔄 Arbitration | Aggregation Event...")
        aggregation_command = await self._arbitatrion_suggestions(AggregationEvent)
        if len(aggregation_command):
            if self._verbose: logging.info(f"Aggregation event resolved. SA Agente that suggest action: {await aggregation_command[0].get_owner}") 
            final_updates = await aggregation_command[0].execute()
            age.update_updates(final_updates)

    async def _arbitatrion_suggestions(self, event_type):
        if self._verbose: logging.info("Waiting for all suggestions done")
        await self.sb.set_event_waited(event_type)
        await self._arbitrator_notification.wait()
        logging.info("waiting released")
        suggestions = await self.sb.get_suggestions(event_type)
        self._arbitrator_notification.clear()
        if not len(suggestions):
            if self._verbose: logging.info("No suggestions for this event | Arbitatrion not required")
            return []

        if self._verbose: logging.info(f"Starting arbitatrion | Number of suggestions received: {len(suggestions)}")
        
        valid_commands: list[SACommand] = []

        for agent, cmd in suggestions:
            has_conflict = False
            to_remove: list[SACommand] = []

            for other in valid_commands:
                if await cmd.conflicts_with(other):
                    if self._verbose: logging.info(f"Conflict detected between -- {await cmd.get_owner()} and {await other.get_owner()} --")
                    if self._verbose: logging.info(f"Action in conflict ({cmd.get_action()}, {other.get_action()})")
                    if cmd.got_higher_priority_than(other.get_prio()):
                        to_remove.append(other)
                    elif cmd.get_prio() == other.get_prio():
                        if await self.ab.tie_break(cmd, other):
                            to_remove.append(other)
                        else:
                            has_conflict = True
                            break
                    else:
                        has_conflict = True
                        break

            if not has_conflict:
                for r in to_remove:
                    await r.discard_command()
                    valid_commands.remove(r)
                valid_commands.append(cmd)

        logging.info("Arbitatrion finished")
        return valid_commands
    
    """                                                     ###############################
                                                            #    SA COMPONENT LOADING     #
                                                            ###############################
    """

    def _to_pascal_case(self, name: str) -> str:
        """Converts a snake_case or compact lowercase name into PascalCase with 'SA' prefix."""
        if name.startswith("sa_"):
            name = name[3:]  # remove 'sa_' prefix
        elif name.startswith("sa"):
            name = name[2:]  # remove 'sa' prefix
        parts = name.split("_") if "_" in name else [name]
        return "SA" + ''.join(part.capitalize() for part in parts)

    async def _loading_sa_components(self):
        """Dynamically loads the SA Components defined in the JSON configuration."""
        self._load_minimal_requirement_config()
        sa_section = self._config["situational_awareness"]["sa_reasoner"]
        components: dict = sa_section["sar_components"]

        for component_name, is_enabled in components.items():
            if is_enabled:
                component_config = sa_section[component_name]
                component_name = component_name.replace("_", "")
                class_name = self._to_pascal_case(component_name) 
                module_path = os.path.join(self.MODULE_PATH, component_name)
                module_file = os.path.join(module_path, f"{component_name}.py")

                if os.path.exists(module_file):
                    module = await self._load_component(class_name, module_file, component_config)
                    if module:
                        logging.info("Addint sa component module to list")
                        self._sa_components[component_name] = module
                else:
                    logging.error(f"⚠️ SA Component {component_name} not found on {module_file}")

        await self._set_minimal_requirements()
        await self._initialize_sa_components()
        
    async def _load_component(self, class_name, component_file, config):
        """Loads a SA Component dynamically and initializes it with its configuration."""
        spec = importlib.util.spec_from_file_location(class_name, component_file)
        if spec and spec.loader:
            component = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(component)
            if hasattr(component, class_name):                     # Verify if class exists
                return getattr(component, class_name)(config)      # Create and instance using component config
            else:
                logging.error(f"⚠️ Cannot create {class_name} SA Component, class not found on {component_file}")
        return None

    async def _initialize_sa_components(self):
        if self._sa_components:
            for sacomp in self._sa_components.values():
                await sacomp.init()

    def _load_minimal_requirement_config(self):
        self._config["situational_awareness"]["sa_reasoner"]["sa_network"]["addr"] = self._addr
        self._config["situational_awareness"]["sa_reasoner"]["sa_network"]["sar"] = self
        self._config["situational_awareness"]["sa_reasoner"]["sa_network"]["strict_topology"] = self._config["situational_awareness"]["strict_topology"]

    async def _set_minimal_requirements(self):
        logging.info(f"sa components found: {len(self._sa_components)}")
        for sac in self._sa_components.keys():
            logging.info(f"Component: {sac}")
        if self._sa_components:
            self._situational_awareness_network = self._sa_components["sanetwork"]         
        else:
            raise ValueError("SA Network not found")



