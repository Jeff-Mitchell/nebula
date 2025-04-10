from abc import abstractmethod, ABC
import asyncio
import logging
from nebula.addons.functions import print_msg_box
from nebula.core.situationalawareness.awareness.suggestionbuffer import SuggestionBuffer
from nebula.core.situationalawareness.awareness.sautils.sacommand import SACommand
from nebula.core.utils.locker import Locker
from nebula.core.nebulaevents import RoundEndEvent
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import RoundEndEvent, AggregationEvent
from nebula.core.network.communications import CommunicationsManager
from nebula.core.situationalawareness.awareness.sautils.sasystemmonitor import SystemMonitor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nebula.core.situationalawareness.nodemanager import NodeManager
    
class SAMComponent(ABC):
    @abstractmethod
    async def init(self):
        raise NotImplementedError
    @abstractmethod
    async def sa_component_actions(self):
        raise NotImplementedError


class SAModule:
    def __init__(
        self,
        nodemanager,
        addr,
        topology,
        verbose = False,
    ):
        print_msg_box(
            msg=f"Starting Situational Awareness module...",
            indent=2,
            title="Situational Awareness module",
        )
        logging.info("🌐  Initializing SAModule")
        self._addr = addr
        self._topology = topology
        self._node_manager: NodeManager = nodemanager
        self._situational_awareness_network = None
        self._situational_awareness_training = None
        self._restructure_process_lock = Locker(name="restructure_process_lock")
        self._restructure_cooldown = 0
        self._arbitrator_notification = asyncio.Event()
        self._suggestion_buffer = SuggestionBuffer(self._arbitrator_notification, verbose=True)
        self._communciation_manager = CommunicationsManager.get_instance()
        self._verbose = verbose
        self._sys_monitor = SystemMonitor()

    @property
    def nm(self):
        return self._node_manager

    @property
    def san(self):
        return self._situational_awareness_network
    
    @property
    def sat(self):
        return self._situational_awareness_training

    @property
    def cm(self):
        return self._communciation_manager
    
    @property
    def sb(self):
        return self._suggestion_buffer
    
    async def init(self):
        from nebula.core.situationalawareness.awareness.sanetwork.sanetwork import SANetwork
        from nebula.core.situationalawareness.awareness.satraining.satraining import SATraining
        self._situational_awareness_network = SANetwork(self, self._addr, self._topology, verbose=True)
        self._situational_awareness_training = SATraining(self, self._addr, "qds", "fastreboot", verbose=True)
        await self.san.init()
        await self.sat.init()
        await EventManager.get_instance().subscribe_node_event(RoundEndEvent, self._process_round_end_event)
        await EventManager.get_instance().subscribe_node_event(AggregationEvent, self._process_aggregation_event)
        
    def is_additional_participant(self):
        return self.nm.is_additional_participant()

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

    async def neighbors_left(self):
        return await self.san.neighbors_left()

    def accept_connection(self, source, joining=False):
        return self.san.accept_connection(source, joining)

    def need_more_neighbors(self):
        return self.san.need_more_neighbors()

    def get_actions(self):
        return self.san.get_actions()

    """                                                     ###############################
                                                            #         ARBITRATION         #
                                                            ###############################
    """
    
    async def _tie_breaker(c1: SACommand, c2: SACommand):
        return True
    
    async def _process_round_end_event(self, ree : RoundEndEvent):
        logging.info("🔄 Arbitration | Round End Event...")
        asyncio.create_task(self.san.sa_component_actions())
        asyncio.create_task(self.sat.sa_component_actions())
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
                        if await self._tie_breaker(cmd, other):
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

