from nebula.core.utils.locker import Locker
from nebula.utils import logging
import asyncio
from nebula.core.situationalawareness.awareness.sautils.samoduleagent import SAModuleAgent
from nebula.core.situationalawareness.awareness.sautils.sacommand import SACommand
from nebula.core.nebulaevents import NodeEvent, RoundEndEvent, AggregationEvent
from collections import defaultdict
from typing import Type

class SuggestionBuffer():
    _instance = None
    _lock = Locker("initialize_sb_lock", async_lock=False)

    def __new__(cls, arbitrator_notification, verbose):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Obtain SuggestionBuffer instance"""
        if cls._instance is None:
            raise ValueError("SuggestionBuffer has not been initialized yet.")
        return cls._instance

    def __init__(self, arbitrator_notification : asyncio.Event, verbose):
        """Initializes the suggestion buffer with thread-safe synchronization."""
        self._arbitrator_notification = arbitrator_notification
        self._arbitrator_notification_lock = Locker("arbitrator_notification_lock", async_lock=True)
        self._verbose = verbose
        self._buffer : dict[Type[NodeEvent], list[tuple[SAModuleAgent, SACommand]]] = defaultdict(list)               
        self._suggestion_buffer_lock = Locker("suggestion_buffer_lock", async_lock=True)
        self._expected_agents: dict[Type[NodeEvent] ,list[SAModuleAgent]] = defaultdict(list)                                               
        self._expected_agents_lock = Locker("expected_agents_lock", async_lock=True)
        self._event_notifications : dict[Type[NodeEvent], list[tuple[SAModuleAgent, asyncio.Event]]] = defaultdict(list)
        self._event_waited = None
                      
    async def register_event_agents(self, event_type, agent: SAModuleAgent):
        """Registers expected agents for a given event."""
        async with self._expected_agents_lock:
            if self._verbose:
                logging.info(f"Registering SA Agent: {await agent.get_agent()} for event: {event_type. __name__}")
                
            if event_type not in self._event_notifications:
                self._event_notifications[event_type] = []
                
            self._expected_agents[event_type].append(agent)
                 
            existing_agents = {a for a, _ in self._event_notifications[event_type]}
            if agent not in existing_agents:
                self._event_notifications[event_type].append((agent, asyncio.Event()))

    async def register_suggestion(self, event_type, agent: SAModuleAgent, suggestion: SACommand):
        """Registers a suggestion from an agent for a specific event."""
        async with self._suggestion_buffer_lock:
            if self._verbose: logging.info(f"Registering Suggestion from SA Agent: {await agent.get_agent()} for event: {event_type. __name__}")
            self._buffer[event_type].append((agent, suggestion))

    async def set_event_waited(self, event_type):
        """Registers event to be waited"""
        if not self._event_waited:
            if self._verbose: logging.info(f"Set notification when all suggestions have being received for event: {event_type. __name__}")
            self._event_waited = event_type
            await self._notify_arbitrator(event_type)

    async def notify_all_suggestions_done_for_agent(self, saa : SAModuleAgent, event_type):
        """SA Agent notification that has registered all the suggestions for event_type."""
        async with self._expected_agents_lock:
            agent_found = False
            for agent, event in self._event_notifications.get(event_type, []):
                if agent == saa:
                    event.set()
                    agent_found = True
                    if self._verbose:
                        logging.info(f"SA Agent: {await saa.get_agent()} notifies all suggestions registered for event: {event_type. __name__}")
                    break
            if not agent_found and self._verbose:
                logging.error(f"SAModuleAgent: {await saa.get_agent()} not found on notifications awaited for event {event_type. __name__}")
        await self._notify_arbitrator(event_type)

    async def _notify_arbitrator(self, event_type):
        """Checks whether to notify the arbitrator that all suggestions for event_type are received."""
        if event_type != self._event_waited:
            return

        async with self._arbitrator_notification_lock:
            async with self._expected_agents_lock:
                expected_agents = self._expected_agents.get(event_type, [])
                notifications = self._event_notifications.get(event_type, list())

                agent_event_map = {a: e for a, e in notifications}
                all_received = all(
                    agent in agent_event_map and agent_event_map[agent].is_set()
                    for agent in expected_agents
                )

                if all_received:
                    self._arbitrator_notification.set()
                    self._event_waited = None
                    await self._reset_notifications_for_agents(event_type, expected_agents)                

    async def _reset_notifications_for_agents(self, event_type, agents):
        """Reset notifications for SA Agents for the given event."""
        notifications = self._event_notifications.get(event_type, set())
        for agent, event in notifications:
            if agent in agents:
                event.clear()

    async def get_suggestions(self, event_type) -> list[tuple[SAModuleAgent, SACommand]]:
        """Retrieves all suggestions registered for a given event."""
        async with self._suggestion_buffer_lock:
            async with  self._expected_agents_lock:
                suggestions = list(self._buffer.get(event_type, []))
                if self._verbose: logging.info(f"Retrieving all sugestions for event: {event_type. __name__}")
                await self._clear_suggestions(event_type)
                return suggestions

    async def _clear_suggestions(self, event_type):
        """Clears all suggestions and metadata stored for a given event."""
        self._buffer[event_type].clear()
    
