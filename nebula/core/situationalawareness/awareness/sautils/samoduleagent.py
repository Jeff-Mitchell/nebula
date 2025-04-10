from abc import abstractmethod, ABC
from nebula.core.situationalawareness.awareness.sautils.sacommand import SACommand

class SAModuleAgent(ABC):

    @abstractmethod
    async def get_agent(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def register_sa_agent(self):
        raise NotImplementedError
    
    @abstractmethod
    async def suggest_action(self, sac : SACommand):
        raise NotImplementedError
    
    @abstractmethod
    async def notify_all_suggestions_done(self, event_type):
        raise NotImplementedError