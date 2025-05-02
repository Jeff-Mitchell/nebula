from nebula.core.utils.locker import Locker
from collections import deque, OrderedDict
import logging
from collections import defaultdict
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateNeighborEvent
from nebula.core.situationalawareness.awareness.sareputation.sareputation import ThreatCategory

class CollaborativeReputation():
    MAX_TRIALS_ACCEPTED = 3
    
    def __init__(self):
        self._trials_recently_received: int = 0
        
    async def init(self):
        await EventManager.get_instance().subscribe(("reputation", "share_reputation"), self._process_share_reputation_message)
        await EventManager.get_instance().subscribe(("reputation", "submit_verdict"), self._process_trial_verdict_message)
        
    async def _process_share_reputation_message(self, source, message):
        pass    
    
    async def _process_trial_verdict_message(self, source, message):
        pass