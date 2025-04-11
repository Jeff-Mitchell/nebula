import asyncio
from nebula.core.situationalawareness.awareness.arbitatrionpolicies.arbitatrionpolicy import ArbitatrionPolicy
from nebula.core.situationalawareness.awareness.sautils.sacommand import SACommand

class SAAP(ArbitatrionPolicy):
    def __init__(self, verbose):
        pass

    async def init(self, config):
        pass

    async def tie_break(self, sac1: SACommand, sac2: SACommand) -> SACommand:
        """
        Tie break conflcited SA Commands
        """
        pass