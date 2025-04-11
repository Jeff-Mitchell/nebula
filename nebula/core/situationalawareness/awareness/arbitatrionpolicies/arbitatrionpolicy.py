from abc import ABC, abstractmethod
from nebula.core.situationalawareness.awareness.sautils.sacommand import SACommand

class ArbitatrionPolicy(ABC):
    
    @abstractmethod
    async def init(self, config):
        raise NotImplementedError
    
    @abstractmethod
    async def tie_break(self, sac1: SACommand, sac2: SACommand) -> bool:
        raise NotImplementedError

def factory_arbitatrion_policy(arbitatrion_policy, verbose) -> ArbitatrionPolicy:
    from nebula.core.situationalawareness.awareness.arbitatrionpolicies.staticarbitatrionpolicy import SAP
    from nebula.core.situationalawareness.awareness.arbitatrionpolicies.saarbitatrionpolicy import SAAP
    
    options = {
        "sap": SAP,     # "Static Arbitatrion Policy"                   (SAP) -- default value
        "saap": SAAP,   # "Situational Awareness Arbitatrion Policy"    (SAAP) 
    } 
    
    cs = options.get(arbitatrion_policy, SAP)
    return cs(verbose)