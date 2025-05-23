from abc import ABC, abstractmethod


class IEncryption(ABC):
    @abstractmethod
    async def init(self):
        pass

class EncryptionImplementation(Exception):
    pass

def factory_encryption_protocol(crypto_protocol, changing_interval) -> IEncryption:
    from nebula.addons.encryption.nebulaencryption import NebulaEncryption

    encryption_protocols = {
        "nebula": NebulaEncryption,
    }

    cryp_prot = encryption_protocols.get(crypto_protocol)

    if cryp_prot:
        return cryp_prot(changing_interval)
    else:
        raise EncryptionImplementation(f"Encryption protocol {crypto_protocol} not found")