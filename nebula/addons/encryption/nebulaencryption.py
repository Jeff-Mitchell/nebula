from nebula.addons.encryption.iencryption import IEncryption
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from collections import defaultdict
from nebula.core.nebulaevents import UpdateNeighborEvent
from nebula.core.utils.locker import Locker
from enum import Enum
from nebula.core.eventmanager import EventManager
from nebula.core.network.communications import CommunicationsManager
import asyncio
import logging

class NebulaEncryptionType(Enum):
    NOT_SECURE = "Not secure"
    UNIDIRECTIONAL = "Unidrectional"
    BIDIRECTIONAL = "Bidirectional"

class NebulaEncryption(IEncryption):

    def __init__(self, changing_interval):
        self._changing_interval = changing_interval
        self._private_key = None
        self._public_key = None
        self._encrypted_connections: dict[str, tuple[NebulaEncryptionType, HKDF]] = defaultdict(tuple(NebulaEncryptionType, HKDF))
        self._ec_lock = Locker("encrypted_connections_lock", async_lock=True)
        self._cm = CommunicationsManager.get_instance()

    @property
    def ec(self):
        """Encrypted connections dictionary"""
        return self._encrypted_connections
    
    @property
    def cm(self):
        """Communication Manager"""
        return self._cm

    async def init(self):
        await self._generate_keys()
        current_connections = await self.cm.get_addrs_current_connections(myself=False)
        async with self._ec_lock:
            for cn in current_connections:
                self.ec[cn] = (NebulaEncryptionType.NOT_SECURE, None)
        #TODO
        # 2- Subscribe to messages events
        
    #TODO
    # 1- create and send keys
    # 2- AFD to manage the process
        
    async def _generate_keys(self):
        self._private_key = private_key = ec.generate_private_key(ec.SECP256R1())
        self._public_key = public_key = private_key.public_key()

        # Seralized public key to share
        # public_bytes = public_key.public_bytes(
        #     encoding=serialization.Encoding.PEM,
        #     format=serialization.PublicFormat.SubjectPublicKeyInfo
        # )

    async def _generate_shared_key(self, node, node_public_Key):
        # Public key from node
        public_key_B = serialization.load_pem_public_key(node_public_Key)

        # Derive shared key
        shared_key_A = self._private_key.exchange(ec.ECDH(), public_key_B)

        # Use HKDF algorithm to derive final secure key
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'dfl-key-agreement'
        ).derive(shared_key_A)
        
        await self._update_encrypted_connection(node, derived_key)

    async def _update_encrypted_connection(self, node, shared_key):
        async with self._ec_lock:
            self.ec[node] = (NebulaEncryptionType.BIDIRECTIONAL, shared_key)
