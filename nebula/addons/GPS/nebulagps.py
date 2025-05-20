import asyncio
import logging
from nebula.addons.GPS.gpsmodule import GPSModule
from nebula.core.nebulaevents import GPSEvent
import socket
from nebula.core.utils.locker import Locker
from geopy import distance
from nebula.core.eventmanager import EventManager

class NebulaGPS(GPSModule):
    BROADCAST_IP = "255.255.255.255"    # Broadcast IP
    BROADCAST_PORT = 50001              # Poort used for GPS
    INTERFACE = "eth2"                  # Interface to avoid network conditions

    def __init__(self, config, addr, update_interval: float = 5.0, verbose=False):
        self._config = config
        self._addr = addr
        self.update_interval = update_interval  # Frecuencia de emisión
        self.running = False
        self._node_locations = {}  # Diccionario para almacenar ubicaciones de nodos
        self._broadcast_socket = None
        self._nodes_location_lock = Locker("nodes_location_lock", async_lock=True)
        self._verbose = verbose
   
    async def start(self):
        """Inicia el servicio de GPS, enviando y recibiendo ubicaciones."""
        logging.info("Starting NebulaGPS service...")
        self.running = True

        # Crear socket de broadcast
        self._broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # Enlazar socket en eth2 para recibir también datos
        self._broadcast_socket.bind(("", self.BROADCAST_PORT))

        # Iniciar tareas de envío y recepción
        asyncio.create_task(self._send_location_loop())
        asyncio.create_task(self._receive_location_loop())
        asyncio.create_task(self._notify_geolocs())

    async def stop(self):
        """Detiene el servicio de GPS."""
        logging.info("Stopping NebulaGPS service...")
        self.running = False
        if self._broadcast_socket:
            self._broadcast_socket.close()
            self._broadcast_socket = None
            
    async def is_running(self):
        return self.running        
    
    async def get_geoloc(self):
        latitude = self._config.participant["mobility_args"]["latitude"]
        longitude = self._config.participant["mobility_args"]["longitude"]
        return (latitude, longitude)
    
    async def calculate_distance(self, self_lat, self_long, other_lat, other_long):
        distance_m = distance.distance((self_lat, self_long), (other_lat, other_long)).m
        return distance_m

    async def _send_location_loop(self):
        """Envia la geolocalización periódicamente por broadcast."""
        while self.running:
            latitude, longitude = await self.get_geoloc()  # Obtener ubicación actual
            message = f"GPS-UPDATE {self._addr} {latitude} {longitude}"
            self._broadcast_socket.sendto(message.encode(), (self.BROADCAST_IP, self.BROADCAST_PORT))
            if self._verbose: logging.info(f"Sent GPS location: ({latitude}, {longitude})")
            await asyncio.sleep(self.update_interval)

    async def _receive_location_loop(self):
        """Escucha y almacena geolocalizaciones de otros nodos."""
        while self.running:
            try:
                data, addr = await asyncio.get_running_loop().run_in_executor(
                    None, self._broadcast_socket.recvfrom, 1024
                )
                message = data.decode().strip()
                if message.startswith("GPS-UPDATE"):
                    _, sender_addr, lat, lon = message.split()
                    if sender_addr != self._addr:
                        async with self._nodes_location_lock:
                            self._node_locations[sender_addr] = (float(lat), float(lon))
                    if self._verbose: logging.info(f"Received GPS from {addr[0]}: {lat}, {lon}")
            except Exception as e:
                logging.error(f"Error receiving GPS update: {e}")

    async def _notify_geolocs(self):
        while True:
            await asyncio.sleep(self.update_interval)
            await self._nodes_location_lock.acquire_async()
            geolocs : dict = self._node_locations.copy()
            await self._nodes_location_lock.release_async()
            if geolocs:
                distances = {}
                self_lat, self_long = await self.get_geoloc()
                for addr, (lat, long) in geolocs.items():
                    dist = await self.calculate_distance(self_lat, self_long, lat, long)
                    distances[addr] = (dist,(lat, long))
                gpsevent = GPSEvent(distances)
                asyncio.create_task(EventManager.get_instance().publish_addonevent(gpsevent))
        