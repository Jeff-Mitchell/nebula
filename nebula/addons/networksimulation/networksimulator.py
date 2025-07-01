from abc import ABC, abstractmethod

_NETWORK_PRESETS = {
    "3G": {
        "bandwidth": "1mbit",
        "delay": "120ms",
        "delay_distro": "30ms",
        "delay_distribution": "normal",
        "loss": 0.5,
        "duplicate": 0.2,
        "corrupt": 0.01,
        "reordering": 0.1,
    },
    "4G": {
        "bandwidth": "10mbit",
        "delay": "60ms",
        "delay_distro": "10ms",
        "delay_distribution": "normal",
        "loss": 0.1,
        "duplicate": 0.1,
        "corrupt": 0.005,
        "reordering": 0.05,
    },
    "5G": {
        "bandwidth": "100mbit",
        "delay": "20ms",
        "delay_distro": "5ms",
        "delay_distribution": "normal",
        "loss": 0.05,
        "duplicate": 0.0,
        "corrupt": 0.001,
        "reordering": 0.01,
    },
}

class NetworkSimulator(ABC):
    """
    Abstract base class representing a network simulator interface.

    This interface defines the required methods for controlling and simulating network conditions between nodes.
    A concrete implementation is expected to manage artificial delays, bandwidth restrictions, packet loss, 
    or other configurable conditions typically used in network emulation or testing.

    Required asynchronous methods:
    - `start()`: Initializes the network simulation module.
    - `stop()`: Shuts down the simulation and cleans up any active conditions.
    - `set_thresholds(thresholds)`: Configures system-wide thresholds (e.g., max/min delay or distance mappings).
    - `set_network_conditions(dest_addr, distance)`: Applies network constraints to a target address based on distance.

    Synchronous method:
    - `clear_network_conditions(interface)`: Clears any simulated network configuration for a given interface.

    All asynchronous methods should be non-blocking to support integration in async systems.
    """

    @abstractmethod
    async def start(self):
        """
        Starts the network simulation module.

        This might involve preparing network interfaces, initializing tools like `tc`, or configuring internal state.
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Stops the network simulation module.

        Cleans up any modifications made to network interfaces or system configuration.
        """
        pass

    @abstractmethod
    async def set_thresholds(self, thresholds: dict):
        """
        Sets threshold values for simulating conditions.

        Args:
            thresholds (dict): A dictionary specifying condition thresholds,
                               e.g., {'low': 100, 'medium': 200, 'high': 300}, or distance-delay mappings.
        """
        pass

    @abstractmethod
    async def set_network_conditions(self, dest_addr, distance):
        """
        Applies network simulation settings to a given destination based on the computed distance.

        Args:
            dest_addr (str): The address of the destination node (e.g., IP or identifier).
            distance (float): The physical or logical distance used to determine the simulation severity.
        """
        pass

    @abstractmethod
    def clear_network_conditions(self, interface):
        """
        Clears any simulated network conditions applied to the specified network interface.

        Args:
            interface (str): The name of the network interface to restore (e.g., 'eth0').
        """
        pass


class NetworkSimulatorException(Exception):
    pass

class NetworkPresetException(Exception):
    pass


def factory_network_simulator(net_sim, changing_interval, interface, verbose) -> NetworkSimulator:
    from nebula.addons.networksimulation.nebulanetworksimulator import NebulaNS
    from nebula.addons.networksimulation.cngnetworksimulator import CNGNetworkSimulator

    SIMULATION_SERVICES = {
        "nebula": NebulaNS,
        "Cellular network generation": CNGNetworkSimulator,
    }

    net_serv = SIMULATION_SERVICES.get(net_sim, NebulaNS)

    if net_serv:
        return net_serv(changing_interval, interface, verbose)
    else:
        raise NetworkSimulatorException(f"Network Simulator {net_sim} not found")
    
def factory_network_preset(preset: str) -> dict:
    net_preset = _NETWORK_PRESETS.get(preset, None)
    if net_preset:
        return net_preset
    else:
        raise NetworkPresetException(f"Network preset {preset} not found")
