import asyncio
import logging
import subprocess
from nebula.addons.networksimulation.networksimulator import NetworkSimulator, factory_network_preset


# Cellular network generations NetworkSimulator
class CNGNetworkSimulator(NetworkSimulator):
    IP_MULTICAST = "239.255.255.250"

    def __init__(self, config: dict):
        self._node_interface = config["interface"]
        self._verbose = True#config["verbose"]
        self._preset = config["preset"]
        self._network_conditions = factory_network_preset(self._preset)
        self._running = asyncio.Event()
        federation_nodes = config["federation"].split()
        self._federation_nodes = set(federation_nodes)

    async def start(self):
        logging.info("🌐  Cellular network generations Network Simulator starting")
        logging.info(f" Setup selected: {self._preset}")
        self._running.set()
        await self.set_network_conditions()

    async def stop(self):
        logging.info("🌐  Nebula Network Simulator stopping...")
        self._running.clear()

    async def is_running(self):
        return self._running.is_set()

    async def set_thresholds(self, thresholds: dict):
        pass
    
    async def set_network_conditions(self):
        for node in self._federation_nodes:
            addr_ip = node.split(":")[0]
            await self._apply_conditions(addr_ip)

    async def _apply_conditions(self, addr_ip: str):
        self._set_network_condition_for_addr(
            interface=self._node_interface,
            network=addr_ip,
            bandwidth=self._network_conditions["bandwidth"],
            delay=self._network_conditions["delay"],
            delay_distro=self._network_conditions["delay_distro"],
            delay_distribution=self._network_conditions["delay_distribution"],
            loss=self._network_conditions["loss"],
            duplicate=self._network_conditions["duplicate"],
            corrupt=self._network_conditions["corrupt"],
            reordering=self._network_conditions["reordering"],
        )
        self._set_network_condition_for_multicast(
            interface=self._node_interface,
            src_network=addr_ip,
            dst_network=self.IP_MULTICAST,
            bandwidth=self._network_conditions["bandwidth"],
            delay=self._network_conditions["delay"],
            delay_distro=self._network_conditions["delay_distro"],
            delay_distribution=self._network_conditions["delay_distribution"],
            loss=self._network_conditions["loss"],
            duplicate=self._network_conditions["duplicate"],
            corrupt=self._network_conditions["corrupt"],
            reordering=self._network_conditions["reordering"],
        )

    def _set_network_condition_for_addr(
        self,
        interface="eth0",
        network="192.168.50.2",
        bandwidth="5Gbps",
        delay="0ms",
        delay_distro="10ms",
        delay_distribution="normal",
        loss="0%",
        duplicate="0%",
        corrupt="0%",
        reordering="0%",
    ):
        if self._verbose:
            logging.info(
                f"🌐  Changing network conditions | Interface: {interface} | Network: {network} | Bandwidth: {bandwidth} | Delay: {delay} | Delay Distro: {delay_distro} | Delay Distribution: {delay_distribution} | Loss: {loss} | Duplicate: {duplicate} | Corrupt: {corrupt} | Reordering: {reordering}"
            )
        try:
            results = subprocess.run(
                [
                    "tcset",
                    str(interface),
                    "--network",
                    str(network) if network is not None else "",
                    "--rate",
                    str(bandwidth),
                    "--delay",
                    str(delay),
                    "--delay-distro",
                    str(delay_distro),
                    "--delay-distribution",
                    str(delay_distribution),
                    "--loss",
                    str(loss),
                    "--duplicate",
                    str(duplicate),
                    "--corrupt",
                    str(corrupt),
                    "--reordering",
                    str(reordering),
                    "--change",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            logging.exception(f"❗️  Network simulation error: {e}")
            return

    def _set_network_condition_for_multicast(
        self,
        interface="eth0",
        src_network="",
        dst_network="",
        bandwidth="5Gbps",
        delay="0ms",
        delay_distro="10ms",
        delay_distribution="normal",
        loss="0%",
        duplicate="0%",
        corrupt="0%",
        reordering="0%",
    ):
        if self._verbose:
            logging.info(
                f"🌐  Changing multicast conditions | Interface: {interface} | Src Network: {src_network} | Bandwidth: {bandwidth} | Delay: {delay} | Delay Distro: {delay_distro} | Delay Distribution: {delay_distribution} | Loss: {loss} | Duplicate: {duplicate} | Corrupt: {corrupt} | Reordering: {reordering}"
            )

        try:
            results = subprocess.run(
                [
                    "tcset",
                    str(interface),
                    "--src-network",
                    str(src_network),
                    "--dst-network",
                    str(dst_network),
                    "--rate",
                    str(bandwidth),
                    "--delay",
                    str(delay),
                    "--delay-distro",
                    str(delay_distro),
                    "--delay-distribution",
                    str(delay_distribution),
                    "--loss",
                    str(loss),
                    "--duplicate",
                    str(duplicate),
                    "--corrupt",
                    str(corrupt),
                    "--reordering",
                    str(reordering),
                    "--direction",
                    "incoming",
                    "--change",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            logging.exception(f"❗️  Network simulation error: {e}")
            return

    def clear_network_conditions(self, interface):
        if self._verbose:
            logging.info("🌐  Resetting network conditions")
        try:
            results = subprocess.run(
                ["tcdel", str(interface), "--all"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            logging.exception(f"❗️  Network simulation error: {e}")
            return
