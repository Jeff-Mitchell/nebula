import asyncio
import logging
from typing import TYPE_CHECKING

from nebula.addons.functions import print_msg_box
from nebula.core.situationalawareness.candidateselection.candidateselector import factory_CandidateSelector
from nebula.core.situationalawareness.modelhandlers.modelhandler import factory_ModelHandler
from nebula.core.situationalawareness.awareness.samodule import SAModule
from nebula.core.utils.locker import Locker
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateNeighborEvent, NodeFoundEvent
from nebula.core.network.communications import CommunicationsManager
from functools import cached_property

if TYPE_CHECKING:
    from nebula.core.engine import Engine

RESTRUCTURE_COOLDOWN = 5


class NodeManager:
    OFFER_TIMEOUT = 5

    def __init__(
        self,
        aditional_participant,
        topology,
        model_handler,
        engine: "Engine",
        verbose=False
    ):
        self._aditional_participant = aditional_participant
        self.topology = topology
        print_msg_box(
            msg=f"Starting NodeManager module...", indent=2, title="NodeManager module"
        )
        logging.info("🌐  Initializing Node Manager")
        self._engine = engine
        self._cm = None
        self.config = engine.get_config()
        logging.info("Initializing Candidate Selector")
        self._candidate_selector = factory_CandidateSelector(self.topology)
        logging.info("Initializing Model Handler")
        self._model_handler = factory_ModelHandler(model_handler)
        self._update_neighbors_lock = Locker(name="_update_neighbors_lock", async_lock=True)
        self.late_connection_process_lock = Locker(name="late_connection_process_lock")
        self.pending_confirmation_from_nodes = set()
        self.pending_confirmation_from_nodes_lock = Locker(name="pending_confirmation_from_nodes_lock", async_lock=True)
        self.accept_candidates_lock = Locker(name="accept_candidates_lock")
        self.recieve_offer_timer = self.OFFER_TIMEOUT
        self.discarded_offers_addr_lock = Locker(name="discarded_offers_addr_lock")
        self.discarded_offers_addr = []
        
        self._situational_awareness_module = SAModule(self, self.config, self.engine.addr, topology, True)
        self._verbose = verbose

    @property
    def engine(self):
        return self._engine
    
    @cached_property
    def cm(self):
        return CommunicationsManager.get_instance()

    @property
    def candidate_selector(self):
        return self._candidate_selector

    @property
    def model_handler(self):
        return self._model_handler

    @property
    def sam(self):
        """Situational Awareness Module"""
        return self._situational_awareness_module

    def is_additional_participant(self):
        return self._aditional_participant

    async def set_configs(self):
        """
        model_handler config:
            - self total rounds
            - self current round
            - self epochs

        candidate_selector config:
            - self model loss
            - self weight distance
            - self weight hetereogeneity
        """
        logging.info("Building NodeManager configurations...")
        await self.register_message_events_callbacks()
        await self.sam.init()
        await EventManager.get_instance().subscribe_node_event(UpdateNeighborEvent, self.update_neighbors)
        logging.info("Building candidate selector configuration..")
        self.candidate_selector.set_config([0, 0.5, 0.5])
        # self.engine.trainer.get_loss(), self.config.participant["molibity_args"]["weight_distance"], self.config.participant["molibity_args"]["weight_het"]
             
    """
                ##############################
                #        CONNECTIONS         #
                ##############################
    """

    def get_restructure_process_lock(self):
        return self.sam.get_restructure_process_lock()

    def accept_connection(self, source, joining=False):
        return self.sam.accept_connection(source, joining)
    
    def still_waiting_for_candidates(self):
        return not self.accept_candidates_lock.locked() and self.late_connection_process_lock.locked()

    async def add_pending_connection_confirmation(self, addr):
        await self._update_neighbors_lock.acquire_async()
        await self.pending_confirmation_from_nodes_lock.acquire_async()
        if addr not in self.sam.get_nodes_known(neighbors_only=True):
            logging.info(f" Addition | pending connection confirmation from: {addr}")
            self.pending_confirmation_from_nodes.add(addr)
        await self.pending_confirmation_from_nodes_lock.release_async()
        await self._update_neighbors_lock.release_async()

    async def _remove_pending_confirmation_from(self, addr):
        await self.pending_confirmation_from_nodes_lock.acquire_async()
        self.pending_confirmation_from_nodes.discard(addr)
        await self.pending_confirmation_from_nodes_lock.release_async()

    async def clear_pending_confirmations(self):
        await self.pending_confirmation_from_nodes_lock.acquire_async()
        self.pending_confirmation_from_nodes.clear()
        await self.pending_confirmation_from_nodes_lock.release_async()

    async def waiting_confirmation_from(self, addr):
        await self.pending_confirmation_from_nodes_lock.acquire_async()
        found = addr in self.pending_confirmation_from_nodes
        await self.pending_confirmation_from_nodes_lock.release_async()
        return found

    async def confirmation_received(self, addr, joining=False):
        logging.info(f" Update | connection confirmation received from: {addr} | joining federation: {joining}")
        await self.cm.connect(addr, direct=True)
        await self._remove_pending_confirmation_from(addr)
        une = UpdateNeighborEvent(addr, joining=joining)
        await EventManager.get_instance().publish_node_event(une)
            
    def add_to_discarded_offers(self, addr_discarded):
        self.discarded_offers_addr_lock.acquire()
        self.discarded_offers_addr.append(addr_discarded)
        self.discarded_offers_addr_lock.release()

    def get_actions(self):
        return self.sam.get_actions()

    async def register_late_neighbor(self, addr, joinning_federation=False):
        if self._verbose: logging.info(f"Registering | late neighbor: {addr}, joining: {joinning_federation}")
        une = UpdateNeighborEvent(addr, joining=joinning_federation)
        await EventManager.get_instance().publish_node_event(une)

    async def update_neighbors(self, une : UpdateNeighborEvent):
        node, remove = await une.get_event_data()
        await self._update_neighbors_lock.acquire_async()
        if not remove:
            await self.meet_node(node)
        await self._remove_pending_confirmation_from(node)
        await self._update_neighbors_lock.release_async()

    async def meet_node(self, node):
        nfe = NodeFoundEvent(node)
        await EventManager.get_instance().publish_node_event(nfe)

    def get_nodes_known(self, neighbors_too=False):
        return self.sam.get_nodes_known(neighbors_too)

    def accept_model_offer(self, source, decoded_model, rounds, round, epochs, n_neighbors, loss):
        if not self.accept_candidates_lock.locked():
            if self._verbose: logging.info(f"🔄 Processing offer from {source}...")
            model_accepted = self.model_handler.accept_model(decoded_model)
            self.model_handler.set_config(config=(rounds, round, epochs, self))
            if model_accepted:
                self.candidate_selector.add_candidate((source, n_neighbors, loss))
                return True
        else:
            return False

    async def get_trainning_info(self):
        return await self.model_handler.get_model(None)

    def add_candidate(self, source, n_neighbors, loss):
        if not self.accept_candidates_lock.locked():
            self.candidate_selector.add_candidate((source, n_neighbors, loss))

    async def stop_not_selected_connections(self):
        try:
            with self.discarded_offers_addr_lock:
                if len(self.discarded_offers_addr) > 0:
                    self.discarded_offers_addr = set(
                        self.discarded_offers_addr
                    ) - await self.cm.get_addrs_current_connections(only_direct=True, myself=False)
                    if self._verbose: logging.info(
                        f"Interrupting connections | discarded offers | nodes discarded: {self.discarded_offers_addr}"
                    )
                    for addr in self.discarded_offers_addr:
                        await self.cm.disconnect(addr, mutual_disconnection=True)
                        await asyncio.sleep(1)
                    self.discarded_offers_addr = []
        except asyncio.CancelledError:
            pass

    async def start_late_connection_process(self, connected=False, msg_type="discover_join", addrs_known=None):
        """
        This function represents the process of discovering the federation and stablish the first
        connections with it. The first step is to send the DISCOVER_JOIN/NODES message to look for nodes,
        the ones that receive that message will send back a OFFER_MODEL/METRIC message. It contains info to do
        a selection process among candidates to later on connect to the best ones.
        The process will repeat until at least one candidate is found and the process will be locked
        to avoid concurrency.
        """
        logging.info("🌐  Initializing late connection process..")

        self.late_connection_process_lock.acquire()
        best_candidates = []
        self.candidate_selector.remove_candidates()
        await self.clear_pending_confirmations()

        # find federation and send discover
        connections_stablished = await self.cm.stablish_connection_to_federation(msg_type, addrs_known)

        # wait offer
        if self._verbose: logging.info(f"Connections stablish after finding federation: {connections_stablished}")
        if connections_stablished:
            if self._verbose: logging.info(f"Waiting: {self.recieve_offer_timer}s to receive offers from federation")
            await asyncio.sleep(self.recieve_offer_timer)

        # acquire lock to not accept late candidates
        self.accept_candidates_lock.acquire()

        if self.candidate_selector.any_candidate():
            if self._verbose: logging.info("Candidates found to connect to...")
            # create message to send to candidates selected
            if not connected:
                msg = self.cm.create_message("connection", "late_connect")
            else:
                msg = self.cm.create_message("connection", "restructure")

            best_candidates = self.candidate_selector.select_candidates()
            if self._verbose: logging.info(f"Candidates | {[addr for addr, _, _ in best_candidates]}")
            try:
                for addr, _, _ in best_candidates:
                    await self.add_pending_connection_confirmation(addr)
                    await self.cm.send_message(addr, msg)
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                upe = UpdateNeighborEvent(addr, removed=True)
                asyncio.create_task(EventManager.get_instance().publish_node_event(upe))
                if self._verbose: logging.info("Error during stablishment")
            self.accept_candidates_lock.release()
            self.late_connection_process_lock.release()
            self.candidate_selector.remove_candidates()
        # if no candidates, repeat process
        else:
            if self._verbose: logging.info("❗️  No Candidates found...")
            self.accept_candidates_lock.release()
            self.late_connection_process_lock.release()
            if not connected:
                if self._verbose: logging.info("❗️  repeating process...")
                await self.start_late_connection_process(connected, msg_type, addrs_known)


    """                                                     ##############################
                                                            #     Mobility callbacks     #
                                                            ##############################
    """

    async def register_message_events_callbacks(self):
        me_dict = self.cm.get_messages_events()
        message_events = [
            (message_name, message_action)
            for (message_name, message_actions) in me_dict.items()
            for message_action in message_actions
        ]
        for event_type, action in message_events:
            callback_name = f"_{event_type}_{action}_callback"
            method = getattr(self, callback_name, None)

            if callable(method):
                await EventManager.get_instance().subscribe((event_type, action), method)

    async def _connection_late_connect_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received late connect message from {source}")
        # Verify if it's a confirmation message from a previous late connection message sent to source
        if await self.waiting_confirmation_from(source):
            await self.confirmation_received(source, joining=True)
            return

        if not self.engine.get_initialization_status():
            logging.info("❗️ Connection refused | Device not initialized yet...")
            return

        if self.accept_connection(source, joining=True):
            logging.info(f"🔗  handle_connection_message | Late connection accepted | source: {source}")
            await self.cm.connect(source, direct=True)

            # Verify conenction is accepted
            conf_msg = self.cm.create_message("connection", "late_connect")
            await self.cm.send_message(source, conf_msg)
            await self.register_late_neighbor(source, joinning_federation=True)

            ct_actions, df_actions = self.get_actions()
            if len(ct_actions):
                cnt_msg = self.cm.create_message("link", "connect_to", addrs=ct_actions)
                await self.cm.send_message(source, cnt_msg)

            if len(df_actions):
                df_msg = self.cm.create_message("link", "disconnect_from", addrs=df_actions)
                await self.cm.send_message(source, df_msg)

        else:
            logging.info(f"❗️  Late connection NOT accepted | source: {source}")

    async def _connection_restructure_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received restructure message from {source}")
        # Verify if it's a confirmation message from a previous restructure connection message sent to source
        if await self.waiting_confirmation_from(source):
            await self.confirmation_received(source)
            return

        if not self.engine.get_initialization_status():
            logging.info("❗️ Connection refused | Device not initialized yet...")
            return

        if self.accept_connection(source, joining=False):
            logging.info(f"🔗  handle_connection_message | Trigger | restructure connection accepted from {source}")
            await self.cm.connect(source, direct=True)

            conf_msg = self.cm.create_message("connection", "restructure")

            await self.cm.send_message(source, conf_msg)

            ct_actions, df_actions = self.get_actions()
            if len(ct_actions):
                cnt_msg = self.cm.create_message("link", "connect_to", addrs=ct_actions)
                await self.cm.send_message(source, cnt_msg)

            if len(df_actions):
                df_msg = self.cm.create_message("link", "disconnect_from", addrs=df_actions)
                await self.cm.send_message(source, df_msg)

            await self.register_late_neighbor(source, joinning_federation=False)
        else:
            logging.info(f"❗️  handle_connection_message | Trigger | restructure connection denied from {source}")

    async def _discover_discover_join_callback(self, source, message):
        logging.info(f"🔍  handle_discover_message | Trigger | Received discover_join message from {source} ")
        if len(self.engine.get_federation_nodes()) > 0:
            await self.engine.trainning_in_progress_lock.acquire_async()
            model, rounds, round = (
                await self.cm.propagator.get_model_information(source, "stable")
                if self.engine.get_round() > 0
                else await self.cm.propagator.get_model_information(source, "initialization")
            )
            await self.engine.trainning_in_progress_lock.release_async()
            if round != -1:
                epochs = self.config.participant["training_args"]["epochs"]
                msg = self.cm.create_message(
                    "offer",
                    "offer_model",
                    len(self.engine.get_federation_nodes()),
                    0,
                    parameters=model,
                    rounds=rounds,
                    round=round,
                    epochs=epochs,
                )
                logging.info(f"Sending offer model to {source}")
                await self.cm.send_message(source, msg, is_compressed=True)
            else:
                logging.info("Discover join received before federation is running..")
                # starter node is going to send info to the new node
        else:
            logging.info(f"🔗  Dissmissing discover join from {source} | no active connections at the moment")

    async def _discover_discover_nodes_callback(self, source, message):
        logging.info(f"🔍  handle_discover_message | Trigger | Received discover_node message from {source} ")
        if len(self.engine.get_federation_nodes()) > 0:
            msg = self.cm.create_message(
                "offer",
                "offer_metric",
                n_neighbors=len(self.engine.get_federation_nodes()),
                loss=0 #self.engine.trainer.get_current_loss(),
            )
            logging.info(f"Sending offer metric to {source}")
            await self.cm.send_message(source, msg)
        else:
            logging.info(f"🔗  Dissmissing discover nodes from {source} | no active connections at the moment")

    async def _offer_offer_model_callback(self, source, message):
        logging.info(f"🔍  handle_offer_message | Trigger | Received offer_model message from {source}")
        await self.meet_node(source)
        if self.still_waiting_for_candidates():
            try:
                model_compressed = message.parameters
                if self.accept_model_offer(
                    source,
                    model_compressed,
                    message.rounds,
                    message.round,
                    message.epochs,
                    message.n_neighbors,
                    message.loss,
                ):
                    logging.info(f"🔧 Model accepted from offer | source: {source}")
                else:
                    logging.info(f"❗️ Model offer discarded | source: {source}")
                    self.add_to_discarded_offers(source)
            except RuntimeError:
                logging.info(f"❗️ Error proccesing offer model from {source}")
        else:
            logging.info(
                f"❗️ handfle_offer_message | NOT accepting offers | restructure: {self.get_restructure_process_lock().locked()} | waiting candidates: {self.still_waiting_for_candidates()}"
            )
            self.add_to_discarded_offers(source)

    async def _offer_offer_metric_callback(self, source, message):
        logging.info(f"🔍  handle_offer_message | Trigger | Received offer_metric message from {source}")
        await self.meet_node(source)
        if self.still_waiting_for_candidates():
            n_neighbors = message.n_neighbors
            loss = message.loss
            self.add_candidate(source, n_neighbors, loss)

    async def _link_connect_to_callback(self, source, message):
        logging.info(f"🔗  handle_link_message | Trigger | Received connect_to message from {source}")
        addrs = message.addrs
        for addr in addrs.split():
            await self.meet_node(addr)

    async def _link_disconnect_from_callback(self, source, message):
        logging.info(f"🔗  handle_link_message | Trigger | Received disconnect_from message from {source}")
        addrs = message.addrs
        for addr in addrs.split():
            await self.cm.disconnect(source, mutual_disconnection=False)
            await self.update_neighbors(addr, remove=True)
                


