import logging
from typing import TYPE_CHECKING

from nebula.core.utils.locker import Locker

if TYPE_CHECKING:
    from nebula.core.situationalawareness.discovery.federationconnector import FederationConnector

VANILLA_LEARNING_RATE = 1e-3
FR_LEARNING_RATE = 1e-3
MAX_ROUNDS = 20
DEFAULT_WEIGHT_MODIFIER = 3


class FastReboot:
    def __init__(
        self,
        node_manager: "FederationConnector",
        max_rounds_application=MAX_ROUNDS,  # Max rounds to be applied FastReboot
        weight_modifier=DEFAULT_WEIGHT_MODIFIER,
        default_learning_rate=VANILLA_LEARNING_RATE,  # Stable value for learning rate
        upgrade_learning_rate=FR_LEARNING_RATE,  # Increased value for learning rate
    ):
        logging.info("🌐  Initializing FastReboot")
        self._node_manager = node_manager
        self._max_rounds = max_rounds_application
        self._weight_mod_value = weight_modifier
        self._default_lr = default_learning_rate
        self._upgrade_lr = upgrade_learning_rate
        self._current_lr = default_learning_rate
        self._learning_rate_lock = Locker(name="learning_rate_lock", async_lock=True)

        self._weight_modifier = {}
        self._weight_modifier_lock = Locker(name="weight_modifier_lock", async_lock=True)
        self._rounds_pushed_lock = Locker(name="rounds_pushed_lock", async_lock=True)
        self._rounds_pushed = 0

        self._fr_in_progress = False

    @property
    def nm(self):
        return self._node_manager

    async def set_rounds_pushed(self, rp):
        await self._rounds_pushed_lock.acquire_async()
        self.rounds_pushed = rp
        await self._rounds_pushed_lock.release_async()

    async def get_current_learning_rate(self):
        await self._learning_rate_lock.acquire_async()
        lr = self._current_lr
        await self._learning_rate_lock.release_async()
        return lr

    async def discard_fastreboot_for(self, addr):
        await self._weight_modifier_lock.acquire_async()
        try:
            del self._weight_modifier[addr]
        except KeyError:
            pass
        await self._weight_modifier_lock.release_async()

    async def _set_learning_rate(self, lr):
        await self._learning_rate_lock.acquire_async()
        self._current_lr = lr
        await self._learning_rate_lock.release_async()

    async def add_fastReboot_addr(self, addr):
        await self._weight_modifier_lock.acquire_async()
        if addr not in self._weight_modifier:
            self._fr_in_progress = True
            wm = self._weight_mod_value
            logging.info(
                f"📝 Registering | FastReboot registered for source {addr} | round application: {self._max_rounds} | multiplier value: {wm}"
            )
            self._weight_modifier[addr] = (wm, 1)
            await self._set_learning_rate(self._upgrade_lr)
            current_lr = await self.get_current_learning_rate()
            await self.nm.update_learning_rate(current_lr)
        await self._weight_modifier_lock.release_async()

    async def _remove_weight_modifier(self, addr):
        logging.info(f"📝 Removing | FastReboot removed for source {addr}")
        del self._weight_modifier[addr]

    async def _weight_modifiers_empty(self):
        await self._weight_modifier_lock.acquire_async()
        empty = False if self._weight_modifier else True
        await self._weight_modifier_lock.release_async()
        return empty

    async def apply_weight_strategy(self, updates: dict):
        if await self._weight_modifiers_empty():
            if self._fr_in_progress:
                await self._end_fastreboot()
            return
        logging.info("🔄  Applying FastReboot Strategy...")
        # We must lower the weight_modifier value if a round jump has been occured
        # as many times as rounds have been jumped
        if self._rounds_pushed:
            logging.info("🔄  There are rounds being pushed...")
            for i in range(0, self.rounds_pushed):
                logging.info("🔄  Update | weights being updated cause of push...")
                self._update_weight_modifiers()
            self.rounds_pushed = 0
        for addr, update in updates.items():
            weightmodifier, rounds = await self._get_weight_modifier(addr)
            if weightmodifier != 1:
                logging.info(
                    f"📝 Appliying FastReboot strategy | addr: {addr} | multiplier value: {weightmodifier}, rounds applied: {rounds}"
                )
                model, weight = update
                updates.update({addr: (model, weight * weightmodifier)})
        await self._update_weight_modifiers()

    # TODO integrar en el get_wegith_modifier para que se actualice cuando se pide y ahi se compruebe si hay q eliminar una entrada
    async def _update_weight_modifiers(self):
        await self._weight_modifier_lock.acquire_async()
        if self._weight_modifier:
            logging.info("🔄  Update | weights being updated")
            remove_addrs = []
            for addr, (weight, rounds) in self._weight_modifier.items():
                new_weight = weight - 1 / (rounds**2)
                rounds = rounds + 1
                if new_weight > 1 and rounds <= self._max_rounds:
                    self._weight_modifier[addr] = (new_weight, rounds)
                else:
                    remove_addrs.append(addr)
            for a in remove_addrs:
                await self._remove_weight_modifier(a)
        await self._weight_modifier_lock.release_async()

    async def _end_fastreboot(self):
        await self._weight_modifier_lock.acquire_async()
        if not self._weight_modifier and await self._is_lr_modified():
            logging.info("🔄  Finishing | FastReboot is completed")
            self._fr_in_progress = False
            await self._set_learning_rate(self._default_lr)
            await self.nm.update_learning_rate(self._default_lr)
        await self._weight_modifier_lock.release_async()

    async def _get_weight_modifier(self, addr):
        await self._weight_modifier_lock.acquire_async()
        wm = self._weight_modifier.get(addr, (1, 0))
        await self._weight_modifier_lock.release_async()
        return wm

    async def _is_lr_modified(self):
        await self._learning_rate_lock.acquire_async()
        mod = self._current_lr == self._upgrade_lr
        await self._learning_rate_lock.release_async()
        return mod
