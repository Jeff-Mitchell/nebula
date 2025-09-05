import asyncio
import logging
from functools import wraps

from nebula.addons.attacks.communications.communicationattack import CommunicationAttack


class DelayerAttack(CommunicationAttack):
    """
    Implements an attack that delays the execution of a target method by a specified amount of time.
    """

    def __init__(self, engine, attack_params: dict):
        """
        Initializes the DelayerAttack with the engine and attack parameters.

        Args:
            engine: The engine managing the attack context.
            attack_params (dict): Parameters for the attack, including the delay duration.
        """
        try:
            round_start = int(attack_params["round_start_attack"])
            round_stop = int(attack_params["round_stop_attack"])
            attack_interval = int(attack_params["attack_interval"])
        except KeyError as e:
            raise ValueError(f"Missing required attack parameter: {e}") from e
        except ValueError as e:
            raise ValueError("Invalid value in attack_params. Ensure all values are integers.") from e
        
        # Handle optional parameters with defaults
        self.delay = int(attack_params.get("delay", 5))
        self.target_percentage = int(attack_params.get("target_percentage", 50))
        self.selection_interval = int(attack_params.get("selection_interval", 1))
        
        # Store poisoned_node_percent if provided (for potential future use)
        self.poisoned_node_percent = attack_params.get("poisoned_node_percent")

        super().__init__(
            engine,
            engine._cm,
            "send_model",
            round_start,
            round_stop,
            attack_interval,
            self.delay,
            self.target_percentage,
            self.selection_interval,
        )

    def decorator(self, delay: int):
        """
        Decorator that adds a delay to the execution of the original method.

        Args:
            delay (int): The time in seconds to delay the method execution.

        Returns:
            function: A decorator function that wraps the target method with the delay logic.
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if len(args) > 1:
                    dest_addr = args[1]
                    if dest_addr in self.targets:
                        logging.info(f"[DelayerAttack] Delaying model propagation to {dest_addr} by {delay} seconds")
                        await asyncio.sleep(delay)
                _, *new_args = args  # Exclude self argument
                return await func(*new_args)

            return wrapper

        return decorator
