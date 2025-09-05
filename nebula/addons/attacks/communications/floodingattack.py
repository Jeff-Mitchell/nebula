import logging
from functools import wraps

from nebula.addons.attacks.communications.communicationattack import CommunicationAttack


class FloodingAttack(CommunicationAttack):
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
        self.flooding_factor = int(attack_params.get("flooding_factor", 3))
        self.target_percentage = int(attack_params.get("target_percentage", 50))
        self.selection_interval = int(attack_params.get("selection_interval", 1))

        # Store poisoned_node_percent if provided (for potential future use)
        self.poisoned_node_percent = attack_params.get("poisoned_node_percent")

        self.verbose = False

        super().__init__(
            engine,
            engine._cm,
            "send_model",
            round_start,
            round_stop,
            attack_interval,
            self.flooding_factor,
            self.target_percentage,
            self.selection_interval,
        )

    def decorator(self, flooding_factor: int):
        """
        Decorator that adds a delay to the execution of the original method.

        Args:
            flooding_factor (int): The number of times to repeat the function execution.

        Returns:
            function: A decorator function that wraps the target method with the delay logic.
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if len(args) > 1:
                    dest_addr = args[1]
                    if dest_addr in self.targets:
                        logging.info(f"[FloodingAttack] Flooding message to {dest_addr} by {flooding_factor} times")
                        for i in range(flooding_factor):
                            if self.verbose:
                                logging.info(
                                    f"[FloodingAttack] Sending duplicate {i + 1}/{flooding_factor} to {dest_addr}"
                                )
                            _, dest_addr, _, serialized_model, weight = args  # Exclude self argument
                            new_args = [dest_addr, i, serialized_model, weight]
                            await func(*new_args, **kwargs)
                _, dest_addr, _, serialized_model, weight = args  # Exclude self argument
                new_args = [dest_addr, i, serialized_model, weight]
                return await func(*new_args)

            return wrapper

        return decorator
