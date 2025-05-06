"""
It is an example of the .json configuration file structure required
for the NebulaPluginLoader to load all plugins defined for the current
scenario.

JSON Configuration Example:
---------------------------
{
    "plugins": ["reputation", "trust"],

    "reputation": {
        "threshold": 0.75,
        "decay_factor": 0.05,
        "history_length": 100
    },

    "trust": {
        "initial_trust": 0.5,
        "trust_update_rate": 0.1,
        "penalty_factor": 0.2
    }
}

Plugin Directory Structure:
---------------------------
All plugins must follow a standardized directory and file naming convention 
to be correctly detected and loaded by NebulaPluginLoader.

Base path where plugins should be located:
    /nebula/nebula/core/

Each plugin should be placed in its own directory inside the base path, 
with a filename matching the plugin name (lowercase).

Example for the "reputation" plugin:
    /nebula/nebula/core/reputation/reputation.py

Example for the "trust" plugin:
    /nebula/nebula/core/trust/trust.py

Plugin Class Requirements:
--------------------------
Each plugin module must define at least one class that:

- Inherits from `NebulaPlugin`
- Implements the `initialize_plugin()` method

There are **no strict naming conventions** for the class name itself; 
`NebulaPluginLoader` will automatically detect and instantiate the first 
valid plugin class found in the module.

Plugins receive their configuration as a dictionary when instantiated.

Example for "reputation":
-----------------------------------------------------
File: /nebula/nebula/core/reputation/reputation.py

from nebula.nebula.core.plugin_loader import NebulaPlugin

class ReputationModel(NebulaPlugin):
    def __init__(self, config: dict):
        self.threshold = config.get("threshold", 0.75)
        self.decay_factor = config.get("decay_factor", 0.05)
        self.history_length = config.get("history_length", 100)

    async def initialize_plugin(self):
        # Initialization logic here
        pass
-----------------------------------------------------

Example for "trust":
-----------------------------------------------------
File: /nebula/nebula/core/trust/trust.py

from nebula.nebula.core.plugin_loader import NebulaPlugin

class TrustSystem(NebulaPlugin):
    def __init__(self, config: dict):
        self.initial_trust = config.get("initial_trust", 0.5)
        self.trust_update_rate = config.get("trust_update_rate", 0.1)
        self.penalty_factor = config.get("penalty_factor", 0.2)

    async def initialize_plugin(self):
        # Initialization logic here
        pass
-----------------------------------------------------

Important Notes:
---------------
- The plugin class name can be arbitrary but must inherit from `NebulaPlugin`.
- Only one valid plugin class is allowed per module; if more exist, only the first will be used.
- The plugin module filename must still match the plugin name (lowercase).
- The plugin class must implement `initialize_plugin()`.
- Each plugin receives its configuration as a dictionary.
- `NebulaPluginLoader` automatically detects and instantiates the appropriate plugin.
"""

import logging
from nebula.addons.functions import print_msg_box
from nebula.core.utils.locker import Locker
from abc import ABC, abstractmethod
import importlib.util
import os
import inspect
import inspect

class NebulaPlugin(ABC):
    @abstractmethod
    async def initialize_plugin(self):
        """Method to be implemented by all plugins. 
        It should handle any necessary initialization logic."""
        raise NotImplementedError
    

class NebulaPluginLoader:
    _instance = None
    _lock = Locker("_nebula_pluging_loader_lock", async_lock=False)

    def __new__(cls, config_json=None, base_path="/nebula/nebula/core"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_json=None, base_path="/nebula/nebula/core"):
        """Initializes the plugin loader with the given configuration JSON and base path."""
        if self._initialized:
            return 

        self._config_json = config_json or {}
        self._base_path = base_path
        self._plugins : dict[str, NebulaPlugin] = {}
        self._initialized = True
        self._verbose = False

    def load_plugins(self):
        """Dynamically loads the plugins defined in the JSON configuration."""
        if not self._config_json:
            raise ValueError("No Configuration file provided. [JSON] file required.")

        plugin_names = self._config_json.get("plugins", [])
        for name in plugin_names:
            module_path = os.path.join(self._base_path, name)
            module_file = os.path.join(module_path, f"{name}.py")

            if os.path.exists(module_file):
                plugin_instance = self._load_plugin(module_file, self._config_json.get(name, {}))
                if plugin_instance:
                    self._plugins[name] = plugin_instance
            else:
                logging.error(f"⚠️ Plugin {name} not found at {module_file}")

    def _load_plugin(self, module_file, config):
        """Loads a plugin dynamically and initializes it with its configuration."""
        spec = importlib.util.spec_from_file_location("plugin_module", module_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, NebulaPlugin) and obj is not NebulaPlugin:
                    return obj(config)

            logging.error(f"⚠️ No valid plugin class found in {module_file}")
        return None

    async def initialize_plugins(self):
        """Calls the asynchronous initialization method of each loaded plugin."""
        for plugin_name, plugin in self._plugins.items():
            if self._verbose:
                logging.info(f"Initializing plugin name: {plugin_name}")
            await plugin.initialize_plugin()
                       
    def get_plugin(self, name):
        """Returns an instance of the plugin if it has been loaded."""
        return self._plugins.get(name, None)