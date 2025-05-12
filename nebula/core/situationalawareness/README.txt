 █████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ██████╗ 
██╔══██╗██║   ██║╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗
███████║██║   ██║   ██║   ███████║██║   ██║██████╔╝
██╔══██║██║   ██║   ██║   ██╔══██║██║   ██║██╔═██╗ 
██║  ██║╚██████╔╝   ██║   ██║  ██║╚██████╔╝██║  ██╗
╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝

Alejandro Avilés Serrano.


==========================================

Overview
--------
The Situational Awareness (SA) module is designed to provide intelligent, policy-driven coordination 
in decentralized or federated environments. It enables discovery of peers, reasoning about system 
state, and proposing and arbitrating actions dynamically. The module is modular and designed 
with extensibility, asynchronous execution, and policy injection in mind.

────────────────────────────────────────────────────────────────────────────
Module Structure
────────────────────────────────────────────────────────────────────────────

SituationalAwareness/
├── ISADiscovery
│   ├── CandidateSelector      # Strategy to select neighbors from discovered models
│   └── ModelHandler           # Handles incoming and outgoing models
│
├── ISAReasoner
│   ├── SAComponents           # Autonomous units proposing SACommands
|   |    ├──SANetwork
│   └── ArbitrationPolicy      # Resolves conflicts between proposed actions

Each part of the system is defined by interfaces (ABCs) and can be extended via plugins or 
custom implementations registered via configuration.

────────────────────────────────────────────────────────────────────────────
Core Responsibilities
────────────────────────────────────────────────────────────────────────────

✔ Discovery of federation participants using customizable model and candidate policies  
✔ Management of dynamic topologies through neighbor policies  
✔ Generation of actionable suggestions by SAComponents  
✔ Resolution of command conflicts via arbitration logic  
✔ Coordination of the reasoning and decision process under asynchronous operation

────────────────────────────────────────────────────────────────────────────
Component Descriptions
────────────────────────────────────────────────────────────────────────────

1. ISADiscovery (FederationConnector)
   - Orchestrates discovery of peers using a selected CandidateSelector and ModelHandler.
   - Applies NeighborPolicy based on strict or adaptive topology configuration.

2. CandidateSelector (ABC)
   - Selects which discovered peers should be considered for inclusion.
   - Strategies include `random`, `greedy`, or `score-based`.

3. ModelHandler (ABC)
   - Manages how models are accepted, validated, and retrieved for dissemination.
   - Responsible for local model lifecycle during discovery.

4. ISAReasoner (SAReasoner)
   - Core decision-maker.
   - Registers and coordinates SAComponents that propose SACommand actions.
   - Interfaces with the SuggestionBuffer and applies an ArbitrationPolicy to select valid actions.

5. SAComponents (SAMComponent subclasses)
   - Autonomous agents that monitor system state and suggest actions.
   - Examples: `sa_network`, `sa_latency`, etc.
   - Emit actionable suggestions (`SACommand`) with priorities.

6. ArbitrationPolicy (ABC)
   - Implements the resolution logic for conflicting commands.
   - Allows tie-breaking strategies to be injected.

7. SuggestionBuffer (singleton)
   - Synchronization hub for collecting SACommand suggestions from SAComponents.
   - Notifies the Reasoner when all expected suggestions for an event are received.

8. SACommand (base class)
   - Encodes the command to be executed (action, target, priority, parallelization, etc.).
   - Supports priority comparison and conflict detection between commands.

9. SAModuleAgent (ABC)
   - Interface that allows SAComponents to report their suggestions and mark completion.

────────────────────────────────────────────────────────────────────────────
Execution Flow
────────────────────────────────────────────────────────────────────────────

1.- FederationConnector begins the discovery phase.

2.- Discovered models are handled by ModelHandler and filtered by CandidateSelector.

3.- If accepted, neighbor connection policies are applied via NeighborPolicy.

4.- SAComponents observe and propose commands asynchronously via SAModuleAgent.

5.- SuggestionBuffer collects all proposals for an event and notifies SAReasoner.

6.- SAReasoner invokes ArbitrationPolicy to resolve conflicts and execute valid commands.

────────────────────────────────────────────────────────────────────────────
Example Configuration (JSON)
────────────────────────────────────────────────────────────────────────────

```json
"situational_awareness": {
  "strict_topology": true,
  "sa_discovery": {
    "candidate_selector": "random",
    "model_handler": "std",
    "verbose": true
  },
  "sa_reasoner": {
    "arbitration_policy": "sap",
    "verbose": true,
    "sar_components": {
      "sa_network": true
    },
    "sa_network": {
      "neighbor_policy": "random",
      "verbose": true
    }
  }
}