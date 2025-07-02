// Reputation System Module (Modal-Aware)
const ReputationManager = (function () {
    let modal; // Referencia al contenedor modal

    function initializeReputationSystem(modalId = "reputation-modal") {
        modal = document.getElementById(modalId);
        if (!modal) {
            console.warn("Reputation modal container not found.");
            return;
        }
        setupWeightingFactor();
        setupWeightValidation();
        setupInitialReputation();
    }

    function withinModal(selector) {
        return modal.querySelector(selector);
    }

    function setupWeightingFactor() {
        withinModal("#weighting-factor").addEventListener("change", function () {
            const showWeights = this.value === "static";
            modal.querySelectorAll(".weight-input").forEach(input => {
                input.style.display = showWeights ? "inline-block" : "none";
            });
        });
    }

    function setupWeightValidation() {
        modal.querySelectorAll(".weight-input").forEach(input => {
            input.addEventListener("input", validateWeights);
        });
    }

    function validateWeights() {
        let totalWeight = 0;
        modal.querySelectorAll(".weight-input").forEach(input => {
            const checkbox = input.closest(".form-group").querySelector("input[type=checkbox]");
            if (checkbox?.checked && input.style.display !== "none" && input.value) {
                totalWeight += parseFloat(input.value) || 0;
            }
        });
        withinModal("#weight-warning").style.display = totalWeight > 1 ? "block" : "none";
    }

    function setupInitialReputation() {
        withinModal("#initial-reputation").addEventListener("blur", function () {
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            const value = parseFloat(this.value);
            if (value < min) this.value = min;
            else if (value > max) this.value = max;
        });
    }

    function getReputationConfig() {
        return {
            enabled: withinModal("#reputationSwitch").checked,
            initialReputation: parseFloat(withinModal("#initial-reputation").value) || 0.2,
            weightingFactor: withinModal("#weighting-factor").value,
            metrics: {
                modelSimilarity: {
                    enabled: withinModal("#model-similarity").checked,
                    weight: parseFloat(withinModal("#weight-model-similarity").value) || 0
                },
                numMessages: {
                    enabled: withinModal("#num-messages").checked,
                    weight: parseFloat(withinModal("#weight-num-messages").value) || 0
                },
                modelArrivalLatency: {
                    enabled: withinModal("#model-arrival-latency").checked,
                    weight: parseFloat(withinModal("#weight-model-arrival-latency").value) || 0
                },
                fractionParametersChanged: {
                    enabled: withinModal("#fraction-parameters-changed").checked,
                    weight: parseFloat(withinModal("#weight-fraction-parameters-changed").value) || 0
                }
            }
        };
    }

    function setReputationConfig(config) {
        if (!config || !modal) return;

        // Set reputation enabled/disabled
        withinModal("#reputationSwitch").checked = config.enabled;
        withinModal("#reputation-metrics").style.display = config.enabled ? "block" : "none";
        withinModal("#reputation-settings").style.display = config.enabled ? "block" : "none";
        withinModal("#weighting-settings").style.display = config.enabled ? "block" : "none";

        // Set initial reputation
        withinModal("#initial-reputation").value = config.initialReputation ?? 0.2;

        // Set weighting factor
        withinModal("#weighting-factor").value = config.weightingFactor ?? "dynamic";
        const showWeights = config.weightingFactor === "static";
        modal.querySelectorAll(".weight-input").forEach(input => {
            input.style.display = showWeights ? "inline-block" : "none";
        });

        // Set metrics
        const metrics = config.metrics || {};
        withinModal("#model-similarity").checked = metrics.modelSimilarity?.enabled || false;
        withinModal("#weight-model-similarity").value = metrics.modelSimilarity?.weight ?? 0;

        withinModal("#num-messages").checked = metrics.numMessages?.enabled || false;
        withinModal("#weight-num-messages").value = metrics.numMessages?.weight ?? 0;

        withinModal("#model-arrival-latency").checked = metrics.modelArrivalLatency?.enabled || false;
        withinModal("#weight-model-arrival-latency").value = metrics.modelArrivalLatency?.weight ?? 0;

        withinModal("#fraction-parameters-changed").checked = metrics.fractionParametersChanged?.enabled || false;
        withinModal("#weight-fraction-parameters-changed").value = metrics.fractionParametersChanged?.weight ?? 0;

        validateWeights();
    }

    function resetReputationConfig() {
        if (!modal) return;

        withinModal("#reputationSwitch").checked = false;
        withinModal("#reputation-metrics").style.display = "none";
        withinModal("#reputation-settings").style.display = "none";
        withinModal("#weighting-settings").style.display = "none";
        withinModal("#initial-reputation").value = "0.2";
        withinModal("#weighting-factor").value = "dynamic";
        withinModal("#weight-warning").style.display = "none";

        // Reset metrics
        withinModal("#model-similarity").checked = false;
        withinModal("#weight-model-similarity").value = "0";
        withinModal("#num-messages").checked = false;
        withinModal("#weight-num-messages").value = "0";
        withinModal("#model-arrival-latency").checked = false;
        withinModal("#weight-model-arrival-latency").value = "0";
        withinModal("#fraction-parameters-changed").checked = false;
        withinModal("#weight-fraction-parameters-changed").value = "0";

        // Hide weight inputs
        modal.querySelectorAll(".weight-input").forEach(input => {
            input.style.display = "none";
        });
    }

    return {
        initializeReputationSystem,
        getReputationConfig,
        setReputationConfig,
        resetReputationConfig,
    };
})();

document.addEventListener("DOMContentLoaded", () => {
    const switchInput = document.getElementById("reputationSwitch");
    const configButton = document.getElementById("reputationConfigureBtn");

    if (!switchInput || !configButton) {
        console.warn("No se encontró switch o botón");
        return;
    }

    // Show/hide button according to initial switch state
    configButton.style.display = switchInput.checked ? "inline-block" : "none";

    // Listener to show/hide configure button when switch changes 
    switchInput.addEventListener("change", () => {
        configButton.style.display = switchInput.checked ? "inline-block" : "none";
    });
});

export default ReputationManager;
