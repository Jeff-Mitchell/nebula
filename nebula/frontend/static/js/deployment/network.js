// Reputation System Module (Modal-Aware)
const NetworkManager = (function () {
    let modal; 

    function initializeNetworkSystem(modalId = "networkModal") {
        modal = document.getElementById(modalId);
        if (!modal) {
            console.warn("Network modal container not found.");
            return;
        }

        const switchInput = document.getElementById("networkSwitch");
        const configButton = document.getElementById("networkConfigureBtn");

        if (!switchInput || !configButton) {
            console.warn("Not found switch/button");
            return;
        }

        // Show/hide button according to initial switch state
        configButton.style.display = switchInput.checked ? "inline-block" : "none";

        // Listener to show/hide configure button when switch changes 
        switchInput.addEventListener("change", () => {
            configButton.style.display = switchInput.checked ? "inline-block" : "none";
        });

        setupModalButtons();
    }

    function withinModal(selector) {
        return modal.querySelector(selector);
    }

    function getNetworkConfig() {
        return {
            enabled: document.getElementById("networkSwitch").checked,
            type: "Cellular_network_generation",
            generation: withinModal("#network_type").value
        };
    }

    function setNetworkConfig(config) {
        if (!config || !modal) return;
        withinModal("#network_type").value = config.network_type
    }

    function setupModalButtons() {
        const resetBtn = withinModal("#resetNetwork");
        const saveBtn = withinModal("#saveNetwork");
        const modal = document.getElementById("networkModal");

        if (resetBtn) {
            resetBtn.addEventListener("click", () => {
                NetworkManager.resetNetworkConfig(true);
            });
        }

        if (saveBtn) {
            saveBtn.addEventListener("click", () => {
                if (modal) {
                    const bsModal = bootstrap.Modal.getInstance(modal);
                    if (bsModal) {
                        bsModal.hide();
                    }
                }
            });
        }
    }

    function setupTooltipUpdater() {
        const select = withinModal("#network_type");
        const infoIcon = withinModal("#cellularGenerationIcon");

        if (!select || !infoIcon) return;

        const tooltips = {
                "3G": `3G Network parameters:
            • Bandwidth: 1mbit
            • Delay: 120ms ±30ms (normal)
            • Loss: 0.5%
            • Duplicate: 0.2%
            • Corrupt: 0.01%
            • Reordering: 0.1%`,

                "4G": `4G Network parameters:
            • Bandwidth: 10mbit
            • Delay: 60ms ±10ms (normal)
            • Loss: 0.1%
            • Duplicate: 0.1%
            • Corrupt: 0.005%
            • Reordering: 0.05%`,

                "5G": `5G Network parameters:
            • Bandwidth: 100mbit
            • Delay: 20ms ±5ms (normal)
            • Loss: 0.05%
            • Duplicate: 0.0%
            • Corrupt: 0.001%
            • Reordering: 0.01%`
            };

        function updateTooltip() {
            const selected = select.value;
            infoIcon.setAttribute("title", tooltips[selected] || "");
        }

        select.addEventListener("change", updateTooltip);
        updateTooltip(); 
    }

    function resetNetworkConfig(rep_switch=false) {
        if (!modal) return;

        document.getElementById("networkSwitch").checked = rep_switch;
        withinModal("#network_type").value = "3G";
        setupTooltipUpdater();
        
        if (!rep_switch){
            const switchInput = document.getElementById("networkSwitch");
            switchInput.dispatchEvent(new Event('change'));
        }
    }

    return {
        initializeNetworkSystem,
        getNetworkConfig,
        setNetworkConfig,
        resetNetworkConfig,
    };
})();

export default NetworkManager;