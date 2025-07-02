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
            network_type: withinModal("network_type").value
        };
    }

    function setNetworkConfig(config) {
        if (!config || !modal) return;
        //TODO acceder a la config correctamente en el json
        withinModal("network_type").value = config.network_type
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

    function resetNetworkConfig(rep_switch=false) {
        if (!modal) return;

        document.getElementById("networkSwitch").checked = rep_switch;
        withinModal("#network_type").value = "3G";
        
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