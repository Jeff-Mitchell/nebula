// Training Configuration Module
const TrainingManager = (function() {
    function initializeTraining() {
        setupCommunicationMode();
    }

    function setupCommunicationMode() {
        const epochMode = document.getElementById("epochMode");
        const batchMode = document.getElementById("batchMode");
        const epochSettings = document.getElementById("epochSettings");
        const batchSettings = document.getElementById("batchSettings");

        // Add event listeners to radio buttons
        epochMode.addEventListener("change", function() {
            if (this.checked) {
                epochSettings.style.display = "block";
                batchSettings.style.display = "none";
            }
        });

        batchMode.addEventListener("change", function() {
            if (this.checked) {
                epochSettings.style.display = "none";
                batchSettings.style.display = "block";
            }
        });
    }

    function getTrainingConfig() {
        const communicationMode = document.querySelector('input[name="communicationMode"]:checked').value;
        return {
            epochs: parseInt(document.getElementById("epochs").value),
            communication_mode: communicationMode,
            batches_per_communication: parseInt(document.getElementById("batchesPerCommunication").value)
        };
    }

    function setTrainingConfig(config) {
        if (!config) return;

        document.getElementById("epochs").value = config.epochs || 3;

        // Set radio button based on communication mode
        const mode = config.communication_mode || "epoch";
        document.getElementById(mode === "epoch" ? "epochMode" : "batchMode").checked = true;

        // Show/hide appropriate settings
        document.getElementById("epochSettings").style.display = mode === "epoch" ? "block" : "none";
        document.getElementById("batchSettings").style.display = mode === "batch" ? "block" : "none";

        document.getElementById("batchesPerCommunication").value = config.batches_per_communication || 1;
    }

    function resetTrainingConfig() {
        document.getElementById("epochs").value = 3;
        document.getElementById("epochMode").checked = true;
        document.getElementById("batchMode").checked = false;
        document.getElementById("batchesPerCommunication").value = 1;
        document.getElementById("epochSettings").style.display = "block";
        document.getElementById("batchSettings").style.display = "none";
    }

    return {
        initializeTraining,
        getTrainingConfig,
        setTrainingConfig,
        resetTrainingConfig
    };
})();

export default TrainingManager;
