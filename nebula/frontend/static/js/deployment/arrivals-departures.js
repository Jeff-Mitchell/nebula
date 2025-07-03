// Reputation System Module (Modal-Aware)
const ArrivalsDeparturesManager = (function () {
    let modal; 

    function initializeArrivalsDeparturesSystem(modalId = "activityModal") {
        modal = document.getElementById(modalId);
        if (!modal) {
            console.warn("Arrivals-Departures modal container not found.");
            return;
        }
        const switchInput = document.getElementById("activitySwitch");
        const configButton = document.getElementById("activityConfigureBtn");

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
        setupAdditionalParticipants();
    }

    function withinModal(selector) {
        return modal.querySelector(selector);
    }

    function getArrivalsDeparturesConfig() {
        //TODO
        // Obtain additionals participants and round of deployment
        // Set scheduled isolation for participants selected
        const config = {};

        const additionalParticipantsCount = parseInt(withinModal("#additionalParticipants").value);
        for (let i = 0; i < additionalParticipantsCount; i++) {
            if(document.getElementById("deploymentRoundSwitch").checked) {
                config.additionalParticipants.push({
                    time_start: parseInt(withinModal("#deploymentRound").value)
                });
            } else {
                config.additionalParticipants.push({
                    time_start: parseInt(withinModal(`#roundsAdditionalParticipant${i}`).value)
                });
            }
        }

        return config;
    }

    function setArrivalsDeparturesConfig(config) {
        if (!config || !modal) return;
        //TODO json configuration schema
        // Access federation nodes? or in controller?

        // Set additional participants
        if (config.additionalParticipants) {
            if (!Array.isArray(config.additionalParticipants)) {
                console.warn('Invalid mobility config: additionalParticipants must be an array');
                return;
            }

            withinModal("#additionalParticipants").value = config.additionalParticipants.length;
            const container = withinModal("#additional-participants-items");
            container.innerHTML = "";

            config.additionalParticipants.forEach((participant, index) => {
                if (typeof participant.round !== 'number') {
                    console.warn(`Invalid mobility config: participant ${index} round must be a number`);
                    return;
                }
                const participantItem = createParticipantItem(index);
                withinModal(`#roundsAdditionalParticipant${index}`).value = participant.round;
                container.appendChild(participantItem);
            });
        }
    }

    function createParticipantItem(index) {
        const participantItem = document.createElement("div");
        participantItem.style.marginLeft = "20px";
        participantItem.classList.add("additional-participant-item");

        const heading = document.createElement("h5");
        heading.textContent = `Round of deployment (participant ${index + 1})`;
        heading.classList.add("step-title")

        const input = document.createElement("input");
        input.type = "number";
        input.classList.add("form-control");
        input.id = `roundsAdditionalParticipant${index}`;
        input.placeholder = "round";
        input.min = "1";
        input.value = "1";
        input.style.display = "inline";
        input.style.width = "20%";

        participantItem.appendChild(heading);
        participantItem.appendChild(input);

        return participantItem;
    }

    function setupAdditionalParticipants() {
        withinModal("#additionalParticipants").addEventListener("change", function() {
            if(this.value > 0) {
                withinModal("#deploymentRoundTitle").style.display = "block";
                withinModal("#deploymentRoundDiv").style.display = "block";

                if (withinModal("#deploymentRoundSwitch").checked == false)
                {
                    const container = withinModal("#additional-participants-items");
                    container.innerHTML = "";

                    let additionalParticipants = withinModal("#additionalParticipants");
                    console.log("creating participant")
                    for (let i = 0; i < additionalParticipants.value; i++) {                    
                        const participantItem = createParticipantItem(i);
                        container.appendChild(participantItem);
                    }
                }
            } else {
                withinModal("#deploymentRoundTitle").style.display = "none";
                withinModal("#deploymentRoundDiv").style.display = "none";
            }
        });

        withinModal("#deploymentRoundSwitch").addEventListener("change", function() {
            if(this.checked) {
                withinModal("#deploymentRound").style.display = "inline";
                $(".additional-participant-item").remove();
            } else {
                withinModal("#deploymentRound").style.display = "none";

                //Generate additional participants
                const container = withinModal("#additional-participants-items");
                container.innerHTML = "";

                let additionalParticipants = withinModal("#additionalParticipants");
                for (let i = 0; i < additionalParticipants.value; i++) {
                    const participantItem = createParticipantItem(i);
                    container.appendChild(participantItem);
                }
            }
        });
    }

    function setupModalButtons() {
        const resetBtn = withinModal("#resetActivity");
        const saveBtn = withinModal("#saveActivity");
        const modal = document.getElementById("activityModal");

        if (resetBtn) {
            resetBtn.addEventListener("click", () => {
                ArrivalsDeparturesManager.resetArrivalsDeparturesConfig(true);
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

    function resetArrivalsDeparturesConfig(rep_switch=false) {
        if (!modal) return;

        document.getElementById("activitySwitch").checked = rep_switch;
        withinModal("#additionalParticipants").value = "0";
        withinModal("#additional-participants-items").innerHTML = "";
        withinModal("#deploymentRoundTitle").style.display = "none";
        withinModal("#deploymentRoundDiv").style.display = "none";

        //TODO same for departures
        
        if (!rep_switch){
            const switchInput = document.getElementById("activitySwitch");
            switchInput.dispatchEvent(new Event('change'));
        }
    }

    return {
        initializeArrivalsDeparturesSystem,
        getArrivalsDeparturesConfig,
        setArrivalsDeparturesConfig,
        resetArrivalsDeparturesConfig,
    };
})();

export default ArrivalsDeparturesManager;