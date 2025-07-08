// Reputation System Module (Modal-Aware)
const ArrivalsDeparturesManager = (function () {
    let modal;
    let departures_count = 0; 

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
        setupDeparturesParticipants();
    }

    function withinModal(selector) {
        return modal.querySelector(selector);
    }

    function getArrivalsDeparturesConfig() {
        const _enabled = document.getElementById("activitySwitch").checked;
        if (! _enabled){
            const config = {
                enabled: false
            };
            return config;
        }

        const config = {
            enabled: _enabled,
            additionalParticipants: [],
            departures: []
        };
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

        let round_departure = 0;
        let departure_duration = 0;
        const topologyData = window.TopologyManager.getData()
        let initialParticipants = topologyData.nodes.length 
        let numberInitiallParticipants = parseInt(initialParticipants, 10);

        let departures_count = additionalParticipantsCount + numberInitiallParticipants

        for (let i = 0; i < departures_count; i++) {
            if (!withinModal("#departuresSwitch").checked)
                break;

            if(withinModal(`#roundDepartureParticipant${i}`).value != ""){
                round_departure = parseInt(withinModal(`#roundDepartureParticipant${i}`).value);
            } else {
                round_departure = withinModal(`#roundDepartureParticipant${i}`).value
            }

            if(withinModal(`#durationDepartureParticipant${i}`).value != ""){
                departure_duration = parseInt(withinModal(`#durationDepartureParticipant${i}`).value);
            } else {
                departure_duration = withinModal(`#durationDepartureParticipant${i}`).value;
            }

            config.departures.push({
                round_start: round_departure,
                duration: departure_duration
            });
        }

        return config;
    }

    function setArrivalsDeparturesConfig(config) {
        if (!config || !modal) return;
   
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
        heading.textContent = `Round of deployment (Additional participant ${index + 1})`;
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

    function createDepartureItem(index) {
        const participantItem = document.createElement("div");
        participantItem.style.marginLeft = "20px";
        participantItem.classList.add("departure-participant-item");

        const heading = document.createElement("h5");
        heading.textContent = `Round of departure - Rounds duration (participant ${index + 1})`;
        heading.classList.add("step-title")

        const input = document.createElement("input");
        input.type = "number";
        input.classList.add("form-control");
        input.id = `roundDepartureParticipant${index}`;
        input.placeholder = "round";
        input.min = "1";
        input.value = "";
        input.style.display = "inline";
        input.style.width = "20%";

        const input2 = document.createElement("input");
        input2.type = "number";
        input2.classList.add("form-control");
        input2.id = `durationDepartureParticipant${index}`;
        input2.placeholder = "infinite";
        input2.min = "1";
        input2.value = "";
        input2.style.display = "inline";
        input2.style.width = "20%";
        input2.style.marginLeft = "40px"

        participantItem.appendChild(heading);
        participantItem.appendChild(input);
        participantItem.appendChild(input2);

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
                console.log("Removing additional participants")
                withinModal("#deploymentRound").style.display = "block";
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

    function setupDeparturesParticipants(){
        withinModal("#departuresDiv").style.display = "block";
        withinModal("#departuresSwitch").addEventListener("change", function() {
            if(!this.checked) {
                console.log("Removing departure configuration")
                withinModal("#departures-participants-items").innerHTML = "";
            } else {
                let additionalParticipants = withinModal("#additionalParticipants").value;
                const topologyData = window.TopologyManager.getData()
                let initialParticipants = topologyData.nodes.length

                let numberAdditionalParticipants = parseInt(additionalParticipants, 10);
                let numberInitiallParticipants = parseInt(initialParticipants, 10);

                departures_count = numberAdditionalParticipants + numberInitiallParticipants

                const container = withinModal("#departures-participants-items");
                container.innerHTML = "";

                for (let i = 0; i < numberInitiallParticipants + numberAdditionalParticipants; i++) {
                    const departureItem = createDepartureItem(i);
                    container.appendChild(departureItem);
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
        withinModal("#departuresSwitch").checked = false;
        withinModal("#additional-participants-items").innerHTML = "";
        withinModal("#departures-participants-items").innerHTML = "";
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