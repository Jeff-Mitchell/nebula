// Dataset-related UI and logic (selection, models, Edge-IIoTset download & progress)
import Utils from './utils.js';

const DatasetManager = (function() {
    // Progress state
    let edgeDlTimer = null;
    let edgeDlProgress = 0;

    function initialize() {
        setupDatasetListeners();
        populateDatasetOptions();
        // Initial toggle for Edge-IIoTset button if selected by default
        const datasetSelect = document.getElementById('datasetSelect');
        if (datasetSelect) {
            updateEdgeDatasetDownloadButton(datasetSelect.value);
        }
    }

    function setupDatasetListeners() {
        const datasetSelect = document.getElementById("datasetSelect");
        if (datasetSelect) {
            datasetSelect.addEventListener("change", () => {
                updateModelOptions(datasetSelect.value);
                updateEdgeDatasetDownloadButton(datasetSelect.value);
            });
        }
    }

    function populateDatasetOptions() {
        const datasetSelect = document.getElementById("datasetSelect");
        if (!datasetSelect) return;

        datasetSelect.innerHTML = "";
        const datasets = ['MNIST', 'FashionMNIST', 'EMNIST', 'CIFAR10', 'CIFAR100', 'Edge-IIoTset', 'Edge-IIoTset-binary'];
        datasets.forEach(dataset => {
            const option = document.createElement("option");
            option.value = dataset;
            option.textContent = dataset;
            datasetSelect.appendChild(option);
        });
        datasetSelect.value = 'MNIST';
        datasetSelect.dispatchEvent(new Event('change'));
    }

    function updateModelOptions(dataset) {
        const modelSelect = document.getElementById("modelSelect");
        if (!modelSelect) return;
        modelSelect.innerHTML = "";
        const models = getModelsForDataset(dataset);
        models.forEach(model => {
            const option = document.createElement("option");
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    }

    function getModelsForDataset(dataset) {
        switch(dataset.toLowerCase()) {
            case 'mnist':
            case 'fashionmnist':
            case 'emnist':
                return ['MLP', 'CNN'];
            case 'cifar10':
                return ['CNN', 'ResNet9', 'fastermobilenet', 'simplemobilenet', 'CNNv2', 'CNNv3'];
            case 'cifar100':
                return ['CNN'];
            case 'edge-iiotset':
                return ['EdgeIIoTsetMLP'];
            case 'edge-iiotset-binary':
                return ['EdgeIIoTsetMLP'];
            default:
                return ['MLP', 'CNN'];
        }
    }

    async function updateEdgeDatasetDownloadButton(selectedDataset) {
        const btn = document.getElementById('download-edge-dataset-btn');
        const progress = document.getElementById('edge-dataset-download-progress');
        const status = document.getElementById('edge-dataset-download-status');
        if (!btn) return;
        const isEdgeDataset = (selectedDataset === 'Edge-IIoTset' || selectedDataset === 'Edge-IIoTset-binary');

        if (!isEdgeDataset) {
            btn.style.display = 'none';
            if (progress) progress.style.display = 'none';
            if (status) status.style.display = 'none';
            return;
        }

        const present = await checkEdgeDatasetPresent();
        btn.style.display = present ? 'none' : 'inline-block';

        if (isEdgeDataset && !btn._nebulaBound) {
            btn.addEventListener('click', async () => {
                startEdgeDownloadProgress();
                try {
                    const res = await fetch('/platform/api/datasets/edge-iiotset/download', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const data = await res.json().catch(() => ({}));
                    console.log('[Edge-IIoTset] Download API response:', res.status, data);
                    if (res.ok) {
                        const status = (data && data.status) || '';
                        if (status === 'already_present' || status === 'ready_from_local') {
                            const msg = data.message || 'Dataset is ready.';
                            completeEdgeDownloadProgress(msg);
                        } else if (status === 'downloaded' || status === 'started') {
                            await pollEdgeDatasetPresenceWithProgress();
                        } else {
                            completeEdgeDownloadProgress(data.message || 'Done');
                        }
                        await checkEdgeDatasetPresent(true);
                        const presentNow = await checkEdgeDatasetPresent();
                        btn.style.display = presentNow ? 'none' : 'inline-block';
                    } else {
                        const detail = data && (data.detail || data.message) ? (data.detail || data.message) : 'Failed to download dataset';
                        errorEdgeDownloadProgress(`Error: ${detail}`);
                    }
                } catch (err) {
                    console.error('[Edge-IIoTset] Download API error:', err);
                    errorEdgeDownloadProgress(`Network error: ${err}`);
                }
            });
            btn._nebulaBound = true;
        }
    }

    async function checkEdgeDatasetPresent(forceRefresh = false) {
        if (window._edgeDatasetPresent !== undefined && !forceRefresh) {
            return window._edgeDatasetPresent;
        }
        try {
            const res = await fetch('/platform/api/datasets/edge-iiotset/status');
            if (!res.ok) throw new Error(String(res.status));
            const data = await res.json();
            console.log('[Edge-IIoTset] Status API response:', res.status, data);
            window._edgeDatasetPresent = !!(data && data.present);
            return window._edgeDatasetPresent;
        } catch (e) {
            console.warn('[Edge-IIoTset] Status API error:', e);
            window._edgeDatasetPresent = false;
            return false;
        }
    }

    function getEdgeProgressElements() {
        return {
            container: document.getElementById('edge-dataset-download-progress'),
            bar: document.getElementById('edge-dataset-download-progress-bar'),
            status: document.getElementById('edge-dataset-download-status'),
        };
    }

    function startEdgeDownloadProgress() {
        const { container, bar, status } = getEdgeProgressElements();
        if (!container || !bar || !status) return;
        edgeDlProgress = 0;
        bar.style.width = '0%';
        bar.setAttribute('aria-valuenow', '0');
        bar.classList.remove('bg-success', 'bg-danger', 'bg-info', 'progress-bar-success', 'progress-bar-danger', 'progress-bar-info');
        bar.classList.add('progress-bar-striped', 'progress-bar-animated');
        try { bar.style.setProperty('background-color', '#0d6efd', 'important'); } catch(e) { bar.style.backgroundColor = '#0d6efd'; }
        bar.style.boxShadow = 'none';
        bar.style.transition = 'width 0.2s ease';
        status.textContent = 'Downloading Edge-IIoTset…';
        container.style.display = 'block';
        container.style.visibility = 'visible';
        status.style.display = 'inline';

        clearInterval(edgeDlTimer);
        edgeDlTimer = setInterval(() => {
            if (edgeDlProgress < 90) {
                edgeDlProgress += Math.max(1, Math.round(Math.random() * 4));
                edgeDlProgress = Math.min(edgeDlProgress, 90);
                bar.style.width = edgeDlProgress + '%';
                bar.setAttribute('aria-valuenow', String(edgeDlProgress));
                bar.textContent = edgeDlProgress + '%';
            }
        }, 200);

        setTimeout(() => {
            if (edgeDlProgress === 0) {
                edgeDlProgress = 5;
                bar.style.width = '5%';
                bar.setAttribute('aria-valuenow', '5');
                bar.textContent = '5%';
            }
        }, 50);
    }

    function completeEdgeDownloadProgress(message) {
        const { container, bar, status } = getEdgeProgressElements();
        if (!container || !bar || !status) return;
        clearInterval(edgeDlTimer);
        edgeDlProgress = 100;
        bar.style.width = '100%';
        bar.setAttribute('aria-valuenow', '100');
        bar.textContent = '100%';
        bar.classList.remove('progress-bar-striped', 'progress-bar-animated', 'bg-danger');
        bar.classList.add('bg-success');
        try { bar.style.setProperty('background-color', '#198754', 'important'); } catch(e) { bar.style.backgroundColor = '#198754'; }
        status.textContent = message || 'Download completed';
    }

    function errorEdgeDownloadProgress(message) {
        const { container, bar, status } = getEdgeProgressElements();
        if (!container || !bar || !status) return;
        clearInterval(edgeDlTimer);
        bar.classList.remove('progress-bar-striped', 'progress-bar-animated', 'bg-success');
        bar.classList.add('bg-danger');
        try { bar.style.setProperty('background-color', '#dc3545', 'important'); } catch(e) { bar.style.backgroundColor = '#dc3545'; }
        status.textContent = message || 'Download failed';
    }

    async function pollEdgeDatasetPresenceWithProgress(timeoutMs = 15 * 60 * 1000, intervalMs = 1000) {
        const start = Date.now();
        const { bar } = getEdgeProgressElements();
        while (Date.now() - start < timeoutMs) {
            if (edgeDlProgress < 99) {
                edgeDlProgress += 1;
                edgeDlProgress = Math.min(edgeDlProgress, 99);
                if (bar) {
                    bar.style.width = edgeDlProgress + '%';
                    bar.setAttribute('aria-valuenow', String(edgeDlProgress));
                    bar.textContent = edgeDlProgress + '%';
                }
            }
            const present = await checkEdgeDatasetPresent(true);
            if (present) {
                completeEdgeDownloadProgress('Download completed');
                return true;
            }
            await new Promise(r => setTimeout(r, intervalMs));
        }
        errorEdgeDownloadProgress('Timeout waiting for dataset');
        return false;
    }

    return {
        initialize,
        updateModelOptions, // Exposed in case other modules need to refresh models
    };
})();

export default DatasetManager;
