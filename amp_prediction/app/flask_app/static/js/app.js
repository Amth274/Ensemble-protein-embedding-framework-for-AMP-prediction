/**
 * Enhanced AMP Prediction Flask App JavaScript
 * Handles UI interactions, API calls, and dynamic content
 */

// Global variables
let loadingModal = null;

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('🧬 AMP Prediction App initialized');

    // Initialize Bootstrap components
    initializeBootstrap();

    // Ensure loading modal is hidden on page load
    setTimeout(() => {
        hideLoadingModal();
    }, 100);

    // Set up global event listeners
    setupGlobalEventListeners();

    // Initialize page-specific functionality
    initializePageSpecific();

    // Check API health
    checkAPIHealth();
}

/**
 * Initialize Bootstrap components
 */
function initializeBootstrap() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Get loading modal instance
    const loadingModalEl = document.getElementById('loadingModal');
    if (loadingModalEl) {
        loadingModal = new bootstrap.Modal(loadingModalEl);
    }
}

/**
 * Set up global event listeners
 */
function setupGlobalEventListeners() {
    // Handle navigation link clicks
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function() {
            // Add active class logic if needed
            console.log('Navigation clicked:', this.textContent.trim());
        });
    });

    // Handle escape key to close modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            hideLoadingModal();
        }
    });

    // Handle form submissions
    document.addEventListener('submit', function(e) {
        const form = e.target;
        if (form.classList.contains('api-form')) {
            e.preventDefault();
            handleFormSubmission(form);
        }
    });
}

/**
 * Initialize page-specific functionality
 */
function initializePageSpecific() {
    const currentPage = getCurrentPage();

    switch(currentPage) {
        case 'predict':
            initializePredictionPage();
            break;
        case 'batch':
            initializeBatchPage();
            break;
        case 'examples':
            initializeExamplesPage();
            break;
        default:
            console.log('No specific initialization for page:', currentPage);
    }
}

/**
 * Get current page name from URL
 */
function getCurrentPage() {
    const path = window.location.pathname;
    if (path.includes('predict')) return 'predict';
    if (path.includes('batch')) return 'batch';
    if (path.includes('examples')) return 'examples';
    if (path.includes('about')) return 'about';
    return 'home';
}

/**
 * Initialize prediction page functionality
 */
function initializePredictionPage() {
    console.log('Initializing prediction page...');

    // Real-time sequence validation
    const sequenceInput = document.getElementById('sequenceInput');
    if (sequenceInput) {
        sequenceInput.addEventListener('input', function() {
            validateSequenceInput(this.value);
        });
    }

    // Threshold slider
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    if (thresholdSlider && thresholdValue) {
        thresholdSlider.addEventListener('input', function() {
            thresholdValue.textContent = parseFloat(this.value).toFixed(2);
        });
    }
}

/**
 * Initialize batch analysis page
 */
function initializeBatchPage() {
    console.log('Initializing batch page...');

    // File upload handling
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }

    // Sample data loading
    const loadSampleBtn = document.getElementById('loadSampleBtn');
    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', loadSampleData);
    }
}

/**
 * Initialize examples page
 */
function initializeExamplesPage() {
    console.log('Initializing examples page...');

    // Example sequence buttons
    document.querySelectorAll('.test-sequence-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const sequence = this.dataset.sequence;
            testExampleSequence(sequence);
        });
    });
}

/**
 * Validate sequence input
 */
function validateSequenceInput(sequence) {
    const cleanSequence = sequence.trim().toUpperCase();
    const validAAs = /^[ACDEFGHIKLMNPQRSTVWY]*$/;

    const feedback = document.getElementById('sequenceFeedback');
    if (!feedback) return;

    if (cleanSequence.length === 0) {
        feedback.innerHTML = '';
        return;
    }

    if (cleanSequence.length < 5) {
        feedback.innerHTML = '<div class="text-warning"><i class="bi bi-exclamation-triangle"></i> Sequence too short (minimum 5 amino acids)</div>';
        return;
    }

    if (cleanSequence.length > 200) {
        feedback.innerHTML = '<div class="text-warning"><i class="bi bi-exclamation-triangle"></i> Sequence too long (maximum 200 amino acids)</div>';
        return;
    }

    if (!validAAs.test(cleanSequence)) {
        feedback.innerHTML = '<div class="text-danger"><i class="bi bi-x-circle"></i> Invalid amino acid characters detected</div>';
        return;
    }

    feedback.innerHTML = '<div class="text-success"><i class="bi bi-check-circle"></i> Valid protein sequence</div>';
}

/**
 * Handle file upload for batch analysis
 */
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
        showAlert('Please upload a CSV file', 'warning');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const csv = e.target.result;
            const sequences = parseCSV(csv);
            displayUploadedSequences(sequences);
        } catch (error) {
            showAlert('Error reading file: ' + error.message, 'danger');
        }
    };
    reader.readAsText(file);
}

/**
 * Parse CSV content
 */
function parseCSV(csv) {
    const lines = csv.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());

    const sequenceColumn = headers.findIndex(h =>
        h.toLowerCase().includes('sequence') || h.toLowerCase().includes('seq')
    );

    if (sequenceColumn === -1) {
        throw new Error('No sequence column found. Please ensure your CSV has a "Sequence" column.');
    }

    const sequences = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        if (values[sequenceColumn] && values[sequenceColumn].trim()) {
            sequences.push(values[sequenceColumn].trim());
        }
    }

    return sequences;
}

/**
 * Display uploaded sequences
 */
function displayUploadedSequences(sequences) {
    const container = document.getElementById('uploadedSequences');
    if (!container) return;

    container.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Uploaded Sequences (${sequences.length})</h6>
            </div>
            <div class="card-body">
                <div class="sequence-list" style="max-height: 300px; overflow-y: auto;">
                    ${sequences.slice(0, 10).map((seq, index) => `
                        <div class="sequence-item">
                            <span class="sequence-number">${index + 1}.</span>
                            <code class="sequence-code">${seq}</code>
                        </div>
                    `).join('')}
                    ${sequences.length > 10 ? `<div class="text-muted">... and ${sequences.length - 10} more sequences</div>` : ''}
                </div>
                <button class="btn btn-primary mt-3" onclick="analyzeBatchSequences(${JSON.stringify(sequences).replace(/"/g, '&quot;')})">
                    <i class="bi bi-search"></i> Analyze All Sequences
                </button>
            </div>
        </div>
    `;
}

/**
 * Load sample data for batch analysis
 */
function loadSampleData() {
    showLoadingModal('Loading sample data...');

    fetch('/api/examples')
        .then(response => response.json())
        .then(data => {
            hideLoadingModal();
            const sequences = data.map(item => item.Sequence);
            displayUploadedSequences(sequences);
            showAlert('Sample data loaded successfully', 'success');
        })
        .catch(error => {
            hideLoadingModal();
            console.error('Error loading sample data:', error);
            showAlert('Failed to load sample data', 'danger');
        });
}

/**
 * Analyze batch sequences
 */
function analyzeBatchSequences(sequences) {
    if (!sequences || sequences.length === 0) {
        showAlert('No sequences to analyze', 'warning');
        return;
    }

    showLoadingModal(`Analyzing ${sequences.length} sequences...`);

    fetch('/api/predict_batch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sequences: sequences })
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingModal();
        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            displayBatchResults(data);
        }
    })
    .catch(error => {
        hideLoadingModal();
        console.error('Error:', error);
        showAlert('Batch analysis failed. Please try again.', 'danger');
    });
}

/**
 * Display batch analysis results
 */
function displayBatchResults(data) {
    const resultsContainer = document.getElementById('batchResults');
    if (!resultsContainer) return;

    const { results, summary } = data;

    // Create summary card
    const summaryHTML = `
        <div class="card mb-4">
            <div class="card-header bg-success text-white">
                <h5 class="mb-0"><i class="bi bi-clipboard-data"></i> Analysis Summary</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-primary">${summary.total_sequences}</h3>
                            <small class="text-muted">Total Sequences</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-success">${summary.predicted_amps}</h3>
                            <small class="text-muted">Predicted AMPs</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-danger">${summary.predicted_non_amps}</h3>
                            <small class="text-muted">Predicted Non-AMPs</small>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="text-center">
                            <h3 class="text-info">${(summary.average_confidence * 100).toFixed(1)}%</h3>
                            <small class="text-muted">Avg Confidence</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Create results table
    const tableHTML = `
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Detailed Results</h6>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Sequence</th>
                                <th>Prediction</th>
                                <th>Confidence</th>
                                <th>Length</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${results.map((result, index) => `
                                <tr>
                                    <td>${index + 1}</td>
                                    <td>
                                        <code class="sequence-display">
                                            ${result.sequence.length > 20 ?
                                                result.sequence.substring(0, 20) + '...' :
                                                result.sequence}
                                        </code>
                                    </td>
                                    <td>
                                        <span class="badge ${result.ensemble.prediction === 1 ? 'bg-success' : 'bg-danger'}">
                                            ${result.ensemble.prediction === 1 ? 'AMP' : 'Non-AMP'}
                                        </span>
                                    </td>
                                    <td>
                                        <div class="progress" style="width: 100px;">
                                            <div class="progress-bar ${result.ensemble.prediction === 1 ? 'bg-success' : 'bg-danger'}"
                                                 style="width: ${result.ensemble.confidence * 100}%">
                                                ${(result.ensemble.confidence * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                    </td>
                                    <td>${result.sequence.length}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div class="text-end mt-3">
                    <button class="btn btn-outline-primary" onclick="downloadBatchResults(${JSON.stringify(results).replace(/"/g, '&quot;')})">
                        <i class="bi bi-download"></i> Download Results
                    </button>
                </div>
            </div>
        </div>
    `;

    resultsContainer.innerHTML = summaryHTML + tableHTML;
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Test example sequence
 */
function testExampleSequence(sequence) {
    // Redirect to prediction page with the sequence
    const url = new URL('/predict', window.location.origin);
    url.searchParams.set('sequence', sequence);
    window.location.href = url.toString();
}

/**
 * Download batch results
 */
function downloadBatchResults(results) {
    const csv = convertToCSV(results);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'amp_predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

/**
 * Convert results to CSV
 */
function convertToCSV(results) {
    const headers = ['Sequence', 'Prediction', 'Confidence', 'Length'];
    const rows = results.map(result => [
        result.sequence,
        result.ensemble.prediction === 1 ? 'AMP' : 'Non-AMP',
        (result.ensemble.confidence * 100).toFixed(2),
        result.sequence.length
    ]);

    return [headers, ...rows].map(row =>
        row.map(cell => `"${cell}"`).join(',')
    ).join('\n');
}

/**
 * Show loading modal
 */
function showLoadingModal(message = 'Processing...') {
    // Update both the header and description
    const modalHeader = document.querySelector('#loadingModal .modal-body h5');
    const modalDesc = document.querySelector('#loadingModal .modal-body p');

    if (modalHeader) {
        modalHeader.textContent = message;
    }
    if (modalDesc) {
        modalDesc.textContent = 'Please wait while we analyze your sequence...';
    }

    console.log('Showing loading modal with message:', message);

    if (loadingModal) {
        loadingModal.show();
    } else {
        console.error('Loading modal not initialized');
        // Fallback: show modal manually
        const modalEl = document.getElementById('loadingModal');
        if (modalEl) {
            modalEl.style.display = 'block';
            modalEl.classList.add('show');
            document.body.classList.add('modal-open');
        }
    }
}

/**
 * Hide loading modal
 */
function hideLoadingModal() {
    console.log('Attempting to hide loading modal');

    // Always try both methods to ensure modal is hidden
    if (loadingModal) {
        loadingModal.hide();
        console.log('Loading modal hidden via Bootstrap');
    }

    // Force hide manually as well to ensure it's gone
    const modalEl = document.getElementById('loadingModal');
    if (modalEl) {
        modalEl.style.display = 'none';
        modalEl.classList.remove('show', 'fade');
        modalEl.setAttribute('aria-hidden', 'true');
        modalEl.removeAttribute('aria-modal');
        modalEl.removeAttribute('role');

        // Remove all backdrops
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.remove();
        });

        // Restore body scroll
        document.body.classList.remove('modal-open');
        document.body.style.paddingRight = '';
        document.body.style.overflow = '';

        console.log('Loading modal force hidden');
    } else {
        console.error('Loading modal element not found');
    }

    // Additional cleanup - remove any lingering modal classes
    setTimeout(() => {
        document.body.classList.remove('modal-open');
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.remove();
        });
    }, 100);
}

/**
 * Show alert
 */
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;

    const alertId = 'alert_' + Date.now();
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            <i class="bi bi-${getAlertIcon(type)}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    alertContainer.insertAdjacentHTML('beforeend', alertHTML);

    // Auto-dismiss after duration
    setTimeout(() => {
        const alertElement = document.getElementById(alertId);
        if (alertElement) {
            const bsAlert = new bootstrap.Alert(alertElement);
            bsAlert.close();
        }
    }, duration);
}

/**
 * Get alert icon based on type
 */
function getAlertIcon(type) {
    const icons = {
        success: 'check-circle',
        danger: 'x-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Check API health
 */
function checkAPIHealth() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            console.log('API Health:', data);
            if (!data.predictor_available) {
                showAlert('Predictor not available. Some features may be limited.', 'warning');
            }
        })
        .catch(error => {
            console.warn('API health check failed:', error);
        });
}

/**
 * Handle form submission
 */
function handleFormSubmission(form) {
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    console.log('Form submitted:', data);
    // Add form-specific handling here
}

/**
 * Utility function to format numbers
 */
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

/**
 * Utility function to debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Copied to clipboard!', 'success', 2000);
    }).catch(() => {
        showAlert('Failed to copy to clipboard', 'danger', 2000);
    });
}

// Export functions for global access
window.AMP = {
    showAlert,
    showLoadingModal,
    hideLoadingModal,
    copyToClipboard,
    testExampleSequence,
    analyzeBatchSequences,
    downloadBatchResults
};