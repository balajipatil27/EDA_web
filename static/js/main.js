// static/js/main.js - Complete and Corrected Version

// Global state
let currentStep = 1;
let activeStep = 1;
let datasetInfo = null;
let originalFile = null;
let processedFile = null;
let targetColumn = null;
let preprocessingSteps = [];
let conversionSteps = [];
let encodingSteps = [];
let missingValueSteps = [];
let outlierSteps = [];
let batchSteps = [];
let columnTypes = null;
let suggestions = null;
let modelResults = null;

// DOM Elements
const stepElements = document.querySelectorAll('.step');
const stepContentElements = document.querySelectorAll('.step-content');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const currentStepElement = document.getElementById('current-step');
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const uploadProgressBar = document.getElementById('upload-progress-bar');
const uploadStatus = document.getElementById('upload-status');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingMessage = document.getElementById('loading-message');
const toast = document.getElementById('toast');
const datasetInfoDiv = document.getElementById('dataset-info');
const targetColumnSelect = document.getElementById('target-column');
const analyzeBtn = document.getElementById('analyze-btn');
const dataQualityMetrics = document.getElementById('data-quality-metrics');
const analysisResults = document.getElementById('analysis-results');
const columnAnalysis = document.getElementById('column-analysis');
const quickActions = document.getElementById('quick-actions');
const preprocessingSuggestions = document.getElementById('preprocessing-suggestions');
const preprocessingControls = document.getElementById('preprocessing-controls');
const applyPreprocessing = document.getElementById('apply-preprocessing');
const clearSteps = document.getElementById('clear-steps');
const preprocessingResults = document.getElementById('preprocessing-results');
const preprocessingReport = document.getElementById('preprocessing-report');
const modelTraining = document.getElementById('model-training');
const trainModelsBtn = document.getElementById('train-models-btn');
const trainingProgress = document.getElementById('training-progress');
const trainingResults = document.getElementById('training-results');
const finalResults = document.getElementById('final-results');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    showStep(1);
});

// Initialize all event listeners
function initializeEventListeners() {
    // Navigation buttons
    prevBtn.addEventListener('click', goToPrevStep);
    nextBtn.addEventListener('click', goToNextStep);

    // Step indicators
    stepElements.forEach(step => {
        step.addEventListener('click', () => {
            const stepNum = parseInt(step.dataset.step);
            if (stepNum <= activeStep) {
                showStep(stepNum);
            }
        });
    });

    // File upload
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Step 2: Analysis
    analyzeBtn.addEventListener('click', analyzeDataset);
    targetColumnSelect.addEventListener('change', function() {
        targetColumn = this.value;
        updateStepState(2);
    });

    // Step 3: Preprocessing
    document.getElementById('quick-clean').addEventListener('click', quickClean);
    document.getElementById('auto-preprocess').addEventListener('click', autoPreprocess);
    document.getElementById('get-suggestions-btn').addEventListener('click', getPreprocessingSuggestions);
    document.getElementById('add-conversion').addEventListener('click', addConversionStep);
    document.getElementById('add-encoding').addEventListener('click', addEncodingStep);
    document.getElementById('quick-fill-all').addEventListener('click', quickFillAll);
    document.getElementById('batch-encode-categorical').addEventListener('click', batchEncodeCategorical);
    document.getElementById('batch-convert-numerical').addEventListener('click', batchConvertNumerical);
    document.getElementById('batch-remove-outliers').addEventListener('click', batchRemoveOutliers);
    clearSteps.addEventListener('click', clearAllSteps);
    applyPreprocessing.addEventListener('click', applyPreprocessingSteps);

    // Step 4: Model Training
    trainModelsBtn.addEventListener('click', trainModels);

    // Step 5: Results
    document.getElementById('download-processed').addEventListener('click', () => downloadFile(processedFile));
    document.getElementById('download-report').addEventListener('click', downloadReport);
    document.getElementById('download-all').addEventListener('click', downloadAll);
    document.getElementById('start-over').addEventListener('click', startOver);
    document.getElementById('back-to-preprocessing').addEventListener('click', () => showStep(3));
    document.getElementById('export-pipeline').addEventListener('click', exportPipeline);

    // Visualization buttons
    document.getElementById('show-accuracy-chart').addEventListener('click', showAccuracyChart);
    document.getElementById('show-improvement-chart').addEventListener('click', showImprovementChart);

    // Modal controls
    document.getElementById('show-help').addEventListener('click', showHelp);
    document.getElementById('show-about').addEventListener('click', showAbout);
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', closeModals);
    });
    
    // Close modals when clicking outside
    window.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal')) {
            closeModals();
        }
    });
}

// Navigation functions
function showStep(stepNumber) {
    // Update current step
    currentStep = stepNumber;
    currentStepElement.textContent = stepNumber;
    
    // Update step indicators
    stepElements.forEach(step => {
        const stepNum = parseInt(step.dataset.step);
        step.classList.remove('active');
        if (stepNum === stepNumber) {
            step.classList.add('active');
        } else if (stepNum < stepNumber) {
            step.classList.add('completed');
        } else {
            step.classList.remove('completed');
        }
    });
    
    // Show corresponding content
    stepContentElements.forEach(content => {
        content.classList.remove('active');
        if (content.id === `step-${stepNumber}`) {
            content.classList.add('active');
        }
    });
    
    // Update navigation buttons
    updateNavigationButtons();
    
    // Load step-specific content
    loadStepContent(stepNumber);
}

function loadStepContent(stepNumber) {
    switch(stepNumber) {
        case 1:
            // Reset for new upload
            break;
        case 2:
            if (datasetInfo) {
                loadDatasetAnalysis();
            }
            break;
        case 3:
            if (datasetInfo) {
                loadPreprocessingControls();
            }
            break;
        case 4:
            if (processedFile) {
                loadModelTraining();
            }
            break;
        case 5:
            if (modelResults) {
                loadFinalResults();
            }
            break;
    }
}

function goToPrevStep() {
    if (currentStep > 1) {
        showStep(currentStep - 1);
    }
}

function goToNextStep() {
    if (currentStep < 5) {
        // Validate current step before proceeding
        if (validateCurrentStep()) {
            showStep(currentStep + 1);
        }
    }
}

function validateCurrentStep() {
    switch(currentStep) {
        case 1:
            if (!datasetInfo) {
                showToast('Please upload a dataset first', 'error');
                return false;
            }
            return true;
        case 2:
            if (!targetColumn) {
                showToast('Please select a target column', 'error');
                return false;
            }
            return true;
        case 3:
            if (preprocessingSteps.length === 0) {
                if (!confirm('No preprocessing steps selected. Continue anyway?')) {
                    return false;
                }
            }
            return true;
        case 4:
            if (!processedFile) {
                showToast('Please apply preprocessing first', 'error');
                return false;
            }
            return true;
        default:
            return true;
    }
}

function updateNavigationButtons() {
    prevBtn.disabled = currentStep === 1;
    
    if (currentStep === 5) {
        nextBtn.disabled = true;
        nextBtn.innerHTML = 'Complete';
    } else {
        nextBtn.disabled = false;
        nextBtn.innerHTML = 'Next <i class="fas fa-arrow-right"></i>';
    }
}

function updateStepState(step) {
    if (step > activeStep) {
        activeStep = step;
    }
    
    // Update step indicators
    stepElements.forEach(stepEl => {
        const stepNum = parseInt(stepEl.dataset.step);
        if (stepNum <= activeStep) {
            stepEl.classList.add('completed');
        }
    });
}

// File upload functions
function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    const maxSize = 500 * 1024 * 1024; // 500MB
    
    // Check file size
    if (file.size > maxSize) {
        showToast('File size exceeds 500MB limit', 'error');
        return;
    }
    
    // Check file type
    const validExtensions = ['.csv', '.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
        showToast('Please upload CSV or Excel files only', 'error');
        return;
    }
    
    // Show file info
    dropArea.innerHTML = `
        <i class="fas fa-file-alt fa-3x"></i>
        <h3>${file.name}</h3>
        <p>${formatFileSize(file.size)}</p>
        <button class="btn btn-primary" id="upload-btn">
            <i class="fas fa-upload"></i> Upload File
        </button>
    `;
    
    document.getElementById('upload-btn').addEventListener('click', () => uploadFile(file));
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    uploadProgress.classList.remove('hidden');
    uploadProgressBar.style.width = '0%';
    uploadStatus.textContent = 'Uploading...';
    
    // Upload to server
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            datasetInfo = data.dataset_info;
            originalFile = data.original_file;
            
            // Update progress
            uploadProgressBar.style.width = '100%';
            uploadStatus.textContent = 'Upload complete!';
            
            // Show success message
            showToast(data.message || 'File uploaded successfully', 'success');
            
            // Update UI
            setTimeout(() => {
                uploadProgress.classList.add('hidden');
                dropArea.innerHTML = `
                    <i class="fas fa-check-circle fa-3x text-success"></i>
                    <h3>Upload Successful!</h3>
                    <p>${datasetInfo.rows.toLocaleString()} rows, ${datasetInfo.columns} columns</p>
                    <button class="btn btn-success" onclick="showStep(2)">
                        <i class="fas fa-arrow-right"></i> Analyze Dataset
                    </button>
                `;
                
                updateStepState(2);
                showStep(2);
            }, 1000);
            
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        uploadProgress.classList.add('hidden');
        dropArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt fa-3x"></i>
            <h3>Drag & Drop Your File Here</h3>
            <p>or click to browse</p>
            <p class="file-types">Supported formats: CSV, Excel (.xlsx, .xls)</p>
            <button class="btn btn-primary" onclick="document.getElementById('file-input').click()">
                <i class="fas fa-folder-open"></i> Browse Files
            </button>
        `;
        showToast(error.message, 'error');
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Step 2: Dataset Analysis
function loadDatasetAnalysis() {
    if (!datasetInfo) return;
    
    // Show dataset info section
    datasetInfoDiv.classList.remove('hidden');
    
    // Populate basic info
    document.getElementById('file-name').textContent = datasetInfo.filename;
    document.getElementById('row-count').textContent = datasetInfo.rows.toLocaleString();
    document.getElementById('col-count').textContent = datasetInfo.columns;
    document.getElementById('dup-count').textContent = datasetInfo.duplicates.toLocaleString();
    
    // Populate target column dropdown
    targetColumnSelect.innerHTML = '<option value="">Select a column</option>';
    datasetInfo.column_names.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        targetColumnSelect.appendChild(option);
    });
    
    // If target column was previously selected, restore it
    if (targetColumn) {
        targetColumnSelect.value = targetColumn;
    }
    
    // Load quick data quality metrics
    loadDataQualityMetrics();
}

function loadDataQualityMetrics() {
    if (!datasetInfo) return;
    
    let missingCount = 0;
    let highMissingCount = 0;
    
    Object.values(datasetInfo.missing_values).forEach(count => {
        if (count > 0) missingCount++;
    });
    
    Object.values(datasetInfo.missing_percentage).forEach(percentage => {
        if (percentage > 50) highMissingCount++;
    });
    
    const qualityScore = calculateDataQualityScore();
    
    dataQualityMetrics.innerHTML = `
        <h3><i class="fas fa-star"></i> Data Quality Metrics</h3>
        <div class="info-cards">
            <div class="info-card">
                <h4><i class="fas fa-exclamation-triangle"></i> Missing Values</h4>
                <p><strong>Columns with missing values:</strong> ${missingCount}/${datasetInfo.columns}</p>
                <p><strong>High missing (>50%):</strong> ${highMissingCount} columns</p>
            </div>
            <div class="info-card">
                <h4><i class="fas fa-clone"></i> Duplicates</h4>
                <p><strong>Duplicate rows:</strong> ${datasetInfo.duplicates.toLocaleString()}</p>
                <p><strong>Percentage:</strong> ${((datasetInfo.duplicates / datasetInfo.rows) * 100).toFixed(2)}%</p>
            </div>
            <div class="info-card">
                <h4><i class="fas fa-chart-line"></i> Quality Score</h4>
                <div class="quality-score">
                    <div class="score-circle" style="--score: ${qualityScore}">
                        <span>${qualityScore}%</span>
                    </div>
                    <p>Overall data quality</p>
                </div>
            </div>
        </div>
    `;
    dataQualityMetrics.classList.remove('hidden');
}

function calculateDataQualityScore() {
    if (!datasetInfo) return 0;
    
    let score = 100;
    
    // Deduct for duplicates
    const duplicatePercentage = (datasetInfo.duplicates / datasetInfo.rows) * 100;
    score -= Math.min(duplicatePercentage * 2, 30);
    
    // Deduct for missing values
    let totalMissingPercentage = 0;
    Object.values(datasetInfo.missing_percentage).forEach(percentage => {
        totalMissingPercentage += percentage;
    });
    const avgMissingPercentage = totalMissingPercentage / datasetInfo.columns;
    score -= Math.min(avgMissingPercentage, 40);
    
    // Deduct for columns with >50% missing
    let highMissingCount = 0;
    Object.values(datasetInfo.missing_percentage).forEach(percentage => {
        if (percentage > 50) highMissingCount++;
    });
    score -= highMissingCount * 10;
    
    return Math.max(Math.round(score), 0);
}

function analyzeDataset() {
    if (!datasetInfo || !targetColumn) {
        showToast('Please select a target column first', 'error');
        return;
    }
    
    showLoading('Analyzing dataset...');
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: originalFile
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            displayAnalysisResults(data.analysis);
            showToast('Analysis completed successfully', 'success');
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Analysis error:', error);
        showToast(error.message, 'error');
    });
}

function displayAnalysisResults(analysis) {
    analysisResults.classList.remove('hidden');
    
    // Display column analysis
    let columnAnalysisHTML = '<h4>Column Analysis</h4><div class="columns-grid">';
    
    analysis.column_analysis.forEach(col => {
        columnAnalysisHTML += `
            <div class="column-card">
                <div class="column-header">
                    <h5>${col.name}</h5>
                    <span class="column-type ${col.type}">${col.type}</span>
                </div>
                <div class="column-details">
                    <p><strong>Data Type:</strong> ${col.dtype}</p>
                    <p><strong>Unique Values:</strong> ${col.unique_values.toLocaleString()}</p>
                    <p><strong>Missing:</strong> ${col.missing_count} (${col.missing_percentage.toFixed(2)}%)</p>
                    
                    ${col.type === 'numerical' ? `
                        <p><strong>Range:</strong> ${col.min !== null ? col.min.toFixed(2) : 'N/A'} - ${col.max !== null ? col.max.toFixed(2) : 'N/A'}</p>
                        <p><strong>Mean:</strong> ${col.mean !== null ? col.mean.toFixed(2) : 'N/A'}</p>
                    ` : ''}
                    
                    ${col.top_values ? `
                        <p><strong>Top Values:</strong></p>
                        <ul class="top-values">
                            ${Object.entries(col.top_values).slice(0, 3).map(([val, count]) => 
                                `<li>${val}: ${count.toLocaleString()}</li>`
                            ).join('')}
                        </ul>
                    ` : ''}
                </div>
            </div>
        `;
    });
    
    columnAnalysisHTML += '</div>';
    columnAnalysis.innerHTML = columnAnalysisHTML;
    
    // Display basic statistics if available
    if (analysis.basic_stats && Object.keys(analysis.basic_stats).length > 0) {
        let basicStatsHTML = '<h4>Basic Statistics</h4><div class="table-responsive"><table class="data-table"><thead><tr><th>Column</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>50%</th><th>75%</th><th>Max</th></tr></thead><tbody>';
        
        for (const [col, stats] of Object.entries(analysis.basic_stats)) {
            basicStatsHTML += `
                <tr>
                    <td>${col}</td>
                    <td>${stats.count ? stats.count.toLocaleString() : 'N/A'}</td>
                    <td>${stats.mean ? stats.mean.toFixed(2) : 'N/A'}</td>
                    <td>${stats.std ? stats.std.toFixed(2) : 'N/A'}</td>
                    <td>${stats.min ? stats.min.toFixed(2) : 'N/A'}</td>
                    <td>${stats['25%'] ? stats['25%'].toFixed(2) : 'N/A'}</td>
                    <td>${stats['50%'] ? stats['50%'].toFixed(2) : 'N/A'}</td>
                    <td>${stats['75%'] ? stats['75%'].toFixed(2) : 'N/A'}</td>
                    <td>${stats.max ? stats.max.toFixed(2) : 'N/A'}</td>
                </tr>
            `;
        }
        
        basicStatsHTML += '</tbody></table></div>';
        columnAnalysis.innerHTML += basicStatsHTML;
    }
    
    // Display correlation matrix if available
    if (analysis.correlation_matrix && analysis.correlation_matrix.length > 0) {
        let correlationHTML = '<h4>Top Correlations</h4><div class="table-responsive"><table class="data-table"><thead><tr><th>Column 1</th><th>Column 2</th><th>Correlation</th><th>Strength</th></tr></thead><tbody>';
        
        // Sort by absolute correlation value
        const sortedCorrelations = [...analysis.correlation_matrix].sort((a, b) => 
            Math.abs(b.correlation) - Math.abs(a.correlation)
        ).slice(0, 20);
        
        sortedCorrelations.forEach(corr => {
            const absCorr = Math.abs(corr.correlation);
            let strength = 'Weak';
            let strengthClass = 'weak';
            
            if (absCorr > 0.7) {
                strength = 'Strong';
                strengthClass = 'strong';
            } else if (absCorr > 0.3) {
                strength = 'Moderate';
                strengthClass = 'moderate';
            }
            
            correlationHTML += `
                <tr>
                    <td>${corr.column1}</td>
                    <td>${corr.column2}</td>
                    <td>${corr.correlation.toFixed(3)}</td>
                    <td><span class="correlation-strength ${strengthClass}">${strength}</span></td>
                </tr>
            `;
        });
        
        correlationHTML += '</tbody></table></div>';
        columnAnalysis.innerHTML += correlationHTML;
    }
    
    // Get column types and suggestions for preprocessing
    getColumnTypes();
}

function getColumnTypes() {
    if (!originalFile) return;
    
    fetch('/get_column_types', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: originalFile
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            columnTypes = data.column_types;
            suggestions = data.suggestions;
            updateStepState(3);
            
            // Enable step 3 in navigation
            stepElements.forEach(step => {
                const stepNum = parseInt(step.dataset.step);
                if (stepNum === 3) {
                    step.classList.add('enabled');
                }
            });
        }
    })
    .catch(error => {
        console.error('Error getting column types:', error);
    });
}

// Step 3: Preprocessing
function loadPreprocessingControls() {
    if (!datasetInfo) return;
    
    quickActions.classList.remove('hidden');
    preprocessingControls.classList.remove('hidden');
    
    // Populate dropdowns
    populatePreprocessingDropdowns();
    
    // Load suggestions if available
    if (suggestions) {
        displayPreprocessingSuggestions();
    }
    
    // Update preprocessing steps summary
    updatePreprocessingStepsSummary();
}

function populatePreprocessingDropdowns() {
    const convertColumnSelect = document.getElementById('convert-column');
    const encodeColumnSelect = document.getElementById('encode-column');
    
    convertColumnSelect.innerHTML = '<option value="">Select column</option>';
    encodeColumnSelect.innerHTML = '<option value="">Select column</option>';
    
    datasetInfo.column_names.forEach(column => {
        // For conversion dropdown
        const option1 = document.createElement('option');
        option1.value = column;
        option1.textContent = column;
        convertColumnSelect.appendChild(option1.cloneNode(true));
        
        // For encoding dropdown
        const option2 = option1.cloneNode(true);
        encodeColumnSelect.appendChild(option2);
    });
}

function displayPreprocessingSuggestions() {
    if (!suggestions) return;
    
    let suggestionsHTML = '<h3><i class="fas fa-lightbulb"></i> Preprocessing Suggestions</h3><div class="suggestions-grid">';
    
    // Encoding suggestions
    if (suggestions.encoding_suggestions && suggestions.encoding_suggestions.length > 0) {
        suggestionsHTML += `
            <div class="suggestion-category">
                <h4><i class="fas fa-code"></i> Encoding Needed</h4>
                <p>The following categorical columns should be encoded:</p>
                <ul class="suggestion-list">
                    ${suggestions.encoding_suggestions.map(suggestion => `
                        <li>
                            <strong>${suggestion.column}</strong> (${suggestion.unique_values} unique values)
                            <span class="suggestion-method">Suggested: ${suggestion.suggested_method} encoding</span>
                            <button class="btn btn-small btn-outline" onclick="addSuggestionEncoding('${suggestion.column}', '${suggestion.suggested_method}')">
                                <i class="fas fa-plus"></i> Add
                            </button>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
    // Outlier suggestions
    if (suggestions.outlier_suggestions && suggestions.outlier_suggestions.length > 0) {
        suggestionsHTML += `
            <div class="suggestion-category">
                <h4><i class="fas fa-filter"></i> Outlier Detection</h4>
                <p>The following columns have potential outliers:</p>
                <ul class="suggestion-list">
                    ${suggestions.outlier_suggestions.map(suggestion => `
                        <li>
                            <strong>${suggestion.column}</strong>
                            <span>${suggestion.outlier_count} outliers (${suggestion.outlier_percentage.toFixed(2)}%)</span>
                            <button class="btn btn-small btn-outline" onclick="addOutlierRemoval('${suggestion.column}')">
                                <i class="fas fa-plus"></i> Add Removal
                            </button>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
    // Data type conversion suggestions
    if (suggestions.dtype_conversion_suggestions && suggestions.dtype_conversion_suggestions.length > 0) {
        suggestionsHTML += `
            <div class="suggestion-category">
                <h4><i class="fas fa-exchange-alt"></i> Data Type Conversion</h4>
                <p>The following columns might need type conversion:</p>
                <ul class="suggestion-list">
                    ${suggestions.dtype_conversion_suggestions.map(suggestion => `
                        <li>
                            <strong>${suggestion.column}</strong>
                            <span>Current: ${suggestion.current_dtype}, Suggested: ${suggestion.suggested_dtype}</span>
                            <button class="btn btn-small btn-outline" onclick="addDataTypeConversion('${suggestion.column}', '${suggestion.suggested_dtype}')">
                                <i class="fas fa-plus"></i> Add
                            </button>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
    suggestionsHTML += '</div>';
    preprocessingSuggestions.innerHTML = suggestionsHTML;
    preprocessingSuggestions.classList.remove('hidden');
}

function addSuggestionEncoding(column, method) {
    document.getElementById('encode-column').value = column;
    document.getElementById('encode-method').value = method;
    addEncodingStep();
}

function addDataTypeConversion(column, dtype) {
    document.getElementById('convert-column').value = column;
    document.getElementById('convert-type').value = dtype;
    addConversionStep();
}

function addOutlierRemoval(column) {
    addPreprocessingStep({
        type: 'remove_outliers',
        column: column
    });
    
    updateOutlierList();
    updatePreprocessingStepsSummary();
}

function addConversionStep() {
    const column = document.getElementById('convert-column').value;
    const dtype = document.getElementById('convert-type').value;
    
    if (!column) {
        showToast('Please select a column', 'error');
        return;
    }
    
    addPreprocessingStep({
        type: 'change_dtype',
        column: column,
        dtype: dtype
    });
    
    updateConversionList();
    updatePreprocessingStepsSummary();
    
    // Clear selection
    document.getElementById('convert-column').value = '';
}

function addEncodingStep() {
    const column = document.getElementById('encode-column').value;
    const method = document.getElementById('encode-method').value;
    
    if (!column) {
        showToast('Please select a column', 'error');
        return;
    }
    
    addPreprocessingStep({
        type: 'encoding',
        column: column,
        method: method
    });
    
    updateEncodingList();
    updatePreprocessingStepsSummary();
    
    // Clear selection
    document.getElementById('encode-column').value = '';
}

function quickFillAll() {
    const method = document.getElementById('quick-fill-method').value;
    
    if (!columnTypes || !columnTypes.numerical) return;
    
    columnTypes.numerical.forEach(column => {
        addPreprocessingStep({
            type: 'handle_missing',
            column: column,
            method: method
        });
    });
    
    updateMissingValueList();
    updatePreprocessingStepsSummary();
    
    showToast(`Added ${method} filling for all numeric columns`, 'success');
}

function batchEncodeCategorical() {
    const method = document.getElementById('batch-encode-method').value;
    
    if (!columnTypes || !columnTypes.categorical) return;
    
    addPreprocessingStep({
        type: 'batch_encoding',
        columns: columnTypes.categorical,
        method: method
    });
    
    updateBatchStepsList();
    updatePreprocessingStepsSummary();
    
    showToast(`Added ${method} encoding for all categorical columns`, 'success');
}

function batchConvertNumerical() {
    if (!columnTypes || !columnTypes.numerical) return;
    
    // This is for converting string columns that should be numeric
    // We need to identify which categorical columns might be numeric
    let columnsToConvert = [];
    
    if (suggestions && suggestions.dtype_conversion_suggestions) {
        columnsToConvert = suggestions.dtype_conversion_suggestions
            .filter(s => s.suggested_dtype === 'numeric')
            .map(s => s.column);
    }
    
    if (columnsToConvert.length > 0) {
        addPreprocessingStep({
            type: 'batch_dtype_conversion',
            columns: columnsToConvert,
            dtype: 'numeric'
        });
        
        updateBatchStepsList();
        updatePreprocessingStepsSummary();
        
        showToast(`Added conversion to numeric for ${columnsToConvert.length} columns`, 'success');
    } else {
        showToast('No columns identified for conversion to numeric', 'info');
    }
}

function batchRemoveOutliers() {
    if (!columnTypes || !columnTypes.numerical) return;
    
    columnTypes.numerical.forEach(column => {
        addPreprocessingStep({
            type: 'remove_outliers',
            column: column
        });
    });
    
    updateOutlierList();
    updatePreprocessingStepsSummary();
    
    showToast(`Added outlier removal for all numerical columns`, 'success');
}

function addPreprocessingStep(step) {
    preprocessingSteps.push(step);
    
    // Categorize steps for display
    if (step.type === 'change_dtype') {
        conversionSteps.push(step);
    } else if (step.type === 'encoding') {
        encodingSteps.push(step);
    } else if (step.type === 'handle_missing') {
        missingValueSteps.push(step);
    } else if (step.type === 'remove_outliers') {
        outlierSteps.push(step);
    } else if (step.type.includes('batch')) {
        batchSteps.push(step);
    }
}

function updateConversionList() {
    const listDiv = document.getElementById('conversion-list');
    listDiv.innerHTML = '';
    
    if (conversionSteps.length === 0) {
        listDiv.innerHTML = '<p class="no-steps">No conversion steps added</p>';
        return;
    }
    
    conversionSteps.forEach((step, index) => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'preprocessing-step';
        stepDiv.innerHTML = `
            <span><i class="fas fa-exchange-alt"></i> Convert "${step.column}" to ${step.dtype}</span>
            <button class="btn btn-small btn-danger" onclick="removePreprocessingStep('conversion', ${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        listDiv.appendChild(stepDiv);
    });
}

function updateEncodingList() {
    const listDiv = document.getElementById('encoding-list');
    listDiv.innerHTML = '';
    
    if (encodingSteps.length === 0) {
        listDiv.innerHTML = '<p class="no-steps">No encoding steps added</p>';
        return;
    }
    
    encodingSteps.forEach((step, index) => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'preprocessing-step';
        stepDiv.innerHTML = `
            <span><i class="fas fa-code"></i> ${step.method} encoding for "${step.column}"</span>
            <button class="btn btn-small btn-danger" onclick="removePreprocessingStep('encoding', ${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        listDiv.appendChild(stepDiv);
    });
}

function updateMissingValueList() {
    const listDiv = document.getElementById('missing-handling-list');
    listDiv.innerHTML = '';
    
    if (missingValueSteps.length === 0) {
        listDiv.innerHTML = '<p class="no-steps">No missing value handling steps added</p>';
        return;
    }
    
    // Group by method
    const groupedByMethod = {};
    missingValueSteps.forEach(step => {
        if (!groupedByMethod[step.method]) {
            groupedByMethod[step.method] = [];
        }
        groupedByMethod[step.method].push(step.column);
    });
    
    for (const [method, columns] of Object.entries(groupedByMethod)) {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'preprocessing-step';
        stepDiv.innerHTML = `
            <span><i class="fas fa-tint"></i> Fill missing with ${method} (${columns.length} columns)</span>
            <button class="btn btn-small btn-danger" onclick="removeMissingValueSteps('${method}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        listDiv.appendChild(stepDiv);
    }
}

function updateOutlierList() {
    const listDiv = document.getElementById('outlier-list');
    listDiv.innerHTML = '';
    
    if (outlierSteps.length === 0) {
        listDiv.innerHTML = '<p class="no-steps">No outlier removal steps added</p>';
        return;
    }
    
    if (outlierSteps.length > 10) {
        // Show summary for many columns
        const stepDiv = document.createElement('div');
        stepDiv.className = 'preprocessing-step';
        stepDiv.innerHTML = `
            <span><i class="fas fa-filter"></i> Remove outliers from ${outlierSteps.length} numerical columns</span>
            <button class="btn btn-small btn-danger" onclick="removeAllOutlierSteps()">
                <i class="fas fa-times"></i>
            </button>
        `;
        listDiv.appendChild(stepDiv);
    } else {
        // Show individual steps
        outlierSteps.forEach((step, index) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'preprocessing-step';
            stepDiv.innerHTML = `
                <span><i class="fas fa-filter"></i> Remove outliers from "${step.column}"</span>
                <button class="btn btn-small btn-danger" onclick="removePreprocessingStep('outlier', ${index})">
                    <i class="fas fa-times"></i>
                </button>
            `;
            listDiv.appendChild(stepDiv);
        });
    }
}

function updateBatchStepsList() {
    const listDiv = document.getElementById('batch-steps-list');
    listDiv.innerHTML = '';
    
    if (batchSteps.length === 0) {
        return;
    }
    
    batchSteps.forEach((step, index) => {
        const stepDiv = document.createElement('div');
        stepDiv.className = 'preprocessing-step';
        
        let description = '';
        if (step.type === 'batch_encoding') {
            description = `Batch ${step.method} encoding for ${step.columns.length} columns`;
        } else if (step.type === 'batch_dtype_conversion') {
            description = `Batch conversion to ${step.dtype} for ${step.columns.length} columns`;
        }
        
        stepDiv.innerHTML = `
            <span><i class="fas fa-layer-group"></i> ${description}</span>
            <button class="btn btn-small btn-danger" onclick="removePreprocessingStep('batch', ${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        listDiv.appendChild(stepDiv);
    });
}

function removePreprocessingStep(category, index) {
    let step;
    
    switch(category) {
        case 'conversion':
            step = conversionSteps.splice(index, 1)[0];
            break;
        case 'encoding':
            step = encodingSteps.splice(index, 1)[0];
            break;
        case 'outlier':
            step = outlierSteps.splice(index, 1)[0];
            break;
        case 'batch':
            step = batchSteps.splice(index, 1)[0];
            break;
    }
    
    // Remove from main steps array
    const stepIndex = preprocessingSteps.findIndex(s => 
        s.type === step.type && 
        s.column === step.column && 
        (s.method === step.method || s.dtype === step.dtype)
    );
    
    if (stepIndex > -1) {
        preprocessingSteps.splice(stepIndex, 1);
    }
    
    // Update UI
    updateConversionList();
    updateEncodingList();
    updateMissingValueList();
    updateOutlierList();
    updateBatchStepsList();
    updatePreprocessingStepsSummary();
}

function removeMissingValueSteps(method) {
    // Remove all steps with this method
    const stepsToRemove = missingValueSteps.filter(step => step.method === method);
    stepsToRemove.forEach(step => {
        const index = preprocessingSteps.findIndex(s => 
            s.type === step.type && 
            s.column === step.column && 
            s.method === step.method
        );
        if (index > -1) {
            preprocessingSteps.splice(index, 1);
        }
    });
    
    // Remove from missingValueSteps array
    missingValueSteps = missingValueSteps.filter(step => step.method !== method);
    
    // Update UI
    updateMissingValueList();
    updatePreprocessingStepsSummary();
}

function removeAllOutlierSteps() {
    // Remove all outlier steps
    outlierSteps.forEach(step => {
        const index = preprocessingSteps.findIndex(s => 
            s.type === step.type && 
            s.column === step.column
        );
        if (index > -1) {
            preprocessingSteps.splice(index, 1);
        }
    });
    
    outlierSteps = [];
    
    // Update UI
    updateOutlierList();
    updatePreprocessingStepsSummary();
}

function updatePreprocessingStepsSummary() {
    const summaryDiv = document.getElementById('preprocessing-steps-summary');
    
    if (preprocessingSteps.length === 0) {
        summaryDiv.innerHTML = '<p>No steps added yet. Add steps from above sections.</p>';
        return;
    }
    
    let summaryHTML = `
        <div class="steps-summary">
            <div class="summary-header">
                <h5>Total Steps: ${preprocessingSteps.length}</h5>
                <span class="steps-count">${preprocessingSteps.length} steps</span>
            </div>
            <div class="steps-breakdown">
    `;
    
    const stepCounts = {
        'Data Cleaning': preprocessingSteps.filter(s => 
            s.type === 'drop_high_missing' || s.type === 'remove_duplicates'
        ).length,
        'Type Conversion': conversionSteps.length + 
            (batchSteps.filter(s => s.type === 'batch_dtype_conversion').length > 0 ? 1 : 0),
        'Encoding': encodingSteps.length + 
            (batchSteps.filter(s => s.type === 'batch_encoding').length > 0 ? 1 : 0),
        'Missing Values': missingValueSteps.length,
        'Outlier Removal': outlierSteps.length
    };
    
    Object.entries(stepCounts).forEach(([category, count]) => {
        if (count > 0) {
            summaryHTML += `
                <div class="step-category">
                    <span class="category-name">${category}</span>
                    <span class="category-count">${count}</span>
                </div>
            `;
        }
    });
    
    summaryHTML += `
            </div>
            <div class="steps-preview">
                <p><strong>First few steps:</strong></p>
                <ul class="steps-list">
    `;
    
    preprocessingSteps.slice(0, 3).forEach(step => {
        let stepDescription = '';
        switch(step.type) {
            case 'drop_high_missing':
                stepDescription = `Drop columns with >${step.threshold || 50}% missing values`;
                break;
            case 'remove_duplicates':
                stepDescription = 'Remove duplicate rows';
                break;
            case 'change_dtype':
                stepDescription = `Convert "${step.column}" to ${step.dtype}`;
                break;
            case 'encoding':
                stepDescription = `Apply ${step.method} encoding to "${step.column}"`;
                break;
            case 'handle_missing':
                stepDescription = `Fill missing values in "${step.column}" with ${step.method}`;
                break;
            case 'remove_outliers':
                stepDescription = `Remove outliers from "${step.column}"`;
                break;
            case 'batch_encoding':
                stepDescription = `Batch ${step.method} encoding (${step.columns.length} columns)`;
                break;
            case 'batch_dtype_conversion':
                stepDescription = `Batch conversion to ${step.dtype} (${step.columns.length} columns)`;
                break;
        }
        
        summaryHTML += `<li>${stepDescription}</li>`;
    });
    
    if (preprocessingSteps.length > 3) {
        summaryHTML += `<li>... and ${preprocessingSteps.length - 3} more</li>`;
    }
    
    summaryHTML += `
                </ul>
            </div>
        </div>
    `;
    
    summaryDiv.innerHTML = summaryHTML;
}

function clearAllSteps() {
    if (preprocessingSteps.length === 0) return;
    
    if (!confirm('Are you sure you want to clear all preprocessing steps?')) {
        return;
    }
    
    preprocessingSteps = [];
    conversionSteps = [];
    encodingSteps = [];
    missingValueSteps = [];
    outlierSteps = [];
    batchSteps = [];
    
    updateConversionList();
    updateEncodingList();
    updateMissingValueList();
    updateOutlierList();
    updateBatchStepsList();
    updatePreprocessingStepsSummary();
    
    showToast('All preprocessing steps cleared', 'success');
}

function quickClean() {
    // Add basic cleaning steps
    const quickSteps = [
        { type: 'drop_high_missing', threshold: 50 },
        { type: 'remove_duplicates' }
    ];
    
    quickSteps.forEach(step => {
        if (!preprocessingSteps.some(s => s.type === step.type)) {
            addPreprocessingStep(step);
        }
    });
    
    // Add missing value handling for numerical columns
    if (columnTypes && columnTypes.numerical) {
        columnTypes.numerical.forEach(column => {
            addPreprocessingStep({
                type: 'handle_missing',
                column: column,
                method: 'median'
            });
        });
    }
    
    updatePreprocessingStepsSummary();
    showToast('Added quick cleaning steps', 'success');
}

function autoPreprocess() {
    if (!suggestions) {
        showToast('No suggestions available. Please analyze data first.', 'error');
        return;
    }
    
    // Clear existing steps
    preprocessingSteps = [];
    conversionSteps = [];
    encodingSteps = [];
    missingValueSteps = [];
    outlierSteps = [];
    batchSteps = [];
    
    // Add basic cleaning
    addPreprocessingStep({ type: 'drop_high_missing', threshold: 50 });
    addPreprocessingStep({ type: 'remove_duplicates' });
    
    // Add encoding for categorical columns
    if (suggestions.encoding_suggestions && suggestions.encoding_suggestions.length > 0) {
        suggestions.encoding_suggestions.forEach(suggestion => {
            addPreprocessingStep({
                type: 'encoding',
                column: suggestion.column,
                method: suggestion.suggested_method
            });
        });
    }
    
    // Add missing value handling for numerical columns
    if (columnTypes && columnTypes.numerical) {
        columnTypes.numerical.forEach(column => {
            addPreprocessingStep({
                type: 'handle_missing',
                column: column,
                method: 'median'
            });
        });
    }
    
    // Add outlier removal for columns with outliers
    if (suggestions.outlier_suggestions && suggestions.outlier_suggestions.length > 0) {
        suggestions.outlier_suggestions.forEach(suggestion => {
            if (suggestion.outlier_count > 0) {
                addPreprocessingStep({
                    type: 'remove_outliers',
                    column: suggestion.column
                });
            }
        });
    }
    
    // Add data type conversions
    if (suggestions.dtype_conversion_suggestions && suggestions.dtype_conversion_suggestions.length > 0) {
        suggestions.dtype_conversion_suggestions.forEach(suggestion => {
            addPreprocessingStep({
                type: 'change_dtype',
                column: suggestion.column,
                dtype: suggestion.suggested_dtype
            });
        });
    }
    
    updatePreprocessingStepsSummary();
    showToast('Auto-preprocessing steps added', 'success');
}

function getPreprocessingSuggestions() {
    if (!datasetInfo) {
        showToast('Please upload and analyze a dataset first', 'error');
        return;
    }
    
    if (!suggestions) {
        getColumnTypes();
    } else {
        displayPreprocessingSuggestions();
    }
}

function applyPreprocessingSteps() {
    if (!datasetInfo || !targetColumn) {
        showToast('Please analyze dataset and select target column first', 'error');
        return;
    }
    
    if (preprocessingSteps.length === 0) {
        if (!confirm('No preprocessing steps selected. Apply basic cleaning only?')) {
            return;
        }
        quickClean();
    }
    
    showLoading('Applying preprocessing steps...');
    
    // Add checkbox-based steps
    if (document.getElementById('drop-high-missing').checked) {
        preprocessingSteps.push({ type: 'drop_high_missing', threshold: 50 });
    }
    
    if (document.getElementById('remove-duplicates').checked) {
        preprocessingSteps.push({ type: 'remove_duplicates' });
    }
    
    fetch('/preprocess', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: originalFile,
            steps: preprocessingSteps,
            target_column: targetColumn
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        
        if (data.success) {
            processedFile = data.processed_file;
            displayPreprocessingResults(data);
            showToast(data.message, 'success');
            updateStepState(4);
        } else {
            throw new Error(data.error || 'Preprocessing failed');
        }
    })
    .catch(error => {
        hideLoading();
        console.error('Preprocessing error:', error);
        showToast(error.message, 'error');
    });
}

function displayPreprocessingResults(data) {
    preprocessingResults.classList.remove('hidden');
    
    // Display report
    let reportHTML = '<div class="report-section"><h4>Processing Steps</h4><ul class="report-list">';
    
    data.preprocessing_report.forEach(step => {
        reportHTML += `
            <li>
                <i class="fas fa-check-circle text-success"></i>
                <div>
                    <strong>${step.step}</strong>
                    <p class="step-details">${step.details}</p>
                </div>
            </li>
        `;
    });
    
    reportHTML += '</ul></div>';
    preprocessingReport.innerHTML = reportHTML;
    
    // Update processed info
    document.getElementById('processed-rows').textContent = data.processed_info.rows.toLocaleString();
    document.getElementById('processed-cols').textContent = data.processed_info.columns;
    document.getElementById('processed-missing').textContent = data.processed_info.missing_values;
    
    // Update performance metrics
    const metrics = data.processed_info.performance_metrics;
    document.getElementById('rows-removed').textContent = metrics.rows_removed || 0;
    document.getElementById('cols-removed').textContent = metrics.columns_removed || 0;
    document.getElementById('outliers-removed').textContent = metrics.outliers_removed || 0;
    
    // Enable step 4
    stepElements.forEach(step => {
        const stepNum = parseInt(step.dataset.step);
        if (stepNum === 4) {
            step.classList.add('enabled');
        }
    });
}

// Step 4: Model Training
function loadModelTraining() {
    if (!processedFile) return;
    
    modelTraining.classList.remove('hidden');
}

function trainModels() {
    if (!originalFile || !processedFile || !targetColumn) {
        showToast('Missing required information for model training', 'error');
        return;
    }
    
    // Get selected models
    const selectedModels = [];
    document.querySelectorAll('.model-option input[type="checkbox"]:checked').forEach(checkbox => {
        selectedModels.push(checkbox.value);
    });
    
    if (selectedModels.length === 0) {
        showToast('Please select at least one model to train', 'error');
        return;
    }
    
    // Show training progress
    trainingProgress.classList.remove('hidden');
    trainingResults.classList.add('hidden');
    
    const trainingProgressBar = document.getElementById('training-progress-bar');
    const trainingStatus = document.getElementById('training-status');
    const modelProgress = document.getElementById('model-progress');
    
    trainingProgressBar.style.width = '0%';
    trainingStatus.textContent = 'Initializing model training...';
    modelProgress.innerHTML = '';
    
    fetch('/train_models', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            original_file: originalFile,
            processed_file: processedFile,
            target_column: targetColumn,
            models: selectedModels
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            modelResults = data;
            displayTrainingResults(data);
            updateStepState(5);
            showToast('Model training completed successfully', 'success');
        } else {
            throw new Error(data.error || 'Model training failed');
        }
    })
    .catch(error => {
        console.error('Training error:', error);
        showToast(error.message, 'error');
        trainingStatus.textContent = 'Training failed: ' + error.message;
    });
}

function displayTrainingResults(data) {
    trainingProgress.classList.add('hidden');
    trainingResults.classList.remove('hidden');
    
    // Display summary
    const summaryDiv = document.getElementById('results-summary');
    summaryDiv.innerHTML = `
        <div class="summary-content">
            <h4><i class="fas fa-chart-line"></i> Training Summary</h4>
            <div class="summary-stats">
                <div class="stat">
                    <span class="stat-label">Problem Type</span>
                    <span class="stat-value">${data.problem_type}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Models Trained</span>
                    <span class="stat-value">${data.results.length}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Best Model</span>
                    <span class="stat-value">${data.summary.best_model || 'N/A'}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Improvement</span>
                    <span class="stat-value ${data.summary.average_improvement > 0 ? 'positive' : 'negative'}">
                        ${data.summary.average_improvement_percent ? data.summary.average_improvement_percent.toFixed(2) + '%' : 'N/A'}
                    </span>
                </div>
            </div>
        </div>
    `;
    
    // Display results table
    const tableContainer = document.getElementById('results-table-container');
    let tableHTML = `
        <h4><i class="fas fa-table"></i> Model Performance Comparison</h4>
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Original Accuracy</th>
                        <th>Processed Accuracy</th>
                        <th>Improvement</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    data.results.forEach(result => {
        const originalAcc = result.original_accuracy_percent !== null ? 
            result.original_accuracy_percent.toFixed(2) + '%' : 'N/A';
        const processedAcc = result.processed_accuracy_percent !== null ? 
            result.processed_accuracy_percent.toFixed(2) + '%' : 'N/A';
        const improvement = result.improvement_percent !== null ? 
            result.improvement_percent.toFixed(2) + '%' : 'N/A';
        
        let improvementClass = '';
        if (result.improvement_percent > 0) {
            improvementClass = 'positive';
        } else if (result.improvement_percent < 0) {
            improvementClass = 'negative';
        }
        
        tableHTML += `
            <tr>
                <td><strong>${formatModelName(result.model)}</strong></td>
                <td>${originalAcc}</td>
                <td>${processedAcc}</td>
                <td class="${improvementClass}">${improvement}</td>
                <td>${result.error ? '<span class="error">Error</span>' : '<span class="success">Success</span>'}</td>
            </tr>
        `;
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    tableContainer.innerHTML = tableHTML;
}

function formatModelName(model) {
    const names = {
        'linear_regression': 'Linear Regression',
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'decision_tree': 'Decision Tree',
        'svm': 'Support Vector Machine',
        'kmeans': 'K-Means Clustering'
    };
    return names[model] || model;
}

function showAccuracyChart() {
    if (!modelResults) return;
    
    const vizContainer = document.getElementById('visualization-container');
    
    // Simple bar chart using HTML/CSS
    let chartHTML = `
        <div class="chart-container">
            <h5>Model Accuracy Comparison</h5>
            <div class="chart-bars">
    `;
    
    modelResults.results.forEach(result => {
        if (result.original_accuracy_percent !== null && result.processed_accuracy_percent !== null) {
            const originalHeight = Math.min(result.original_accuracy_percent, 100);
            const processedHeight = Math.min(result.processed_accuracy_percent, 100);
            
            chartHTML += `
                <div class="chart-bar-group">
                    <div class="chart-bar-label">${formatModelName(result.model).substring(0, 12)}...</div>
                    <div class="chart-bars-container">
                        <div class="chart-bar original" style="height: ${originalHeight}%" 
                             title="Original: ${result.original_accuracy_percent.toFixed(2)}%"></div>
                        <div class="chart-bar processed" style="height: ${processedHeight}%" 
                             title="Processed: ${result.processed_accuracy_percent.toFixed(2)}%"></div>
                    </div>
                </div>
            `;
        }
    });
    
    chartHTML += `
            </div>
            <div class="chart-legend">
                <span class="legend-item"><div class="legend-color original"></div> Original</span>
                <span class="legend-item"><div class="legend-color processed"></div> Processed</span>
            </div>
        </div>
    `;
    
    vizContainer.innerHTML = chartHTML;
    vizContainer.classList.remove('hidden');
}

function showImprovementChart() {
    if (!modelResults) return;
    
    const vizContainer = document.getElementById('visualization-container');
    
    let chartHTML = `
        <div class="chart-container">
            <h5>Improvement After Preprocessing</h5>
            <div class="improvement-chart">
    `;
    
    const validResults = modelResults.results.filter(r => r.improvement_percent !== null);
    
    validResults.forEach(result => {
        const improvement = result.improvement_percent;
        const barClass = improvement > 0 ? 'positive' : 'negative';
        const barWidth = Math.min(Math.abs(improvement) * 2, 100);
        
        chartHTML += `
            <div class="improvement-row">
                <div class="model-name">${formatModelName(result.model)}</div>
                <div class="improvement-bar-container">
                    <div class="improvement-bar ${barClass}" style="width: ${barWidth}%">
                        <span class="improvement-value">${improvement.toFixed(2)}%</span>
                    </div>
                </div>
            </div>
        `;
    });
    
    chartHTML += `
            </div>
            <div class="chart-note">
                <p>Shows percentage improvement in accuracy after preprocessing</p>
            </div>
        </div>
    `;
    
    vizContainer.innerHTML = chartHTML;
    vizContainer.classList.remove('hidden');
}

// Step 5: Results & Download
function loadFinalResults() {
    if (!modelResults) return;
    
    finalResults.classList.remove('hidden');
    
    // Update final summary
    const finalSummary = document.getElementById('final-summary');
    finalSummary.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item">
                <i class="fas fa-database"></i>
                <div>
                    <h4>Original Dataset</h4>
                    <p>${modelResults.original_shape[0]} rows × ${modelResults.original_shape[1]} columns</p>
                </div>
            </div>
            <div class="summary-item">
                <i class="fas fa-cogs"></i>
                <div>
                    <h4>Processed Dataset</h4>
                    <p>${modelResults.processed_shape[0]} rows × ${modelResults.processed_shape[1]} columns</p>
                </div>
            </div>
            <div class="summary-item">
                <i class="fas fa-robot"></i>
                <div>
                    <h4>Models Trained</h4>
                    <p>${modelResults.results.length} models compared</p>
                </div>
            </div>
            <div class="summary-item">
                <i class="fas fa-trophy"></i>
                <div>
                    <h4>Best Performance</h4>
                    <p>${modelResults.summary.best_model || 'N/A'}: ${modelResults.summary.best_accuracy_percent ? modelResults.summary.best_accuracy_percent.toFixed(2) + '%' : 'N/A'}</p>
                </div>
            </div>
        </div>
    `;
    
    // Update insights
    document.getElementById('best-model-name').textContent = modelResults.summary.best_model || '-';
    document.getElementById('accuracy-improvement').textContent = 
        modelResults.summary.average_improvement_percent ? 
        modelResults.summary.average_improvement_percent.toFixed(2) + '%' : '-';
    
    const qualityScore = calculateDataQualityScore();
    document.getElementById('data-quality-score').textContent = qualityScore + '%';
    
    // Calculate processing time (estimate)
    const processingTime = Math.round(preprocessingSteps.length * 0.5 + modelResults.results.length * 2);
    document.getElementById('processing-time').textContent = processingTime + ' seconds';
}

function downloadFile(filename) {
    if (!filename) {
        showToast('No file available for download', 'error');
        return;
    }
    
    window.open(`/download/${filename}`, '_blank');
}

function downloadReport() {
    if (!modelResults) {
        showToast('No results available for report', 'error');
        return;
    }
    
    // Create a simple report
    const report = {
        dataset_info: datasetInfo,
        preprocessing_steps: preprocessingSteps,
        model_results: modelResults,
        timestamp: new Date().toISOString()
    };
    
    const reportBlob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const reportUrl = URL.createObjectURL(reportBlob);
    
    const a = document.createElement('a');
    a.href = reportUrl;
    a.download = 'ml_preprocessing_report.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(reportUrl);
    showToast('Report downloaded successfully', 'success');
}

function downloadAll() {
    if (!processedFile) {
        showToast('No processed data available', 'error');
        return;
    }
    
    // Download processed file
    downloadFile(processedFile);
    
    // Download report
    setTimeout(() => {
        downloadReport();
    }, 1000);
    
    showToast('Starting download of all files...', 'success');
}

function startOver() {
    if (confirm('Are you sure you want to start over? All current data will be lost.')) {
        // Reset all state
        currentStep = 1;
        activeStep = 1;
        datasetInfo = null;
        originalFile = null;
        processedFile = null;
        targetColumn = null;
        preprocessingSteps = [];
        conversionSteps = [];
        encodingSteps = [];
        missingValueSteps = [];
        outlierSteps = [];
        batchSteps = [];
        columnTypes = null;
        suggestions = null;
        modelResults = null;
        
        // Reset UI
        stepElements.forEach(step => {
            step.classList.remove('active', 'completed', 'enabled');
        });
        
        // Reset content areas
        document.querySelectorAll('.hidden').forEach(el => {
            el.classList.add('hidden');
        });
        
        // Reset file upload area
        dropArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt fa-3x"></i>
            <h3>Drag & Drop Your File Here</h3>
            <p>or click to browse</p>
            <p class="file-types">Supported formats: CSV, Excel (.xlsx, .xls)</p>
            <button class="btn btn-primary" onclick="document.getElementById('file-input').click()">
                <i class="fas fa-folder-open"></i> Browse Files
            </button>
        `;
        
        // Reset target column select
        targetColumnSelect.innerHTML = '<option value="">Select a column</option>';
        
        // Go back to step 1
        showStep(1);
        
        showToast('Started new analysis', 'success');
    }
}

function exportPipeline() {
    if (!preprocessingSteps || preprocessingSteps.length === 0) {
        showToast('No preprocessing steps to export', 'error');
        return;
    }
    
    let pythonCode = `# ML Preprocessing Pipeline
# Generated on ${new Date().toISOString()}
# Dataset: ${datasetInfo ? datasetInfo.filename : 'unknown'}
# Target column: ${targetColumn || 'not specified'}

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_pipeline(df):
    """
    Apply preprocessing steps to the dataframe.
    Returns: Processed dataframe
    """
    df_processed = df.copy()
    
    # Preprocessing steps
`;
    
    preprocessingSteps.forEach(step => {
        pythonCode += `\n    # Step: ${step.type}\n`;
        
        switch(step.type) {
            case 'drop_high_missing':
                pythonCode += `    # Drop columns with >${step.threshold || 50}% missing values\n`;
                pythonCode += `    missing_percentage = (df_processed.isnull().sum() / len(df_processed)) * 100\n`;
                pythonCode += `    columns_to_drop = missing_percentage[missing_percentage > ${step.threshold || 50}].index.tolist()\n`;
                pythonCode += `    df_processed.drop(columns=columns_to_drop, inplace=True)\n`;
                break;
                
            case 'remove_duplicates':
                pythonCode += `    # Remove duplicate rows\n`;
                pythonCode += `    df_processed.drop_duplicates(inplace=True)\n`;
                break;
                
            case 'change_dtype':
                pythonCode += `    # Change data type of column '${step.column}' to ${step.dtype}\n`;
                if (step.dtype === 'numeric') {
                    pythonCode += `    df_processed['${step.column}'] = pd.to_numeric(df_processed['${step.column}'], errors='coerce')\n`;
                } else if (step.dtype === 'datetime') {
                    pythonCode += `    df_processed['${step.column}'] = pd.to_datetime(df_processed['${step.column}'], errors='coerce')\n`;
                } else if (step.dtype === 'category') {
                    pythonCode += `    df_processed['${step.column}'] = df_processed['${step.column}'].astype('category')\n`;
                }
                break;
                
            case 'encoding':
                if (step.method === 'label') {
                    pythonCode += `    # Apply Label Encoding to column '${step.column}'\n`;
                    pythonCode += `    le = LabelEncoder()\n`;
                    pythonCode += `    df_processed['${step.column}'] = le.fit_transform(df_processed['${step.column}'].astype(str))\n`;
                } else if (step.method === 'onehot') {
                    pythonCode += `    # Apply One-Hot Encoding to column '${step.column}'\n`;
                    pythonCode += `    dummies = pd.get_dummies(df_processed['${step.column}'], prefix='${step.column}', drop_first=True)\n`;
                    pythonCode += `    df_processed = pd.concat([df_processed.drop('${step.column}', axis=1), dummies], axis=1)\n`;
                }
                break;
                
            case 'handle_missing':
                pythonCode += `    # Handle missing values in column '${step.column}' using ${step.method}\n`;
                if (step.method === 'drop') {
                    pythonCode += `    df_processed = df_processed.dropna(subset=['${step.column}'])\n`;
                } else {
                    let fillValue = '0';
                    if (step.method === 'mean') {
                        fillValue = `df_processed['${step.column}'].mean()`;
                    } else if (step.method === 'median') {
                        fillValue = `df_processed['${step.column}'].median()`;
                    } else if (step.method === 'mode') {
                        fillValue = `df_processed['${step.column}'].mode()[0] if not df_processed['${step.column}'].mode().empty else 0`;
                    }
                    pythonCode += `    df_processed['${step.column}'].fillna(${fillValue}, inplace=True)\n`;
                }
                break;
                
            case 'remove_outliers':
                pythonCode += `    # Remove outliers from column '${step.column}' using IQR method\n`;
                pythonCode += `    Q1 = df_processed['${step.column}'].quantile(0.25)\n`;
                pythonCode += `    Q3 = df_processed['${step.column}'].quantile(0.75)\n`;
                pythonCode += `    IQR = Q3 - Q1\n`;
                pythonCode += `    lower_bound = Q1 - 1.5 * IQR\n`;
                pythonCode += `    upper_bound = Q3 + 1.5 * IQR\n`;
                pythonCode += `    df_processed = df_processed[(df_processed['${step.column}'] >= lower_bound) & (df_processed['${step.column}'] <= upper_bound)]\n`;
                break;
                
            case 'batch_encoding':
                pythonCode += `    # Batch ${step.method} encoding for ${step.columns.length} columns\n`;
                step.columns.forEach(col => {
                    if (step.method === 'label') {
                        pythonCode += `    le = LabelEncoder()\n`;
                        pythonCode += `    df_processed['${col}'] = le.fit_transform(df_processed['${col}'].astype(str))\n`;
                    } else if (step.method === 'onehot') {
                        pythonCode += `    dummies = pd.get_dummies(df_processed['${col}'], prefix='${col}', drop_first=True)\n`;
                        pythonCode += `    df_processed = pd.concat([df_processed.drop('${col}', axis=1), dummies], axis=1)\n`;
                    }
                });
                break;
        }
    });
    
    pythonCode += `\n    return df_processed\n`;
    
    // Create download
    const pythonBlob = new Blob([pythonCode], { type: 'text/x-python' });
    const pythonUrl = URL.createObjectURL(pythonBlob);
    
    const a = document.createElement('a');
    a.href = pythonUrl;
    a.download = 'preprocessing_pipeline.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(pythonUrl);
    showToast('Python pipeline exported successfully', 'success');
}

// Utility functions
function showLoading(message = 'Processing...') {
    loadingMessage.textContent = message;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = 'toast show ' + type;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showHelp() {
    document.getElementById('help-modal').classList.remove('hidden');
}

function showAbout() {
    document.getElementById('about-modal').classList.remove('hidden');
}

function closeModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.add('hidden');
    });
}

// Make functions available globally
window.showStep = showStep;
window.downloadFile = downloadFile;
window.downloadReport = downloadReport;
window.downloadAll = downloadAll;
window.startOver = startOver;
window.exportPipeline = exportPipeline;
window.showAccuracyChart = showAccuracyChart;
window.showImprovementChart = showImprovementChart;
window.addSuggestionEncoding = addSuggestionEncoding;
window.addDataTypeConversion = addDataTypeConversion;
window.addOutlierRemoval = addOutlierRemoval;
window.removePreprocessingStep = removePreprocessingStep;
window.removeMissingValueSteps = removeMissingValueSteps;
window.removeAllOutlierSteps = removeAllOutlierSteps;