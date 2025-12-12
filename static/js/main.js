// static/js/main.js - Complete and Corrected Version with Column Removal

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
let scalingSteps = [];
let featureSteps = [];
let batchSteps = [];
let columnTypes = null;
let suggestions = null;
let modelResults = null;
let columnsToRemove = [];
let columnAnalysis = null;

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
const columnAnalysisDiv = document.getElementById('column-analysis');
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

// Column removal elements
const columnRemovalSection = document.getElementById('column-removal-section');
const columnSelectionGrid = document.getElementById('column-selection-grid');
const removalSummary = document.getElementById('removal-summary');
const selectedCount = document.getElementById('selected-count');
const selectedColumnsList = document.getElementById('selected-columns-list');
const selectHighMissingBtn = document.getElementById('select-high-missing');
const selectConstantBtn = document.getElementById('select-constant');
const clearSelectionBtn = document.getElementById('clear-selection');

// Analysis control elements
const toggleColumnStatsBtn = document.getElementById('toggle-column-stats');
const toggleCorrelationsBtn = document.getElementById('toggle-correlations');
const exportAnalysisBtn = document.getElementById('export-analysis');
const columnAnalysisSection = document.getElementById('column-analysis-section');
const statisticsSection = document.getElementById('statistics-section');
const correlationSection = document.getElementById('correlation-section');
const basicStatsDiv = document.getElementById('basic-stats');
const correlationMatrixDiv = document.getElementById('correlation-matrix');

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
        if (targetColumn) {
            updateStepState(2);
            // Enable column removal section when target is selected
            columnRemovalSection.classList.remove('hidden');
        }
    });
    
    // Column removal buttons
    selectHighMissingBtn.addEventListener('click', selectHighMissingColumns);
    selectConstantBtn.addEventListener('click', selectConstantColumns);
    clearSelectionBtn.addEventListener('click', clearColumnSelection);
    
    // Analysis controls
    toggleColumnStatsBtn.addEventListener('click', toggleColumnStats);
    toggleCorrelationsBtn.addEventListener('click', toggleCorrelations);
    exportAnalysisBtn.addEventListener('click', exportAnalysis);

    // Step 3: Preprocessing
    document.getElementById('quick-clean').addEventListener('click', quickClean);
    document.getElementById('auto-preprocess').addEventListener('click', autoPreprocess);
    document.getElementById('get-suggestions-btn').addEventListener('click', getPreprocessingSuggestions);
    document.getElementById('add-conversion').addEventListener('click', addConversionStep);
    document.getElementById('add-encoding').addEventListener('click', addEncodingStep);
    document.getElementById('add-outlier').addEventListener('click', addOutlierStep);
    document.getElementById('add-scaling').addEventListener('click', addScalingStep);
    document.getElementById('add-feature').addEventListener('click', addFeatureStep);
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
            if (preprocessingSteps.length === 0 && columnsToRemove.length === 0) {
                if (!confirm('No preprocessing steps or columns to remove selected. Continue anyway?')) {
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
    document.getElementById('memory-usage').textContent = datasetInfo.memory_usage_mb ? 
        datasetInfo.memory_usage_mb.toFixed(2) + ' MB' : 'N/A';
    
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
        columnRemovalSection.classList.remove('hidden');
    }
    
    // Load column selection grid
    loadColumnSelectionGrid();
    
    // Load quick data quality metrics
    loadDataQualityMetrics();
}

function loadColumnSelectionGrid() {
    if (!datasetInfo) return;
    
    columnSelectionGrid.innerHTML = '';
    
    datasetInfo.column_names.forEach(column => {
        if (column === targetColumn) return; // Don't allow removing target column
        
        const columnDiv = document.createElement('div');
        columnDiv.className = 'column-checkbox';
        
        const checkboxId = `col-${column.replace(/\W/g, '_')}`;
        const isSelected = columnsToRemove.includes(column);
        
        // Determine column type for badge
        let columnType = 'unknown';
        let typeBadge = '';
        
        if (columnTypes) {
            if (columnTypes.categorical && columnTypes.categorical.includes(column)) {
                columnType = 'categorical';
                typeBadge = '<span class="column-type-badge categorical">Cat</span>';
            } else if (columnTypes.numerical && columnTypes.numerical.includes(column)) {
                columnType = 'numerical';
                typeBadge = '<span class="column-type-badge numerical">Num</span>';
            } else if (columnTypes.datetime && columnTypes.datetime.includes(column)) {
                columnType = 'datetime';
                typeBadge = '<span class="column-type-badge datetime">Date</span>';
            }
        }
        
        // Get missing percentage for tooltip
        const missingPct = datasetInfo.missing_percentage ? 
            (datasetInfo.missing_percentage[column] || 0).toFixed(1) : '0';
        
        columnDiv.innerHTML = `
            <input type="checkbox" id="${checkboxId}" ${isSelected ? 'checked' : ''}>
            <label for="${checkboxId}" title="Missing: ${missingPct}%">
                ${column}
                ${typeBadge}
            </label>
        `;
        
        // Add event listener
        const checkbox = columnDiv.querySelector('input');
        checkbox.addEventListener('change', function() {
            updateColumnSelection(column, this.checked);
        });
        
        columnSelectionGrid.appendChild(columnDiv);
    });
    
    updateRemovalSummary();
}

function updateColumnSelection(column, isSelected) {
    if (isSelected) {
        if (!columnsToRemove.includes(column)) {
            columnsToRemove.push(column);
        }
    } else {
        const index = columnsToRemove.indexOf(column);
        if (index > -1) {
            columnsToRemove.splice(index, 1);
        }
    }
    
    updateRemovalSummary();
}

function updateRemovalSummary() {
    if (columnsToRemove.length > 0) {
        removalSummary.classList.remove('hidden');
        selectedCount.textContent = columnsToRemove.length;
        
        selectedColumnsList.innerHTML = '';
        columnsToRemove.forEach(column => {
            const tag = document.createElement('div');
            tag.className = 'selected-column-tag';
            tag.innerHTML = `
                ${column}
                <button onclick="removeColumnFromSelection('${column}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            selectedColumnsList.appendChild(tag);
        });
    } else {
        removalSummary.classList.add('hidden');
    }
}

function removeColumnFromSelection(column) {
    const index = columnsToRemove.indexOf(column);
    if (index > -1) {
        columnsToRemove.splice(index, 1);
    }
    
    // Uncheck the checkbox
    const checkboxId = `col-${column.replace(/\W/g, '_')}`;
    const checkbox = document.getElementById(checkboxId);
    if (checkbox) {
        checkbox.checked = false;
    }
    
    updateRemovalSummary();
}

function selectHighMissingColumns() {
    if (!datasetInfo || !datasetInfo.missing_percentage) return;
    
    columnsToRemove = [];
    
    Object.entries(datasetInfo.missing_percentage).forEach(([column, percentage]) => {
        if (percentage > 50 && column !== targetColumn) {
            columnsToRemove.push(column);
            
            // Check the checkbox
            const checkboxId = `col-${column.replace(/\W/g, '_')}`;
            const checkbox = document.getElementById(checkboxId);
            if (checkbox) {
                checkbox.checked = true;
            }
        }
    });
    
    updateRemovalSummary();
    showToast(`Selected ${columnsToRemove.length} columns with >50% missing values`, 'success');
}

function selectConstantColumns() {
    if (!columnAnalysis) return;
    
    columnsToRemove = [];
    
    columnAnalysis.forEach(col => {
        if (col.unique_count <= 1 && col.name !== targetColumn) {
            columnsToRemove.push(col.name);
            
            // Check the checkbox
            const checkboxId = `col-${col.name.replace(/\W/g, '_')}`;
            const checkbox = document.getElementById(checkboxId);
            if (checkbox) {
                checkbox.checked = true;
            }
        }
    });
    
    updateRemovalSummary();
    showToast(`Selected ${columnsToRemove.length} constant columns`, 'success');
}

function clearColumnSelection() {
    columnsToRemove = [];
    
    // Uncheck all checkboxes
    document.querySelectorAll('#column-selection-grid input[type="checkbox"]').forEach(checkbox => {
        checkbox.checked = false;
    });
    
    updateRemovalSummary();
    showToast('Column selection cleared', 'info');
}

function loadDataQualityMetrics() {
    if (!datasetInfo) return;
    
    let missingCount = 0;
    let highMissingCount = 0;
    
    if (datasetInfo.missing_percentage) {
        Object.values(datasetInfo.missing_percentage).forEach(percentage => {
            if (percentage > 0) missingCount++;
            if (percentage > 50) highMissingCount++;
        });
    }
    
    const qualityScore = calculateDataQualityScore();
    
    dataQualityMetrics.innerHTML = `
        <h3><i class="fas fa-star"></i> Data Quality Metrics</h3>
        <div class="info-cards">
            <div class="info-card">
                <h4><i class="fas fa-exclamation-triangle"></i> Missing Values</h4>
                <p><strong>Columns with missing values:</strong> ${missingCount}/${datasetInfo.columns}</p>
                <p><strong>High missing (>50%):</strong> ${highMissingCount} columns</p>
                <p><strong>Total missing cells:</strong> ${datasetInfo.missing_values ? Object.values(datasetInfo.missing_values).reduce((a, b) => a + b, 0).toLocaleString() : '0'}</p>
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
    if (datasetInfo.missing_percentage) {
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
    }
    
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
            columnAnalysis = data.analysis.column_analysis;
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
    columnAnalysisSection.classList.remove('hidden');
    
    // Store analysis data
    columnAnalysis = analysis.column_analysis;
    
    // Display column analysis
    displayColumnAnalysis(analysis.column_analysis);
    
    // Display basic statistics if available
    if (analysis.basic_stats && Object.keys(analysis.basic_stats).length > 0) {
        displayBasicStatistics(analysis.basic_stats);
    }
    
    // Display correlation matrix if available
    if (analysis.correlation_matrix && analysis.correlation_matrix.length > 0) {
        displayCorrelationMatrix(analysis.correlation_matrix);
    }
    
    // Get column types and suggestions for preprocessing
    getColumnTypes();
}

function displayColumnAnalysis(columns) {
    let columnAnalysisHTML = '';
    
    columns.forEach(col => {
        // Determine column type class
        let typeClass = 'other';
        if (col.type === 'numerical') {
            typeClass = 'numerical';
        } else if (col.type === 'categorical') {
            typeClass = 'categorical';
        }
        
        columnAnalysisHTML += `
            <div class="column-card">
                <div class="column-header">
                    <h5>${col.name}</h5>
                    <span class="column-type ${typeClass}">${col.type || col.dtype}</span>
                </div>
                <div class="column-details">
                    <p><strong>Data Type:</strong> ${col.dtype}</p>
                    <p><strong>Total Values:</strong> ${col.total_count ? col.total_count.toLocaleString() : 'N/A'}</p>
                    <p><strong>Non-Null Values:</strong> ${col.non_null_count ? col.non_null_count.toLocaleString() : 'N/A'}</p>
                    <p><strong>Missing Values:</strong> ${col.null_count ? col.null_count.toLocaleString() : '0'} (${col.null_percentage ? col.null_percentage.toFixed(2) : '0'}%)</p>
                    <p><strong>Unique Values:</strong> ${col.unique_count ? col.unique_count.toLocaleString() : 'N/A'}</p>
                    
                    ${col.type === 'numerical' || col.min !== undefined ? `
                        <p><strong>Range:</strong> ${col.min !== null && col.min !== undefined ? col.min.toFixed(2) : 'N/A'} - ${col.max !== null && col.max !== undefined ? col.max.toFixed(2) : 'N/A'}</p>
                        <p><strong>Mean:</strong> ${col.mean !== null && col.mean !== undefined ? col.mean.toFixed(2) : 'N/A'}</p>
                        <p><strong>Std Dev:</strong> ${col.std !== null && col.std !== undefined ? col.std.toFixed(2) : 'N/A'}</p>
                        ${col.outlier_count ? `<p><strong>Outliers:</strong> ${col.outlier_count} (${col.outlier_percentage ? col.outlier_percentage.toFixed(2) : '0'}%)</p>` : ''}
                    ` : ''}
                    
                    ${col.type === 'categorical' && col.top_values ? `
                        <p><strong>Top Values:</strong></p>
                        <ul class="top-values">
                            ${Object.entries(col.top_values).slice(0, 5).map(([val, count]) => 
                                `<li>${val}: ${count.toLocaleString()}</li>`
                            ).join('')}
                        </ul>
                    ` : ''}
                </div>
            </div>
        `;
    });
    
    columnAnalysisDiv.innerHTML = columnAnalysisHTML;
}

function displayBasicStatistics(stats) {
    let basicStatsHTML = '<table class="data-table"><thead><tr><th>Column</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>50%</th><th>75%</th><th>Max</th></tr></thead><tbody>';
    
    for (const [col, colStats] of Object.entries(stats)) {
        basicStatsHTML += `
            <tr>
                <td>${col}</td>
                <td>${colStats.count ? colStats.count.toLocaleString() : 'N/A'}</td>
                <td>${colStats.mean ? colStats.mean.toFixed(2) : 'N/A'}</td>
                <td>${colStats.std ? colStats.std.toFixed(2) : 'N/A'}</td>
                <td>${colStats.min ? colStats.min.toFixed(2) : 'N/A'}</td>
                <td>${colStats['25%'] ? colStats['25%'].toFixed(2) : 'N/A'}</td>
                <td>${colStats['50%'] ? colStats['50%'].toFixed(2) : 'N/A'}</td>
                <td>${colStats['75%'] ? colStats['75%'].toFixed(2) : 'N/A'}</td>
                <td>${colStats.max ? colStats.max.toFixed(2) : 'N/A'}</td>
            </tr>
        `;
    }
    
    basicStatsHTML += '</tbody></table>';
    basicStatsDiv.innerHTML = basicStatsHTML;
}

function displayCorrelationMatrix(correlations) {
    let correlationHTML = '<table class="data-table"><thead><tr><th>Column 1</th><th>Column 2</th><th>Correlation</th><th>Strength</th></tr></thead><tbody>';
    
    correlations.forEach(corr => {
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
    
    correlationHTML += '</tbody></table>';
    correlationMatrixDiv.innerHTML = correlationHTML;
}

function toggleColumnStats() {
    statisticsSection.classList.toggle('hidden');
    if (!statisticsSection.classList.contains('hidden')) {
        showToast('Basic statistics displayed', 'info');
    }
}

function toggleCorrelations() {
    correlationSection.classList.toggle('hidden');
    if (!correlationSection.classList.contains('hidden')) {
        showToast('Correlation matrix displayed', 'info');
    }
}

function exportAnalysis() {
    if (!columnAnalysis) {
        showToast('No analysis data to export', 'error');
        return;
    }
    
    const analysisData = {
        dataset_info: datasetInfo,
        column_analysis: columnAnalysis,
        columns_to_remove: columnsToRemove,
        timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(analysisData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dataset_analysis.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
    showToast('Analysis exported successfully', 'success');
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
    const outlierColumnSelect = document.getElementById('outlier-column');
    const scaleColumnSelect = document.getElementById('scale-column');
    const featureColumn1Select = document.getElementById('feature-column1');
    const featureColumn2Select = document.getElementById('feature-column2');
    
    // Clear all dropdowns
    const dropdowns = [convertColumnSelect, encodeColumnSelect, outlierColumnSelect, 
                      scaleColumnSelect, featureColumn1Select, featureColumn2Select];
    
    dropdowns.forEach(select => {
        select.innerHTML = '<option value="">Select column</option>';
    });
    
    // Populate with available columns
    const availableColumns = datasetInfo.column_names.filter(col => !columnsToRemove.includes(col));
    
    availableColumns.forEach(column => {
        // Skip target column for feature operations
        if (column === targetColumn && (featureColumn1Select || featureColumn2Select)) {
            return;
        }
        
        // For all dropdowns
        dropdowns.forEach(select => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            select.appendChild(option.cloneNode(true));
        });
    });
}

function displayPreprocessingSuggestions() {
    if (!suggestions) return;
    
    let suggestionsHTML = '<h3><i class="fas fa-lightbulb"></i> Preprocessing Suggestions</h3><div class="suggestions-grid">';
    
    // Column removal suggestions
    if (suggestions.column_removal_suggestions && suggestions.column_removal_suggestions.length > 0) {
        suggestionsHTML += `
            <div class="suggestion-category">
                <h4><i class="fas fa-trash"></i> Column Removal Suggested</h4>
                <p>The following columns might need to be removed:</p>
                <ul class="suggestion-list">
                    ${suggestions.column_removal_suggestions.map(suggestion => `
                        <li>
                            <strong>${suggestion.column}</strong>
                            <span class="suggestion-method">${suggestion.reason}</span>
                            <button class="btn btn-small btn-outline" onclick="addColumnToRemove('${suggestion.column}')">
                                <i class="fas fa-plus"></i> Select
                            </button>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    }
    
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
    
    // Missing value suggestions
    if (suggestions.missing_value_suggestions && suggestions.missing_value_suggestions.length > 0) {
        suggestionsHTML += `
            <div class="suggestion-category">
                <h4><i class="fas fa-tint"></i> Missing Value Handling</h4>
                <p>The following columns have missing values:</p>
                <ul class="suggestion-list">
                    ${suggestions.missing_value_suggestions.slice(0, 10).map(suggestion => `
                        <li>
                            <strong>${suggestion.column}</strong>
                            <span>${suggestion.null_count} missing (${suggestion.null_percentage.toFixed(2)}%)</span>
                            <span class="suggestion-method">Suggested: ${suggestion.suggested_action}</span>
                        </li>
                    `).join('')}
                    ${suggestions.missing_value_suggestions.length > 10 ? 
                        `<li>... and ${suggestions.missing_value_suggestions.length - 10} more columns with missing values</li>` : ''}
                </ul>
            </div>
        `;
    }
    
    suggestionsHTML += '</div>';
    preprocessingSuggestions.innerHTML = suggestionsHTML;
    preprocessingSuggestions.classList.remove('hidden');
}

function addColumnToRemove(column) {
    if (!columnsToRemove.includes(column)) {
        columnsToRemove.push(column);
        updateRemovalSummary();
        showToast(`Added ${column} to removal list`, 'success');
    }
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
    document.getElementById('outlier-column').value = column;
    addOutlierStep();
}

function addConversionStep() {
    const column = document.getElementById('convert-column').value;
    const dtype = document.getElementById('convert-type').value;
    
    if (!column) {
        showToast('Please select a column', 'error');
        return;
    }
    
    if (columnsToRemove.includes(column)) {
        showToast(`Cannot add step for column "${column}" as it is marked for removal`, 'warning');
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
    
    if (columnsToRemove.includes(column)) {
        showToast(`Cannot add step for column "${column}" as it is marked for removal`, 'warning');
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

function addOutlierStep() {
    const column = document.getElementById('outlier-column').value;
    const method = document.getElementById('outlier-method').value;
    
    if (!column) {
        showToast('Please select a column', 'error');
        return;
    }
    
    if (columnsToRemove.includes(column)) {
        showToast(`Cannot add step for column "${column}" as it is marked for removal`, 'warning');
        return;
    }
    
    addPreprocessingStep({
        type: 'remove_outliers',
        column: column,
        method: method
    });
    
    updateOutlierList();
    updatePreprocessingStepsSummary();
    
    // Clear selection
    document.getElementById('outlier-column').value = '';
}

function addScalingStep() {
    const column = document.getElementById('scale-column').value;
    const method = document.getElementById('scale-method').value;
    
    if (!column) {
        showToast('Please select a column', 'error');
        return;
    }
    
    if (columnsToRemove.includes(column)) {
        showToast(`Cannot add step for column "${column}" as it is marked for removal`, 'warning');
        return;
    }
    
    addPreprocessingStep({
        type: 'scale_column',
        column: column,
        method: method
    });
    
    updateScalingList();
    updatePreprocessingStepsSummary();
    
    // Clear selection
    document.getElementById('scale-column').value = '';
}

function addFeatureStep() {
    const column1 = document.getElementById('feature-column1').value;
    const operation = document.getElementById('feature-operation').value;
    const column2 = document.getElementById('feature-column2').value;
    const newColumn = document.getElementById('new-feature-name').value;
    
    if (!column1 || !newColumn) {
        showToast('Please select at least one column and provide a new column name', 'error');
        return;
    }
    
    if (columnsToRemove.includes(column1) || (column2 && columnsToRemove.includes(column2))) {
        showToast('Cannot create feature from columns marked for removal', 'warning');
        return;
    }
    
    const step = {
        type: 'create_feature',
        column1: column1,
        operation: operation,
        new_column: newColumn
    };
    
    if (column2) {
        step.column2 = column2;
    }
    
    addPreprocessingStep(step);
    updateFeatureList();
    updatePreprocessingStepsSummary();
    
    // Clear selection
    document.getElementById('feature-column1').value = '';
    document.getElementById('feature-column2').value = '';
    document.getElementById('new-feature-name').value = '';
}

function quickFillAll() {
    const method = document.getElementById('quick-fill-method').value;
    
    if (!columnTypes || !columnTypes.numerical) return;
    
    // Only include columns not marked for removal
    const columnsToFill = columnTypes.numerical.filter(col => !columnsToRemove.includes(col));
    
    if (columnsToFill.length === 0) {
        showToast('No numeric columns available for filling', 'info');
        return;
    }
    
    columnsToFill.forEach(column => {
        addPreprocessingStep({
            type: 'handle_missing',
            column: column,
            method: method
        });
    });
    
    updateMissingValueList();
    updatePreprocessingStepsSummary();
    
    showToast(`Added ${method} filling for ${columnsToFill.length} numeric columns`, 'success');
}

function batchEncodeCategorical() {
    const method = document.getElementById('batch-encode-method').value;
    
    if (!columnTypes || !columnTypes.categorical) return;
    
    // Only include columns not marked for removal
    const columnsToEncode = columnTypes.categorical.filter(col => !columnsToRemove.includes(col));
    
    if (columnsToEncode.length === 0) {
        showToast('No categorical columns available for encoding', 'info');
        return;
    }
    
    addPreprocessingStep({
        type: 'batch_encoding',
        columns: columnsToEncode,
        method: method
    });
    
    updateBatchStepsList();
    updatePreprocessingStepsSummary();
    
    showToast(`Added ${method} encoding for ${columnsToEncode.length} categorical columns`, 'success');
}

function batchConvertNumerical() {
    if (!columnTypes || !columnTypes.numerical) return;
    
    // This is for converting string columns that should be numeric
    // We need to identify which categorical columns might be numeric
    let columnsToConvert = [];
    
    if (suggestions && suggestions.dtype_conversion_suggestions) {
        columnsToConvert = suggestions.dtype_conversion_suggestions
            .filter(s => s.suggested_dtype === 'numeric')
            .map(s => s.column)
            .filter(col => !columnsToRemove.includes(col));
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
    
    // Only include columns not marked for removal
    const columnsToProcess = columnTypes.numerical.filter(col => !columnsToRemove.includes(col));
    
    if (columnsToProcess.length === 0) {
        showToast('No numerical columns available for outlier removal', 'info');
        return;
    }
    
    columnsToProcess.forEach(column => {
        addPreprocessingStep({
            type: 'remove_outliers',
            column: column,
            method: 'iqr'
        });
    });
    
    updateOutlierList();
    updatePreprocessingStepsSummary();
    
    showToast(`Added outlier removal for ${columnsToProcess.length} numerical columns`, 'success');
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
    } else if (step.type === 'scale_column') {
        scalingSteps.push(step);
    } else if (step.type === 'create_feature') {
        featureSteps.push(step);
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
                <span><i class="fas fa-filter"></i> Remove outliers from "${step.column}" (${step.method})</span>
                <button class="btn btn-small btn-danger" onclick="removePreprocessingStep('outlier', ${index})">
                    <i class="fas fa-times"></i>
                </button>
            `;
            listDiv.appendChild(stepDiv);
        });
    }
}

function updateScalingList() {
    const listDiv = document.getElementById('outlier-list'); // Reusing same list for now
    // Can create separate list if needed
}

function updateFeatureList() {
    // Implementation for feature list display
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
    const totalSteps = preprocessingSteps.length + (columnsToRemove.length > 0 ? 1 : 0);
    
    if (totalSteps === 0) {
        summaryDiv.innerHTML = '<p>No steps added yet. Add steps from above sections.</p>';
        return;
    }
    
    let summaryHTML = `
        <div class="steps-summary">
            <div class="summary-header">
                <h5>Total Steps: ${totalSteps}</h5>
                <span class="steps-count">${totalSteps} steps</span>
            </div>
            <div class="steps-breakdown">
    `;
    
    const stepCounts = {
        'Column Removal': columnsToRemove.length > 0 ? 1 : 0,
        'Data Cleaning': preprocessingSteps.filter(s => 
            s.type === 'drop_high_missing' || s.type === 'remove_duplicates'
        ).length,
        'Type Conversion': conversionSteps.length + 
            (batchSteps.filter(s => s.type === 'batch_dtype_conversion').length > 0 ? 1 : 0),
        'Encoding': encodingSteps.length + 
            (batchSteps.filter(s => s.type === 'batch_encoding').length > 0 ? 1 : 0),
        'Missing Values': missingValueSteps.length,
        'Outlier Removal': outlierSteps.length,
        'Feature Engineering': featureSteps.length,
        'Scaling': scalingSteps.length
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
    
    // Add column removal step if any
    if (columnsToRemove.length > 0) {
        summaryHTML += `<li>Remove ${columnsToRemove.length} columns: ${columnsToRemove.slice(0, 3).join(', ')}${columnsToRemove.length > 3 ? '...' : ''}</li>`;
    }
    
    // Add preprocessing steps
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
                stepDescription = `Remove outliers from "${step.column}" using ${step.method}`;
                break;
            case 'scale_column':
                stepDescription = `Scale "${step.column}" using ${step.method} scaling`;
                break;
            case 'create_feature':
                stepDescription = `Create "${step.new_column}" from ${step.column1} ${step.operation} ${step.column2 || ''}`;
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
    
    const remainingSteps = totalSteps - Math.min(3, preprocessingSteps.length + (columnsToRemove.length > 0 ? 1 : 0));
    if (remainingSteps > 0) {
        summaryHTML += `<li>... and ${remainingSteps} more</li>`;
    }
    
    summaryHTML += `
                </ul>
            </div>
        </div>
    `;
    
    summaryDiv.innerHTML = summaryHTML;
}

function clearAllSteps() {
    if (preprocessingSteps.length === 0 && columnsToRemove.length === 0) return;
    
    if (!confirm('Are you sure you want to clear all preprocessing steps and column selections?')) {
        return;
    }
    
    preprocessingSteps = [];
    conversionSteps = [];
    encodingSteps = [];
    missingValueSteps = [];
    outlierSteps = [];
    scalingSteps = [];
    featureSteps = [];
    batchSteps = [];
    columnsToRemove = [];
    
    // Update UI
    updateConversionList();
    updateEncodingList();
    updateMissingValueList();
    updateOutlierList();
    updateBatchStepsList();
    updatePreprocessingStepsSummary();
    updateRemovalSummary();
    
    showToast('All preprocessing steps and column selections cleared', 'success');
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
        const columnsToFill = columnTypes.numerical.filter(col => !columnsToRemove.includes(col));
        columnsToFill.forEach(column => {
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
    scalingSteps = [];
    featureSteps = [];
    batchSteps = [];
    
    // Add basic cleaning
    addPreprocessingStep({ type: 'drop_high_missing', threshold: 50 });
    addPreprocessingStep({ type: 'remove_duplicates' });
    
    // Add encoding for categorical columns
    if (suggestions.encoding_suggestions && suggestions.encoding_suggestions.length > 0) {
        suggestions.encoding_suggestions.forEach(suggestion => {
            if (!columnsToRemove.includes(suggestion.column)) {
                addPreprocessingStep({
                    type: 'encoding',
                    column: suggestion.column,
                    method: suggestion.suggested_method
                });
            }
        });
    }
    
    // Add missing value handling for numerical columns
    if (columnTypes && columnTypes.numerical) {
        const columnsToFill = columnTypes.numerical.filter(col => !columnsToRemove.includes(col));
        columnsToFill.forEach(column => {
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
            if (!columnsToRemove.includes(suggestion.column) && suggestion.outlier_count > 0) {
                addPreprocessingStep({
                    type: 'remove_outliers',
                    column: suggestion.column,
                    method: 'iqr'
                });
            }
        });
    }
    
    // Add data type conversions
    if (suggestions.dtype_conversion_suggestions && suggestions.dtype_conversion_suggestions.length > 0) {
        suggestions.dtype_conversion_suggestions.forEach(suggestion => {
            if (!columnsToRemove.includes(suggestion.column)) {
                addPreprocessingStep({
                    type: 'change_dtype',
                    column: suggestion.column,
                    dtype: suggestion.suggested_dtype
                });
            }
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
    
    if (preprocessingSteps.length === 0 && columnsToRemove.length === 0) {
        if (!confirm('No preprocessing steps or columns to remove selected. Apply basic cleaning only?')) {
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
            target_column: targetColumn,
            columns_to_remove: columnsToRemove
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
        const statusIcon = step.status === 'success' ? 'fa-check-circle text-success' : 
                          step.status === 'error' ? 'fa-times-circle text-danger' : 
                          'fa-info-circle text-info';
        
        reportHTML += `
            <li>
                <i class="fas ${statusIcon}"></i>
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
    document.getElementById('processed-memory').textContent = data.processed_info.memory_usage_mb ? 
        data.processed_info.memory_usage_mb.toFixed(2) + ' MB' : 'N/A';
    
    // Update performance metrics
    const metrics = data.processed_info.performance_metrics;
    document.getElementById('rows-removed').textContent = metrics.rows_removed || 0;
    document.getElementById('cols-removed').textContent = metrics.columns_removed || 0;
    document.getElementById('outliers-removed').textContent = metrics.outliers_removed || 0;
    document.getElementById('missing-filled').textContent = metrics.missing_values_filled || 0;
    
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
    
    // Create a comprehensive report
    const report = {
        dataset_info: datasetInfo,
        preprocessing_steps: preprocessingSteps,
        columns_removed: columnsToRemove,
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
        scalingSteps = [];
        featureSteps = [];
        batchSteps = [];
        columnTypes = null;
        suggestions = null;
        modelResults = null;
        columnsToRemove = [];
        columnAnalysis = null;
        
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
        
        // Reset analysis sections
        columnAnalysisDiv.innerHTML = '';
        basicStatsDiv.innerHTML = '';
        correlationMatrixDiv.innerHTML = '';
        statisticsSection.classList.add('hidden');
        correlationSection.classList.add('hidden');
        
        // Go back to step 1
        showStep(1);
        
        showToast('Started new analysis', 'success');
    }
}

function exportPipeline() {
    if (preprocessingSteps.length === 0 && columnsToRemove.length === 0) {
        showToast('No preprocessing steps or columns to remove to export', 'error');
        return;
    }
    
    let pythonCode = `# ML Preprocessing Pipeline
# Generated on ${new Date().toISOString()}
# Dataset: ${datasetInfo ? datasetInfo.filename : 'unknown'}
# Target column: ${targetColumn || 'not specified'}
# Columns removed: ${columnsToRemove.length}

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

def preprocess_pipeline(df):
    """
    Apply preprocessing steps to the dataframe.
    Returns: Processed dataframe
    """
    df_processed = df.copy()
    
    # Preprocessing steps
`;
    
    // Add column removal
    if (columnsToRemove.length > 0) {
        pythonCode += `
    # Remove selected columns
    columns_to_remove = ${JSON.stringify(columnsToRemove)}
    df_processed = df_processed.drop(columns=[col for col in columns_to_remove if col in df_processed.columns])
    print(f"Removed {len([col for col in columns_to_remove if col in df_processed.columns])} columns")
`;
    }
    
    preprocessingSteps.forEach(step => {
        pythonCode += `\n    # Step: ${step.type}\n`;
        
        switch(step.type) {
            case 'drop_high_missing':
                pythonCode += `    # Drop columns with >${step.threshold || 50}% missing values\n`;
                pythonCode += `    missing_percentage = (df_processed.isnull().sum() / len(df_processed)) * 100\n`;
                pythonCode += `    columns_to_drop = missing_percentage[missing_percentage > ${step.threshold || 50}].index.tolist()\n`;
                pythonCode += `    df_processed.drop(columns=columns_to_drop, inplace=True)\n`;
                pythonCode += `    print(f"Dropped {len(columns_to_drop)} columns with >${step.threshold || 50}% missing values")\n`;
                break;
                
            case 'remove_duplicates':
                pythonCode += `    # Remove duplicate rows\n`;
                pythonCode += `    before = len(df_processed)\n`;
                pythonCode += `    df_processed.drop_duplicates(inplace=True)\n`;
                pythonCode += `    after = len(df_processed)\n`;
                pythonCode += `    print(f"Removed {before - after} duplicate rows")\n`;
                break;
                
            case 'change_dtype':
                pythonCode += `    # Change data type of column '${step.column}' to ${step.dtype}\n`;
                if (step.dtype === 'numeric') {
                    pythonCode += `    df_processed['${step.column}'] = pd.to_numeric(df_processed['${step.column}'], errors='coerce')\n`;
                } else if (step.dtype === 'datetime') {
                    pythonCode += `    df_processed['${step.column}'] = pd.to_datetime(df_processed['${step.column}'], errors='coerce')\n`;
                } else if (step.dtype === 'category') {
                    pythonCode += `    df_processed['${step.column}'] = df_processed['${step.column}'].astype('category')\n`;
                } else if (step.dtype === 'string') {
                    pythonCode += `    df_processed['${step.column}'] = df_processed['${step.column}'].astype('str')\n`;
                }
                break;
                
            case 'encoding':
                if (step.method === 'label') {
                    pythonCode += `    # Apply Label Encoding to column '${step.column}'\n`;
                    pythonCode += `    le = LabelEncoder()\n`;
                    pythonCode += `    df_processed['${step.column}'] = le.fit_transform(df_processed['${step.column}'].astype(str))\n`;
                    pythonCode += `    print(f"Applied Label Encoding to '{step.column}' with {len(le.classes_)} classes")\n`;
                } else if (step.method === 'onehot') {
                    pythonCode += `    # Apply One-Hot Encoding to column '${step.column}'\n`;
                    pythonCode += `    dummies = pd.get_dummies(df_processed['${step.column}'], prefix='${step.column}')\n`;
                    pythonCode += `    df_processed = pd.concat([df_processed.drop('${step.column}', axis=1), dummies], axis=1)\n`;
                    pythonCode += `    print(f"Applied One-Hot Encoding to '{step.column}', created {len(dummies.columns)} new columns")\n`;
                }
                break;
                
            case 'handle_missing':
                pythonCode += `    # Handle missing values in column '${step.column}' using ${step.method}\n`;
                pythonCode += `    missing_count = df_processed['${step.column}'].isnull().sum()\n`;
                
                if (step.method === 'drop') {
                    pythonCode += `    before = len(df_processed)\n`;
                    pythonCode += `    df_processed = df_processed.dropna(subset=['${step.column}'])\n`;
                    pythonCode += `    after = len(df_processed)\n`;
                    pythonCode += `    print(f"Dropped {before - after} rows with missing values in '{step.column}'")\n`;
                } else {
                    let fillValue = '0';
                    if (step.method === 'mean') {
                        fillValue = `df_processed['${step.column}'].mean()`;
                    } else if (step.method === 'median') {
                        fillValue = `df_processed['${step.column}'].median()`;
                    } else if (step.method === 'mode') {
                        fillValue = `df_processed['${step.column}'].mode()[0] if not df_processed['${step.column}'].mode().empty else 0`;
                    } else if (step.method === 'forward_fill') {
                        pythonCode += `    df_processed['${step.column}'].fillna(method='ffill', inplace=True)\n`;
                        fillValue = null;
                    } else if (step.method === 'backward_fill') {
                        pythonCode += `    df_processed['${step.column}'].fillna(method='bfill', inplace=True)\n`;
                        fillValue = null;
                    } else if (step.method === 'interpolate') {
                        pythonCode += `    df_processed['${step.column}'].interpolate(method='linear', inplace=True)\n`;
                        fillValue = null;
                    }
                    
                    if (fillValue) {
                        pythonCode += `    df_processed['${step.column}'].fillna(${fillValue}, inplace=True)\n`;
                    }
                    pythonCode += `    print(f"Filled {missing_count} missing values in '{step.column}' using ${step.method}")\n`;
                }
                break;
                
            case 'remove_outliers':
                pythonCode += `    # Remove outliers from column '${step.column}' using ${step.method || 'iqr'} method\n`;
                if (step.method === 'zscore') {
                    pythonCode += `    z_scores = np.abs(stats.zscore(df_processed['${step.column}'].dropna()))\n`;
                    pythonCode += `    threshold = ${step.threshold || 3}\n`;
                    pythonCode += `    before = len(df_processed)\n`;
                    pythonCode += `    df_processed = df_processed[(z_scores < threshold) | df_processed['${step.column}'].isna()]\n`;
                    pythonCode += `    after = len(df_processed)\n`;
                } else {
                    pythonCode += `    Q1 = df_processed['${step.column}'].quantile(0.25)\n`;
                    pythonCode += `    Q3 = df_processed['${step.column}'].quantile(0.75)\n`;
                    pythonCode += `    IQR = Q3 - Q1\n`;
                    pythonCode += `    lower_bound = Q1 - 1.5 * IQR\n`;
                    pythonCode += `    upper_bound = Q3 + 1.5 * IQR\n`;
                    pythonCode += `    before = len(df_processed)\n`;
                    pythonCode += `    df_processed = df_processed[(df_processed['${step.column}'] >= lower_bound) & (df_processed['${step.column}'] <= upper_bound)]\n`;
                    pythonCode += `    after = len(df_processed)\n`;
                }
                pythonCode += `    print(f"Removed {before - after} outliers from '{step.column}'")\n`;
                break;
                
            case 'scale_column':
                pythonCode += `    # Scale column '${step.column}' using ${step.method} scaling\n`;
                if (step.method === 'standard') {
                    pythonCode += `    scaler = StandardScaler()\n`;
                    pythonCode += `    df_processed['${step.column}'] = scaler.fit_transform(df_processed[['${step.column}']])\n`;
                } else if (step.method === 'minmax') {
                    pythonCode += `    min_val = df_processed['${step.column}'].min()\n`;
                    pythonCode += `    max_val = df_processed['${step.column}'].max()\n`;
                    pythonCode += `    if max_val > min_val:\n`;
                    pythonCode += `        df_processed['${step.column}'] = (df_processed['${step.column}'] - min_val) / (max_val - min_val)\n`;
                }
                break;
                
            case 'create_feature':
                pythonCode += `    # Create new feature '${step.new_column}' from '${step.column1}' ${step.operation} ${step.column2 || ''}\n`;
                if (step.operation === 'add' && step.column2) {
                    pythonCode += `    df_processed['${step.new_column}'] = df_processed['${step.column1}'] + df_processed['${step.column2}']\n`;
                } else if (step.operation === 'subtract' && step.column2) {
                    pythonCode += `    df_processed['${step.new_column}'] = df_processed['${step.column1}'] - df_processed['${step.column2}']\n`;
                } else if (step.operation === 'multiply' && step.column2) {
                    pythonCode += `    df_processed['${step.new_column}'] = df_processed['${step.column1}'] * df_processed['${step.column2}']\n`;
                } else if (step.operation === 'divide' && step.column2) {
                    pythonCode += `    df_processed['${step.new_column}'] = df_processed['${step.column1}'] / df_processed['${step.column2}'].replace(0, np.nan)\n`;
                } else if (step.operation === 'square') {
                    pythonCode += `    df_processed['${step.new_column}'] = df_processed['${step.column1}'] ** 2\n`;
                } else if (step.operation === 'sqrt') {
                    pythonCode += `    df_processed['${step.new_column}'] = np.sqrt(df_processed['${step.column1}'].abs())\n`;
                } else if (step.operation === 'log') {
                    pythonCode += `    df_processed['${step.new_column}'] = np.log(df_processed['${step.column1}'].replace(0, np.nan).abs() + 1)\n`;
                }
                break;
                
            case 'batch_encoding':
                pythonCode += `    # Batch ${step.method} encoding for ${step.columns.length} columns\n`;
                pythonCode += `    for col in ${JSON.stringify(step.columns)}:\n`;
                pythonCode += `        if col in df_processed.columns:\n`;
                if (step.method === 'label') {
                    pythonCode += `            le = LabelEncoder()\n`;
                    pythonCode += `            df_processed[col] = le.fit_transform(df_processed[col].astype(str))\n`;
                } else if (step.method === 'onehot') {
                    pythonCode += `            dummies = pd.get_dummies(df_processed[col], prefix=col)\n`;
                    pythonCode += `            df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)\n`;
                }
                break;
                
            case 'batch_dtype_conversion':
                pythonCode += `    # Batch conversion to ${step.dtype} for ${step.columns.length} columns\n`;
                pythonCode += `    for col in ${JSON.stringify(step.columns)}:\n`;
                pythonCode += `        if col in df_processed.columns:\n`;
                if (step.dtype === 'numeric') {
                    pythonCode += `            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')\n`;
                } else if (step.dtype === 'datetime') {
                    pythonCode += `            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')\n`;
                } else if (step.dtype === 'category') {
                    pythonCode += `            df_processed[col] = df_processed[col].astype('category')\n`;
                }
                break;
        }
    });
    
    pythonCode += `
    return df_processed

# Usage example:
# df = pd.read_csv('your_dataset.csv')
# df_processed = preprocess_pipeline(df)
# df_processed.to_csv('processed_dataset.csv', index=False)
`;
    
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
window.removeColumnFromSelection = removeColumnFromSelection;
window.addColumnToRemove = addColumnToRemove;
