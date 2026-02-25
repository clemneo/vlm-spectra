// Demo application JavaScript for image upload and prediction
class DemoApp {
    constructor() {
        this.modelReady = false;
        this.isPredicting = false;
        this.uploadedImage = null;
        this.modelSelector = document.getElementById('modelSelector');

        // Square generation mode properties
        this.inputMode = 'upload';  // 'upload' or 'generate'
        this.gridColors = [];       // 2D array of color strings
        this.gridRows = 10;
        this.gridCols = 10;
        this.selectedColor = 'black';
        this.patchInfo = null;
        this.generatedImage = null;
        this.isGridCreated = false;

        this.initializeEventListeners();
        this.initializeModeSelector();
        // Delay initial checks to let external resources load
        setTimeout(() => {
            this.checkDependencies();
            this.loadModelOptions().finally(() => {
                this.checkModelStatus();
            });
        }, 500);
    }
    
    initializeEventListeners() {
        // Analysis buttons
        document.getElementById('analyzeGenerationBtn').addEventListener('click', () => {
            this.handleAnalysis('generation');
        });
        
        document.getElementById('analyzeForwardBtn').addEventListener('click', () => {
            this.handleAnalysis('forward');
        });
        
        // Image upload
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('imageUpload');
        
        // Click to upload
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        // File selection
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0]);
            }
        });
        
        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0) {
                this.handleImageUpload(e.dataTransfer.files[0]);
            }
        });
        
        // Remove image
        document.getElementById('removeImage').addEventListener('click', () => {
            this.removeImage();
        });
        
        // Paste functionality
        document.addEventListener('paste', (e) => {
            this.handlePaste(e);
        });
    }

    initializeModeSelector() {
        const modeSelector = document.getElementById('inputModeSelector');
        const uploadSection = document.getElementById('uploadModeSection');
        const generateSection = document.getElementById('generateModeSection');

        if (!modeSelector) return;

        modeSelector.addEventListener('change', (e) => {
            this.inputMode = e.target.value;

            if (this.inputMode === 'upload') {
                uploadSection.classList.remove('d-none');
                generateSection.classList.add('d-none');
                // Enable buttons only if we have an uploaded image
                if (this.uploadedImage && this.modelReady) {
                    this.enableAnalysisButtons();
                } else {
                    this.disableAnalysisButtons();
                }
            } else {
                uploadSection.classList.add('d-none');
                generateSection.classList.remove('d-none');
                // Fetch patch info when switching to generate mode
                this.fetchPatchInfo();
                // Enable buttons if grid is created and model is ready
                if (this.isGridCreated && this.modelReady) {
                    this.enableAnalysisButtons();
                } else {
                    this.disableAnalysisButtons();
                }
            }
        });

        // Initialize grid controls
        this.initializeGridControls();
    }

    initializeGridControls() {
        const applyGridBtn = document.getElementById('applyGridBtn');
        const gridRowsInput = document.getElementById('gridRows');
        const gridColsInput = document.getElementById('gridCols');
        const fillAllBtn = document.getElementById('fillAllBtn');
        const clearGridBtn = document.getElementById('clearGridBtn');
        const colorPalette = document.getElementById('colorPalette');

        if (applyGridBtn) {
            applyGridBtn.addEventListener('click', () => {
                this.gridRows = parseInt(gridRowsInput.value) || 10;
                this.gridCols = parseInt(gridColsInput.value) || 10;
                // Clamp values
                this.gridRows = Math.max(1, Math.min(50, this.gridRows));
                this.gridCols = Math.max(1, Math.min(50, this.gridCols));
                gridRowsInput.value = this.gridRows;
                gridColsInput.value = this.gridCols;
                this.createGridCells();
                this.updateGridInfoText();
            });
        }

        // Grid size input listeners
        if (gridRowsInput) {
            gridRowsInput.addEventListener('change', () => this.updateGridInfoText());
        }
        if (gridColsInput) {
            gridColsInput.addEventListener('change', () => this.updateGridInfoText());
        }

        // Fill all button
        if (fillAllBtn) {
            fillAllBtn.addEventListener('click', () => {
                if (!this.isGridCreated) return;
                for (let row = 0; row < this.gridRows; row++) {
                    for (let col = 0; col < this.gridCols; col++) {
                        this.gridColors[row][col] = this.selectedColor;
                    }
                }
                this.refreshGridDisplay();
            });
        }

        // Clear grid button
        if (clearGridBtn) {
            clearGridBtn.addEventListener('click', () => {
                if (!this.isGridCreated) return;
                for (let row = 0; row < this.gridRows; row++) {
                    for (let col = 0; col < this.gridCols; col++) {
                        this.gridColors[row][col] = 'black';
                    }
                }
                this.refreshGridDisplay();
            });
        }

        // Color palette
        if (colorPalette) {
            colorPalette.addEventListener('click', (e) => {
                if (e.target.classList.contains('color-btn')) {
                    // Remove active from all buttons
                    colorPalette.querySelectorAll('.color-btn').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    // Set new active
                    e.target.classList.add('active');
                    this.selectedColor = e.target.dataset.color;
                }
            });
        }
    }

    async fetchPatchInfo() {
        try {
            const response = await fetch('/api/model/patch-info');
            const data = await response.json();
            this.patchInfo = data;
            this.updateGridInfoText();
        } catch (error) {
            console.error('Error fetching patch info:', error);
            // Use default patch info
            this.patchInfo = {
                patch_size: 14,
                spatial_merge_size: 2,
                effective_patch_size: 28
            };
            this.updateGridInfoText();
        }
    }

    updateGridInfoText() {
        const infoText = document.getElementById('gridInfoText');
        const gridRowsInput = document.getElementById('gridRows');
        const gridColsInput = document.getElementById('gridCols');

        if (!infoText) return;

        const rows = parseInt(gridRowsInput?.value) || this.gridRows;
        const cols = parseInt(gridColsInput?.value) || this.gridCols;
        const patchSize = this.patchInfo?.effective_patch_size || 28;

        const imgWidth = cols * patchSize;
        const imgHeight = rows * patchSize;

        infoText.textContent = `${rows}x${cols} patches = ${imgWidth}x${imgHeight} pixels (patch size: ${patchSize}px)`;
    }

    createGridCells() {
        const container = document.getElementById('colorGrid');
        if (!container) return;

        container.innerHTML = '';

        // Initialize grid colors array with black
        this.gridColors = [];
        for (let row = 0; row < this.gridRows; row++) {
            this.gridColors[row] = [];
            const rowDiv = document.createElement('div');
            rowDiv.className = 'grid-row';

            for (let col = 0; col < this.gridCols; col++) {
                this.gridColors[row][col] = 'black';

                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.style.backgroundColor = 'black';

                // Click to paint
                cell.addEventListener('click', () => this.paintCell(row, col));

                // Drag to paint (mouse enter while button pressed)
                cell.addEventListener('mouseenter', (e) => {
                    if (e.buttons === 1) {
                        this.paintCell(row, col);
                    }
                });

                rowDiv.appendChild(cell);
            }
            container.appendChild(rowDiv);
        }

        this.isGridCreated = true;

        // Clear generated image preview since grid changed
        this.generatedImage = null;
        document.getElementById('generatedImagePreview')?.classList.add('d-none');

        // Enable analysis buttons if model is ready
        if (this.modelReady) {
            this.enableAnalysisButtons();
        }
    }

    paintCell(row, col) {
        this.gridColors[row][col] = this.selectedColor;
        const cell = document.querySelector(
            `#colorGrid .grid-cell[data-row="${row}"][data-col="${col}"]`
        );
        if (cell) {
            cell.style.backgroundColor = this.selectedColor;
        }
    }

    refreshGridDisplay() {
        for (let row = 0; row < this.gridRows; row++) {
            for (let col = 0; col < this.gridCols; col++) {
                const cell = document.querySelector(
                    `#colorGrid .grid-cell[data-row="${row}"][data-col="${col}"]`
                );
                if (cell) {
                    cell.style.backgroundColor = this.gridColors[row][col];
                }
            }
        }
    }

    async generateSquareImage() {
        try {
            const response = await fetch('/api/generate-square', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    grid_colors: this.gridColors,
                    grid_rows: this.gridRows,
                    grid_cols: this.gridCols
                })
            });

            const result = await response.json();

            if (result.success) {
                this.generatedImage = result;

                // Update preview
                const previewImg = document.getElementById('generatedPreview');
                const previewContainer = document.getElementById('generatedImagePreview');
                const infoText = document.getElementById('generatedImageInfo');

                if (previewImg && previewContainer) {
                    previewImg.src = result.url;
                    previewContainer.classList.remove('d-none');
                }

                if (infoText) {
                    infoText.textContent = `${result.dimensions.width}x${result.dimensions.height} pixels (${result.grid_size.rows}x${result.grid_size.cols} patches @ ${result.patch_size}px)`;
                }
            }

            return result;
        } catch (error) {
            console.error('Error generating square image:', error);
            return { success: false, error: error.message };
        }
    }

    async handleImageUpload(file) {
        // Validate file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, WebP)');
            return;
        }
        
        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB');
            return;
        }
        
        // Show loading state
        const dropZone = document.getElementById('dropZone');
        dropZone.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Uploading...</span></div>';
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.uploadedImage = result;
                this.showImagePreview(result.url);
                this.enableAnalysisButtons();
            } else {
                this.showError(result.error || 'Upload failed');
                this.resetDropZone();
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Failed to upload image');
            this.resetDropZone();
        }
    }
    
    handlePaste(e) {
        const items = e.clipboardData?.items;
        if (!items) return;
        
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            
            // Check if the item is an image
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) {
                    this.handleImageUpload(file);
                }
                break;
            }
        }
    }
    
    showImagePreview(imageUrl) {
        const previewContainer = document.getElementById('imagePreviewContainer');
        const previewImage = document.getElementById('imagePreview');
        
        previewImage.src = imageUrl;
        previewContainer.classList.remove('d-none');
        
        // Hide drop zone
        document.getElementById('dropZone').style.display = 'none';
    }
    
    removeImage() {
        this.uploadedImage = null;
        document.getElementById('imagePreviewContainer').classList.add('d-none');
        document.getElementById('analyzeGenerationBtn').disabled = true;
        document.getElementById('analyzeForwardBtn').disabled = true;
        this.resetDropZone();
    }
    
    resetDropZone() {
        const dropZone = document.getElementById('dropZone');
        dropZone.style.display = 'block';
        dropZone.innerHTML = `
            <div class="drop-zone-content">
                <i class="fas fa-cloud-upload-alt fa-2x mb-2"></i>
                <p class="mb-2">Drag & drop your image here</p>
                <p class="text-muted small">or click to browse, or paste from clipboard</p>
                <input type="file" id="imageUpload" name="image" accept=".png,.jpg,.jpeg,.gif,.bmp,.webp" class="d-none">
            </div>
        `;
        
        // Re-attach event listeners for new elements
        const fileInput = document.getElementById('imageUpload');
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0]);
            }
        });
    }
    
    enableAnalysisButtons() {
        if (this.modelReady) {
            // Check based on current input mode
            if (this.inputMode === 'upload' && this.uploadedImage) {
                document.getElementById('analyzeGenerationBtn').disabled = false;
                document.getElementById('analyzeForwardBtn').disabled = false;
            } else if (this.inputMode === 'generate' && this.isGridCreated) {
                document.getElementById('analyzeGenerationBtn').disabled = false;
                document.getElementById('analyzeForwardBtn').disabled = false;
            }
        }
    }

    disableAnalysisButtons() {
        document.getElementById('analyzeGenerationBtn').disabled = true;
        document.getElementById('analyzeForwardBtn').disabled = true;
    }

    async loadModelOptions() {
        if (!this.modelSelector) {
            return;
        }

        try {
            const response = await fetch('/api/model/options');
            const data = await response.json();

            this.modelSelector.innerHTML = '';

            data.options.forEach((optionInfo) => {
                const option = document.createElement('option');
                option.value = optionInfo.id;
                option.textContent = optionInfo.available
                    ? optionInfo.label
                    : `${optionInfo.label} (unavailable)`;
                option.disabled = !optionInfo.available;
                this.modelSelector.appendChild(option);
            });

            this.modelSelector.value = data.active_model_id || '';
            this.modelSelector.disabled = false;

            this.modelSelector.addEventListener('change', (event) => {
                this.handleModelSelection(event.target.value);
            });
        } catch (error) {
            console.error('Error loading model options:', error);
            this.modelSelector.innerHTML = '<option value="">Failed to load models</option>';
            this.modelSelector.disabled = true;
        }
    }

    async handleModelSelection(modelId) {
        if (!modelId) {
            return;
        }

        this.modelReady = false;
        this.disableAnalysisButtons();

        try {
            const response = await fetch('/api/model/select', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_id: modelId })
            });

            const result = await response.json();

            if (!response.ok) {
                const statusElement = document.getElementById('modelStatus');
                statusElement.className = 'alert alert-danger';
                statusElement.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i>Error: ${result.error || 'Failed to select model'}`;
                return;
            }

            const statusElement = document.getElementById('modelStatus');
            if (result.status === 'ready') {
                statusElement.className = 'alert alert-success';
                statusElement.innerHTML = `<i class="fas fa-check-circle me-2"></i>Model ready: ${result.model_label}`;
                this.modelReady = true;
                this.enableAnalysisButtons();
            } else if (result.status === 'queued') {
                statusElement.className = 'alert alert-info';
                statusElement.innerHTML = `<i class="spinner-border spinner-border-sm me-2"></i>Queued model load: ${result.model_label}`;
            } else {
                statusElement.className = 'alert alert-info';
                statusElement.innerHTML = `<i class="spinner-border spinner-border-sm me-2"></i>Loading model: ${result.model_label}`;
            }

            this.checkModelStatus();
        } catch (error) {
            console.error('Error selecting model:', error);
            const statusElement = document.getElementById('modelStatus');
            statusElement.className = 'alert alert-danger';
            statusElement.innerHTML = '<i class="fas fa-exclamation-circle me-2"></i>Failed to select model';
        }
    }
    
    checkDependencies() {
        const dependencies = [
            {
                name: 'Plotly',
                check: () => typeof Plotly !== 'undefined',
                elementId: 'plotlyStatus'
            },
            {
                name: 'Bootstrap',
                check: () => typeof bootstrap !== 'undefined' || typeof window.bootstrap !== 'undefined',
                elementId: 'bootstrapStatus'
            },
            {
                name: 'FontAwesome',
                check: () => document.querySelectorAll('link[href*="font-awesome"]').length > 0,
                elementId: 'fontawesomeStatus'
            }
        ];
        
        let allLoaded = true;
        const failedDeps = [];
        
        dependencies.forEach(dep => {
            const element = document.getElementById(dep.elementId);
            if (!element) return;
            
            try {
                if (dep.check()) {
                    element.className = 'badge bg-success';
                    element.innerHTML = `<i class="fas fa-check me-1"></i>${dep.name}`;
                } else {
                    element.className = 'badge bg-danger';
                    element.innerHTML = `<i class="fas fa-times me-1"></i>${dep.name}`;
                    allLoaded = false;
                    failedDeps.push(dep.name);
                }
            } catch (error) {
                console.warn(`Error checking dependency ${dep.name}:`, error);
                element.className = 'badge bg-warning';
                element.innerHTML = `<i class="fas fa-question me-1"></i>${dep.name}`;
                allLoaded = false;
                failedDeps.push(dep.name);
            }
        });
        
        const statusCard = document.getElementById('dependenciesStatus');
        const messageElement = document.getElementById('dependenciesMessage');
        
        if (allLoaded) {
            statusCard.className = 'alert alert-success';
            messageElement.innerHTML = '<i class="fas fa-check-circle me-1"></i>All dependencies loaded successfully!';
        } else {
            statusCard.className = 'alert alert-warning';
            messageElement.innerHTML = `<i class="fas fa-exclamation-triangle me-1"></i>Some dependencies failed to load: ${failedDeps.join(', ')}. <strong>Please refresh the page.</strong>`;
            
            // Recheck dependencies after a delay in case they're still loading
            setTimeout(() => this.checkDependencies(), 2000);
        }
    }
    
    async checkModelStatus() {
        try {
            const response = await fetch('/api/model/status');
            const status = await response.json();
            
            const statusElement = document.getElementById('modelStatus');
            const generationBtn = document.getElementById('analyzeGenerationBtn');
            const forwardBtn = document.getElementById('analyzeForwardBtn');
            const modelLabel = status.model_label || 'Model';
            
            if (status.ready) {
                statusElement.className = 'alert alert-success';
                statusElement.innerHTML = `<i class="fas fa-check-circle me-2"></i>Model ready: ${modelLabel}`;
                this.modelReady = true;
                // Enable buttons based on current mode
                if (this.inputMode === 'upload' && this.uploadedImage) {
                    generationBtn.disabled = false;
                    forwardBtn.disabled = false;
                } else if (this.inputMode === 'generate' && this.isGridCreated) {
                    generationBtn.disabled = false;
                    forwardBtn.disabled = false;
                }
                // Also fetch patch info for generate mode if we switch to it
                if (this.inputMode === 'generate') {
                    this.fetchPatchInfo();
                }
            } else if (status.error) {
                statusElement.className = 'alert alert-danger';
                statusElement.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i>Error: ${status.error}`;
                generationBtn.disabled = true;
                forwardBtn.disabled = true;
            } else {
                const pendingLabel = status.pending_model_label ? ` (queued: ${status.pending_model_label})` : '';
                statusElement.className = 'alert alert-info';
                statusElement.innerHTML = `<i class="spinner-border spinner-border-sm me-2"></i>Loading ${modelLabel}...${pendingLabel}`;
                generationBtn.disabled = true;
                forwardBtn.disabled = true;
                // Continue checking
                setTimeout(() => this.checkModelStatus(), 2000);
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            const statusElement = document.getElementById('modelStatus');
            statusElement.className = 'alert alert-danger';
            statusElement.innerHTML = '<i class="fas fa-exclamation-circle me-2"></i>Failed to check model status';
        }
    }
    
    async handleAnalysis(mode) {
        if (!this.modelReady || this.isPredicting) {
            return;
        }

        // Check if we have valid input based on mode
        if (this.inputMode === 'upload' && !this.uploadedImage) {
            return;
        }
        if (this.inputMode === 'generate' && !this.isGridCreated) {
            return;
        }

        this.isPredicting = true;
        const isGeneration = mode === 'generation';
        const btnId = isGeneration ? 'analyzeGenerationBtn' : 'analyzeForwardBtn';
        const spinnerId = isGeneration ? 'generationSpinner' : 'forwardSpinner';
        const endpoint = isGeneration ? '/api/predict' : '/api/forward';

        const button = document.getElementById(btnId);
        const spinner = document.getElementById(spinnerId);

        // Update UI
        document.getElementById('analyzeGenerationBtn').disabled = true;
        document.getElementById('analyzeForwardBtn').disabled = true;
        spinner.classList.remove('d-none');

        // Determine which image to use
        let imageFilename;
        if (this.inputMode === 'upload') {
            imageFilename = this.uploadedImage.filename;
        } else {
            // Generate mode - create image first
            const genResult = await this.generateSquareImage();
            if (!genResult.success) {
                this.showError(genResult.error || 'Failed to generate image');
                document.getElementById('analyzeGenerationBtn').disabled = false;
                document.getElementById('analyzeForwardBtn').disabled = false;
                spinner.classList.add('d-none');
                this.isPredicting = false;
                return;
            }
            imageFilename = genResult.filename;
        }

        const formData = {
            filename: imageFilename,
            task: document.getElementById('task').value,
            assistant_prefill: document.getElementById('assistantPrefill').value
        };
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                if (isGeneration) {
                    this.displayGenerationResults(result);
                } else {
                    this.displayForwardResults(result);
                }
            } else {
                this.showError(result.error || `${mode} analysis failed`);
            }
        } catch (error) {
            console.error(`Error during ${mode} analysis:`, error);
            this.showError('Failed to connect to server');
        } finally {
            // Reset UI
            document.getElementById('analyzeGenerationBtn').disabled = false;
            document.getElementById('analyzeForwardBtn').disabled = false;
            spinner.classList.add('d-none');
            this.isPredicting = false;
        }
    }
    
    displayGenerationResults(result) {
        const container = document.getElementById('resultsContainer');
        const template = document.getElementById('generationResultsTemplate');
        
        // Clone template
        const resultsContent = template.cloneNode(true);
        resultsContent.classList.remove('d-none');
        resultsContent.id = '';
        
        // Generate unique IDs for collapsible sections
        const timestamp = Date.now();
        this.updateCollapsibleIds(resultsContent, timestamp);
        
        // Populate image
        const img = resultsContent.querySelector('#resultImage');
        img.src = result.image_url;
        img.onload = () => {
            this.drawCoordinateOverlay(img, result);
        };
        
        // Populate data
        resultsContent.querySelector('#imageSize').textContent = `${result.image_size[0]}×${result.image_size[1]}`;
        
        resultsContent.querySelector('#modelOutput').textContent = result.output_text;
        resultsContent.querySelector('#taskText').textContent = result.task;
        resultsContent.querySelector('#prefillText').textContent = result.prefill || '(none)';
        
        // Prediction
        const pred = result.prediction;
        resultsContent.querySelector('#prediction').textContent = 
            pred.x !== null ? `(${pred.x}, ${pred.y})` : 'No prediction';
        
        // Since we don't have ground truth for uploaded images, hide accuracy info
        const accuracyRow = resultsContent.querySelector('#accuracyBadge').closest('tr');
        if (accuracyRow) {
            accuracyRow.style.display = 'none';
        }
        
        // Inference time
        resultsContent.querySelector('#inferenceTime').textContent = `${result.inference_time}s`;
        
        // Replace container content
        container.innerHTML = '';
        container.appendChild(resultsContent);
        
        // Add collapse icon toggle functionality
        this.initializeCollapseIcons(resultsContent);
    }
    
    displayForwardResults(result) {
        const container = document.getElementById('resultsContainer');
        const template = document.getElementById('forwardResultsTemplate');
        
        // Clone template
        const resultsContent = template.cloneNode(true);
        resultsContent.classList.remove('d-none');
        resultsContent.id = '';
        
        // Generate unique IDs for collapsible sections
        const timestamp = Date.now();
        this.updateCollapsibleIds(resultsContent, timestamp);
        
        // Populate data (no image display)
        resultsContent.querySelector('#imageSizeForward').textContent = `${result.image_size[0]}×${result.image_size[1]}`;
        resultsContent.querySelector('#taskTextForward').textContent = result.task;
        resultsContent.querySelector('#prefillTextForward').textContent = result.prefill || '(none)';
        resultsContent.querySelector('#tokenPosition').textContent = result.token_position || 'Next token';
        resultsContent.querySelector('#inferenceTimeForward').textContent = `${result.inference_time}s`;
        
        // Create interactive Plotly bar chart for token predictions
        if (result.top_tokens) {
            this.createTokenPredictionsChart(resultsContent, result.top_tokens);
        }
        
        // Set up token layer analysis if data is available
        if (result.layer_probabilities && result.top_tokens) {
            this.setupTokenLayerAnalysis(resultsContent, result.top_tokens, result.layer_probabilities);
        }
        
        // Set up logit lens analysis
        this.setupLogitLensAnalysis(resultsContent);

        // Set up attention explorer if model info is available
        if (result.model_info) {
            this.setupAttentionExplorer(resultsContent, result.model_info);
        }

        // Set up DLA analysis if data is available (populate token selector)
        this.setupDlaAnalysis(resultsContent, result.top_tokens);
        
        // Enable DLA button now that forward pass analysis is complete
        const dlaBtn = resultsContent.querySelector('#analyzeDlaBtn');
        if (dlaBtn) {
            dlaBtn.disabled = false;
        }
        
        // Replace container content
        container.innerHTML = '';
        container.appendChild(resultsContent);
        
        // Add collapse icon toggle functionality
        this.initializeCollapseIcons(resultsContent);
    }
    
    createTokenPredictionsChart(resultsContent, topTokens) {
        const chartContainer = resultsContent.querySelector('#tokenPredictionsChart');
        
        // Check if Plotly is available
        if (typeof Plotly === 'undefined') {
            console.error('Plotly is not loaded');
            chartContainer.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle me-2"></i>Chart library not loaded. Please refresh the page.</div>';
            return;
        }
        
        // Prepare data for the chart
        const tokens = topTokens.map((token, index) => {
            // Replace whitespace characters with underscores for visibility
            return token.token.replace(/\s/g, '_');
        });
        
        const probabilities = topTokens.map(token => (token.probability * 100));
        const logits = topTokens.map(token => token.logit);
        
        // Create the bar chart data
        const data = [{
            x: tokens,
            y: probabilities,
            type: 'bar',
            marker: {
                color: '#0d6efd'
            },
            text: probabilities.map(p => `${p.toFixed(2)}%`),
            textposition: 'outside',
            hovertemplate: 
                '<b>Token:</b> %{x}<br>' +
                '<b>Probability:</b> %{y:.2f}%<br>' +
                '<b>Logit:</b> %{customdata:.4f}<extra></extra>',
            customdata: logits
        }];
        
        // Configure the layout
        const layout = {
            title: {
                text: 'Token Prediction Probabilities',
                x: 0.5,
                font: { size: 16 }
            },
            xaxis: {
                title: 'Tokens',
                showgrid: false,
                tickfont: { family: 'monospace', size: 12 }
            },
            yaxis: {
                title: 'Probability (%)',
                showgrid: true,
                gridcolor: '#f0f0f0'
            },
            margin: {
                l: 60,
                r: 60,
                t: 60,
                b: 120
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
            font: {
                family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                size: 12
            }
        };
        
        // Configure options
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        };
        
        // Create the plot
        Plotly.newPlot(chartContainer, data, layout, config);
    }
    
    setupTokenLayerAnalysis(resultsContent, topTokens, layerProbabilities) {
        const tokenSelector = resultsContent.querySelector('#tokenSelector');
        const chartContainer = resultsContent.querySelector('#tokenLayerChart');
        
        if (!tokenSelector || !chartContainer) {
            console.warn('Token layer analysis elements not found');
            return;
        }
        
        // Populate token selector dropdown
        tokenSelector.innerHTML = '';
        topTokens.forEach((token, index) => {
            const option = document.createElement('option');
            option.value = index;
            // Replace whitespace with underscores for display, similar to top 10 token predictions
            const displayToken = token.token.replace(/\s/g, '_');
            option.textContent = `${displayToken} (${(token.probability * 100).toFixed(2)}%)`;
            tokenSelector.appendChild(option);
        });
        
        // Store data for chart updates
        tokenSelector.dataset.topTokens = JSON.stringify(topTokens);
        tokenSelector.dataset.layerProbabilities = JSON.stringify(layerProbabilities);
        
        // Create initial chart for first token
        this.createTokenLayerChart(chartContainer, topTokens, layerProbabilities, 0);
        
        // Add event listener for token selection changes
        tokenSelector.addEventListener('change', (event) => {
            const selectedIndex = parseInt(event.target.value);
            this.createTokenLayerChart(chartContainer, topTokens, layerProbabilities, selectedIndex);
        });
    }
    
    createTokenLayerChart(chartContainer, topTokens, layerProbabilities, selectedTokenIndex) {
        if (!chartContainer || !topTokens || !layerProbabilities) return;
        
        // Check if Plotly is available
        if (typeof Plotly === 'undefined') {
            console.error('Plotly is not loaded');
            chartContainer.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle me-2"></i>Chart library not loaded. Please refresh the page.</div>';
            return;
        }
        
        const selectedToken = topTokens[selectedTokenIndex];
        const numLayers = layerProbabilities.length;
        
        // Replace whitespace with underscores for display consistency
        const displayToken = selectedToken.token.replace(/\s/g, '_');
        
        // Extract probabilities for the selected token across all layers
        const layerNumbers = Array.from({length: numLayers}, (_, i) => i);
        const probabilities = layerProbabilities.map(layerProbs => layerProbs[selectedTokenIndex] * 100);
        
        // Create the line chart data
        const data = [{
            x: layerNumbers,
            y: probabilities,
            type: 'scatter',
            mode: 'lines+markers',
            name: `Token: "${displayToken}"`,
            line: {
                color: '#0d6efd',
                width: 3
            },
            marker: {
                size: 6,
                color: '#0d6efd'
            },
            hovertemplate: 
                '<b>Layer:</b> %{x}<br>' +
                '<b>Probability:</b> %{y:.3f}%<br>' +
                `<b>Token:</b> "${displayToken}"<extra></extra>`
        }];
        
        // Configure the layout
        const layout = {
            title: {
                text: `Token Probability Over Layers: "${displayToken}"`,
                x: 0.5,
                font: { size: 16 }
            },
            xaxis: {
                title: 'Layer Number',
                showgrid: true,
                gridcolor: '#f0f0f0',
                tickmode: 'linear',
                dtick: 1
            },
            yaxis: {
                title: 'Probability (%)',
                showgrid: true,
                gridcolor: '#f0f0f0',
                range: [0, 100]  // Fixed y-axis range from 0% to 100%
            },
            margin: {
                l: 60,
                r: 60,
                t: 60,
                b: 60
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff',
            font: {
                family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                size: 12
            }
        };
        
        // Configure options
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        };
        
        // Create or update the plot
        Plotly.newPlot(chartContainer, data, layout, config);
    }
    
    updateCollapsibleIds(element, timestamp) {
        // Update IDs for generation results template
        const idMappings = [
            { old: 'imageSection', new: `imageSection_${timestamp}` },
            { old: 'modelOutputSection', new: `modelOutputSection_${timestamp}` },
            { old: 'predictionResultsSection', new: `predictionResultsSection_${timestamp}` },
            { old: 'tokenPredictionsSection', new: `tokenPredictionsSection_${timestamp}` },
            { old: 'tokenLayerSection', new: `tokenLayerSection_${timestamp}` },
            { old: 'logitLensSection', new: `logitLensSection_${timestamp}` },
            { old: 'attentionSection', new: `attentionSection_${timestamp}` },
            { old: 'dlaSection', new: `dlaSection_${timestamp}` },
            { old: 'analysisDetailsSection', new: `analysisDetailsSection_${timestamp}` }
        ];
        
        idMappings.forEach(mapping => {
            const collapseElement = element.querySelector(`#${mapping.old}`);
            if (collapseElement) {
                collapseElement.id = mapping.new;
                
                // Update corresponding button's data-bs-target
                const button = element.querySelector(`[data-bs-target="#${mapping.old}"]`);
                if (button) {
                    button.setAttribute('data-bs-target', `#${mapping.new}`);
                }
            }
        });
    }
    
    initializeCollapseIcons(element) {
        const toggleButtons = element.querySelectorAll('[data-bs-toggle="collapse"]');
        
        toggleButtons.forEach(button => {
            const target = button.getAttribute('data-bs-target');
            const collapseElement = element.querySelector(target);
            const icon = button.querySelector('.collapse-icon');
            
            if (collapseElement && icon) {
                // Set initial icon state based on collapse state
                if (collapseElement.classList.contains('show')) {
                    icon.textContent = '▼';
                    button.setAttribute('aria-expanded', 'true');
                } else {
                    icon.textContent = '▶';
                    button.setAttribute('aria-expanded', 'false');
                }
                
                // Listen for collapse events
                collapseElement.addEventListener('shown.bs.collapse', () => {
                    icon.textContent = '▼';
                    button.setAttribute('aria-expanded', 'true');
                });
                
                collapseElement.addEventListener('hidden.bs.collapse', () => {
                    icon.textContent = '▶';
                    button.setAttribute('aria-expanded', 'false');
                });
            }
        });
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    drawCoordinateOverlay(img, result) {
        const canvas = document.querySelector('#overlayCanvas');
        if (!canvas) return;
        
        const rect = img.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
        
        const ctx = canvas.getContext('2d');
        const scaleX = rect.width / img.naturalWidth;
        const scaleY = rect.height / img.naturalHeight;
        
        // Draw prediction (red cross)
        if (result.prediction.x !== null) {
            const predX = result.prediction.x * scaleX;
            const predY = result.prediction.y * scaleY;
            const crossSize = 12;
            
            // White outline for visibility
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 5;
            ctx.beginPath();
            ctx.moveTo(predX, predY - crossSize);
            ctx.lineTo(predX, predY + crossSize);
            ctx.moveTo(predX - crossSize, predY);
            ctx.lineTo(predX + crossSize, predY);
            ctx.stroke();
            
            // Red cross on top
            ctx.strokeStyle = '#dc3545';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(predX, predY - crossSize);
            ctx.lineTo(predX, predY + crossSize);
            ctx.moveTo(predX - crossSize, predY);
            ctx.lineTo(predX + crossSize, predY);
            ctx.stroke();
            
            // Label with background
            ctx.fillStyle = '#fff';
            ctx.fillRect(predX - 20, predY - 30, 40, 15);
            ctx.strokeStyle = '#dc3545';
            ctx.lineWidth = 1;
            ctx.strokeRect(predX - 20, predY - 30, 40, 15);
            
            ctx.fillStyle = '#dc3545';
            ctx.font = 'bold 11px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('CLICK', predX, predY - 18);
            ctx.textAlign = 'start';
        }
    }
    
    showError(message) {
        const container = document.getElementById('resultsContainer');
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
    }
    
    // ---- Logit Lens ----

    setupLogitLensAnalysis(resultsContent) {
        const btn = resultsContent.querySelector('#analyzeLogitLensBtn');
        if (!btn) return;
        btn.disabled = false;
        btn.addEventListener('click', () => {
            this.handleLogitLensAnalysis();
        });
    }

    async handleLogitLensAnalysis() {
        if (!this.modelReady || this.isPredicting) return;
        if (this.inputMode === 'upload' && !this.uploadedImage) return;
        if (this.inputMode === 'generate' && !this.generatedImage) return;

        this.isPredicting = true;
        const button = document.getElementById('analyzeLogitLensBtn');
        const spinner = document.getElementById('logitLensSpinner');
        button.disabled = true;
        spinner.classList.remove('d-none');

        const imageFilename = this.inputMode === 'upload'
            ? this.uploadedImage.filename
            : this.generatedImage.filename;

        const formData = {
            filename: imageFilename,
            task: document.getElementById('task').value,
            assistant_prefill: document.getElementById('assistantPrefill').value
        };

        try {
            const response = await fetch('/api/logit-lens', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const result = await response.json();

            if (result.success) {
                const container = document.getElementById('logitLensContainer');
                container.classList.remove('d-none');
                this.renderLogitLensTable(result);
            } else {
                this.showError(result.error || 'Logit lens analysis failed');
            }
        } catch (error) {
            console.error('Error during logit lens analysis:', error);
            this.showError('Failed to connect to server for logit lens analysis');
        } finally {
            button.disabled = false;
            spinner.classList.add('d-none');
            this.isPredicting = false;
        }
    }

    renderLogitLensTable(result) {
        const { all_top_tokens, token_labels, num_layers, image_token_range, grid_info, image_url } = result;
        const numPositions = token_labels.length;

        // Set up image for patch highlighting
        const img = document.getElementById('logitLensImage');
        const canvas = document.getElementById('logitLensOverlay');
        const imgContainer = document.getElementById('logitLensImageContainer');

        if (image_url && image_token_range.start >= 0) {
            img.src = image_url;
            imgContainer.style.display = '';
        } else {
            imgContainer.style.display = 'none';
        }

        // Determine which layers to show (subsample if too many)
        const maxLayerCols = 60;
        let layerIndices = [];
        if (num_layers <= maxLayerCols) {
            for (let i = 0; i < num_layers; i++) layerIndices.push(i);
        } else {
            const step = (num_layers - 1) / (maxLayerCols - 1);
            for (let i = 0; i < maxLayerCols; i++) {
                layerIndices.push(Math.round(i * step));
            }
        }

        const imgStart = image_token_range.start;
        const imgEnd = image_token_range.end;
        let positionIndices = [];
        for (let i = 0; i < numPositions; i++) {
            positionIndices.push(i);
        }

        // Build the table
        const wrapper = document.getElementById('logitLensTableWrapper');
        const table = document.createElement('table');
        table.className = 'logit-lens-table';
        table.style.cssText = 'border-collapse: separate; border-spacing: 0; font-size: 11px; font-family: monospace;';

        // Header row
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        const cornerTh = document.createElement('th');
        cornerTh.textContent = 'Pos / Layer';
        cornerTh.style.cssText = 'position: sticky; top: 0; left: 0; z-index: 3; background: #f8f9fa; padding: 4px 6px; border: 1px solid #dee2e6; min-width: 100px;';
        headerRow.appendChild(cornerTh);

        for (const li of layerIndices) {
            const th = document.createElement('th');
            th.textContent = `L${li}`;
            th.style.cssText = 'position: sticky; top: 0; z-index: 2; background: #f8f9fa; padding: 4px 6px; border: 1px solid #dee2e6; white-space: nowrap; text-align: center;';
            headerRow.appendChild(th);
        }
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');

        // Tooltip element (shared)
        let tooltip = document.getElementById('logitLensTooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'logitLensTooltip';
            tooltip.style.cssText = 'display: none; position: fixed; z-index: 9999; background: #222; color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 12px; font-family: monospace; pointer-events: none; max-width: 300px; white-space: pre-line; box-shadow: 0 2px 8px rgba(0,0,0,0.3);';
            document.body.appendChild(tooltip);
        }

        // Locked patch state
        let lockedPatchIdx = null;

        const highlightPatch = (patchIdx) => {
            if (!img.naturalWidth || imgContainer.style.display === 'none') return;
            const rect = img.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (patchIdx === null || patchIdx === undefined) return;

            const gw = grid_info.merged_grid_w;
            const gh = grid_info.merged_grid_h;
            const row = Math.floor(patchIdx / gw);
            const col = patchIdx % gw;
            const epatch = grid_info.effective_patch_size;

            const scaleX = rect.width / grid_info.original_width;
            const scaleY = rect.height / grid_info.original_height;
            const imgScaleX = grid_info.original_width / grid_info.resized_width;
            const imgScaleY = grid_info.original_height / grid_info.resized_height;

            const x = col * epatch * imgScaleX * scaleX;
            const y = row * epatch * imgScaleY * scaleY;
            const w = epatch * imgScaleX * scaleX;
            const h = epatch * imgScaleY * scaleY;

            // Dim everything
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            // Clear the highlighted patch
            ctx.clearRect(x, y, w, h);
            // Border
            ctx.strokeStyle = '#ff0';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
        };

        for (const pi of positionIndices) {
            const tr = document.createElement('tr');
            const isImageToken = imgStart >= 0 && pi >= imgStart && pi <= imgEnd;
            const patchIdx = isImageToken ? (pi - imgStart) : null;

            // Row header (sticky left)
            const rowTh = document.createElement('th');
            const label = token_labels[pi];
            rowTh.textContent = `${pi}: ${label}`;
            rowTh.title = `Position ${pi}`;
            rowTh.style.cssText = 'position: sticky; left: 0; z-index: 1; background: ' + (isImageToken ? '#e8f4fd' : '#f8f9fa') + '; padding: 4px 6px; border: 1px solid #dee2e6; white-space: nowrap; cursor: default;';

            if (isImageToken) {
                rowTh.style.cursor = 'pointer';
                rowTh.addEventListener('mouseenter', () => {
                    if (lockedPatchIdx === null) highlightPatch(patchIdx);
                });
                rowTh.addEventListener('mouseleave', () => {
                    if (lockedPatchIdx === null) highlightPatch(null);
                });
                rowTh.addEventListener('click', () => {
                    if (lockedPatchIdx === patchIdx) {
                        lockedPatchIdx = null;
                        highlightPatch(null);
                    } else {
                        lockedPatchIdx = patchIdx;
                        highlightPatch(patchIdx);
                    }
                });
            }
            tr.appendChild(rowTh);

            // Data cells
            for (const li of layerIndices) {
                const td = document.createElement('td');
                const cellData = all_top_tokens[li][pi]; // [[token, prob], ...]
                if (cellData && cellData.length > 0) {
                    const topToken = cellData[0][0];
                    td.textContent = topToken.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
                    td.dataset.prob = cellData[0][1];
                }
                td.style.cssText += '; padding: 3px 5px; border: 1px solid #dee2e6; white-space: nowrap; text-align: center; max-width: 80px; overflow: hidden; text-overflow: ellipsis; cursor: default;';

                // Hover tooltip showing top-5
                td.addEventListener('mouseenter', (e) => {
                    if (!cellData || cellData.length === 0) return;
                    let lines = `Layer ${li}, Pos ${pi} (${token_labels[pi]})\n`;
                    for (let k = 0; k < cellData.length; k++) {
                        const tok = cellData[k][0].replace(/\n/g, '\\n').replace(/\t/g, '\\t');
                        const prob = (parseFloat(cellData[k][1]) * 100).toFixed(2);
                        lines += `${k + 1}. "${tok}" ${prob}%\n`;
                    }
                    tooltip.textContent = lines;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX + 12) + 'px';
                    tooltip.style.top = (e.clientY + 12) + 'px';
                });
                td.addEventListener('mousemove', (e) => {
                    tooltip.style.left = (e.clientX + 12) + 'px';
                    tooltip.style.top = (e.clientY + 12) + 'px';
                });
                td.addEventListener('mouseleave', () => {
                    tooltip.style.display = 'none';
                });

                // Image patch highlight on hover for image token rows
                if (isImageToken) {
                    td.addEventListener('mouseenter', () => {
                        if (lockedPatchIdx === null) highlightPatch(patchIdx);
                    });
                    td.addEventListener('mouseleave', () => {
                        if (lockedPatchIdx === null) highlightPatch(null);
                    });
                }

                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);

        wrapper.innerHTML = '';
        wrapper.appendChild(table);

        // Shading logic
        const applyShading = (enabled) => {
            const cells = table.querySelectorAll('td[data-prob]');
            cells.forEach(td => {
                if (enabled) {
                    const prob = parseFloat(td.dataset.prob);
                    const alpha = Math.min(prob, 1.0);
                    const r = Math.round(255 - alpha * 200);
                    const g = Math.round(255 - alpha * 50);
                    const b = Math.round(255 - alpha * 200);
                    td.style.backgroundColor = `rgb(${r},${g},${b})`;
                    td.style.color = alpha > 0.6 ? '#fff' : '';
                } else {
                    td.style.backgroundColor = '';
                    td.style.color = '';
                }
            });
        };

        const toggle = document.getElementById('logitLensShadingToggle');
        applyShading(toggle.checked);
        toggle.addEventListener('change', () => applyShading(toggle.checked));
    }

    setupDlaAnalysis(resultsContent, topTokens) {
        const dlaTokenSelector = resultsContent.querySelector('#dlaTokenSelector');
        const dlaComponentSelector = resultsContent.querySelector('#dlaComponentSelector');
        const dlaBtn = resultsContent.querySelector('#analyzeDlaBtn');

        if (!dlaTokenSelector || !topTokens || !dlaBtn) {
            return;
        }

        // Populate DLA token selector dropdown
        dlaTokenSelector.innerHTML = '';
        topTokens.forEach((token, index) => {
            const option = document.createElement('option');
            option.value = index;
            // Replace whitespace with underscores for display
            const displayToken = token.token.replace(/\s/g, '_');
            option.textContent = `${displayToken} (${(token.probability * 100).toFixed(2)}%)`;
            dlaTokenSelector.appendChild(option);
        });

        // Store data for DLA analysis
        dlaTokenSelector.dataset.topTokens = JSON.stringify(topTokens);

        // Setup enhanced dropdown controls for token selector
        this.setupEnhancedDropdown({
            selector: dlaTokenSelector,
            decrementBtn: resultsContent.querySelector('#dlaTokenDecrement'),
            incrementBtn: resultsContent.querySelector('#dlaTokenIncrement'),
            minValue: 0,
            maxValue: topTokens.length - 1,
            onChange: (value) => {
                this.handleDlaAnalysis();
            }
        });

        // Setup enhanced dropdown controls for component selector
        this.setupEnhancedDropdown({
            selector: dlaComponentSelector,
            decrementBtn: resultsContent.querySelector('#dlaComponentDecrement'),
            incrementBtn: resultsContent.querySelector('#dlaComponentIncrement'),
            minValue: 0,
            maxValue: dlaComponentSelector.options.length - 1,
            onChange: (value) => {
                this.handleDlaAnalysis();
            }
        });

        // Add event listener to DLA button in this specific instance
        dlaBtn.addEventListener('click', () => {
            this.handleDlaAnalysis();
        });
    }
    
    async handleDlaAnalysis() {
        if (!this.modelReady || this.isPredicting) {
            return;
        }

        // Check for valid input based on mode
        if (this.inputMode === 'upload' && !this.uploadedImage) {
            return;
        }
        if (this.inputMode === 'generate' && !this.generatedImage) {
            return;
        }

        this.isPredicting = true;
        const button = document.getElementById('analyzeDlaBtn');
        const spinner = document.getElementById('dlaSpinner');

        // Update UI
        button.disabled = true;
        spinner.classList.remove('d-none');

        // Determine filename based on mode
        const imageFilename = this.inputMode === 'upload'
            ? this.uploadedImage.filename
            : this.generatedImage.filename;

        const formData = {
            filename: imageFilename,
            task: document.getElementById('task').value,
            assistant_prefill: document.getElementById('assistantPrefill').value
        };
        
        try {
            const response = await fetch('/api/dla', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Store DLA data globally for chart updates
                this.dlaData = result.dla_data;
                this.dlaTopTokens = result.top_tokens;
                
                // Update DLA chart with default settings
                this.updateDlaChart();
                
                // Add event listeners for selector changes
                const tokenSelector = document.getElementById('dlaTokenSelector');
                const componentSelector = document.getElementById('dlaComponentSelector');
                
                tokenSelector.addEventListener('change', () => {
                    this.updateDlaChart();
                });
                
                componentSelector.addEventListener('change', () => {
                    this.updateDlaChart();
                });
                
            } else {
                this.showError(result.error || 'DLA analysis failed');
            }
        } catch (error) {
            console.error('Error during DLA analysis:', error);
            this.showError('Failed to connect to server for DLA analysis');
        } finally {
            // Reset UI
            button.disabled = false;
            spinner.classList.add('d-none');
            this.isPredicting = false;
        }
    }
    
    updateDlaChart() {
        if (!this.dlaData || !this.dlaTopTokens) {
            console.warn('No DLA data available');
            return;
        }
        
        const tokenSelector = document.getElementById('dlaTokenSelector');
        const componentSelector = document.getElementById('dlaComponentSelector');
        const chartContainer = document.getElementById('dlaChart');
        
        if (!tokenSelector || !componentSelector || !chartContainer) {
            console.warn('DLA UI elements not found');
            return;
        }
        
        const selectedTokenIndex = parseInt(tokenSelector.value);
        const componentType = componentSelector.value;
        const selectedToken = this.dlaTopTokens[selectedTokenIndex];
        const displayToken = selectedToken.token.replace(/\s/g, '_');
        
        // Check if Plotly is available
        if (typeof Plotly === 'undefined') {
            console.error('Plotly is not loaded');
            chartContainer.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle me-2"></i>Chart library not loaded. Please refresh the page.</div>';
            return;
        }
        
        if (componentType === 'att_mlp') {
            this.createAttentionMlpHeatmap(chartContainer, selectedTokenIndex, displayToken);
        } else {
            this.createAttentionHeadsHeatmap(chartContainer, selectedTokenIndex, displayToken);
        }
    }
    
    createAttentionMlpHeatmap(chartContainer, tokenIndex, displayToken) {
        const numLayers = this.dlaData.num_layers;
        const attContribs = this.dlaData.layer_att_contributions;
        const mlpContribs = this.dlaData.mlp_contributions;
        
        // Create heatmap with X-axis as Att-MLP pairs and Y-axis as layers
        const zValues = [];
        const yLabels = [];
        const xLabels = ['Attention', 'MLP'];
        
        // Build data: each row is a layer, each column is Att or MLP
        for (let layer = 0; layer < numLayers; layer++) {
            const layerValues = [
                attContribs[layer][tokenIndex],  // Attention contribution
                mlpContribs[layer][tokenIndex]   // MLP contribution
            ];
            zValues.push(layerValues);
            yLabels.push(`Layer ${layer}`);
        }
        
        const data = [{
            z: zValues,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,  // Center colormap at 0
            hovertemplate: 
                '<b>Layer:</b> %{y}<br>' +
                '<b>Component:</b> %{x}<br>' +
                '<b>Token:</b> ' + displayToken + '<br>' +
                '<b>Contribution:</b> %{z:.4f}<extra></extra>',
            showscale: true,
            colorbar: {
                title: 'Contribution'
            }
        }];
        
        const layout = {
            title: {
                text: `Direct Logit Attribution: "${displayToken}" (Attention + MLP)`,
                x: 0.5,
                font: { size: 16 }
            },
            xaxis: {
                title: 'Component Type',
                tickvals: [0, 1],
                ticktext: xLabels,
                showgrid: false
            },
            yaxis: {
                title: 'Layer',
                tickvals: Array.from({length: yLabels.length}, (_, i) => i),
                ticktext: yLabels,
                showgrid: false
            },
            margin: {
                l: 80,
                r: 60,
                t: 60,
                b: 60
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(chartContainer, data, layout, config);
    }
    
    createAttentionHeadsHeatmap(chartContainer, tokenIndex, displayToken) {
        const numLayers = this.dlaData.num_layers;
        const numHeads = this.dlaData.num_heads;
        const headContribs = this.dlaData.head_contributions;
        
        // Create 2D array for heatmap: [layers][heads]
        const zValues = [];
        
        for (let layer = 0; layer < numLayers; layer++) {
            const layerHeads = [];
            for (let head = 0; head < numHeads; head++) {
                layerHeads.push(headContribs[layer][head][tokenIndex]);
            }
            zValues.push(layerHeads);
        }
        
        const data = [{
            z: zValues,
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,  // Center colormap at 0
            hovertemplate: 
                '<b>Layer:</b> %{y}<br>' +
                '<b>Head:</b> %{x}<br>' +
                '<b>Token:</b> ' + displayToken + '<br>' +
                '<b>Contribution:</b> %{z:.4f}<extra></extra>',
            showscale: true,
            colorbar: {
                title: 'Contribution'
            }
        }];
        
        const layout = {
            title: {
                text: `Direct Logit Attribution: "${displayToken}" (Attention Heads)`,
                x: 0.5,
                font: { size: 16 }
            },
            xaxis: {
                title: 'Head Number',
                showgrid: false
            },
            yaxis: {
                title: 'Layer Number',
                showgrid: false
            },
            margin: {
                l: 60,
                r: 60,
                t: 60,
                b: 60
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(chartContainer, data, layout, config);
    }
    
    addImageInteractions(img, canvas, patches) {
        // Add hover functionality to image
        const handleImageHover = (event) => {
            const rect = img.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Convert canvas coordinates to image coordinates
            const imageX = (x / rect.width) * img.naturalWidth;
            const imageY = (y / rect.height) * img.naturalHeight;
            
            // Find which patch this corresponds to
            let hoveredPatch = null;
            for (const patch of patches) {
                const [x1, y1, x2, y2] = patch.bbox;
                if (imageX >= x1 && imageX <= x2 && imageY >= y1 && imageY <= y2) {
                    hoveredPatch = patch;
                    break;
                }
            }
            
            if (hoveredPatch) {
                // Calculate corresponding token index (if we have image token range data)
                let tokenIndexInfo = '';
                if (this.currentAttentionData && this.currentAttentionData.image_token_range) {
                    const tokenIndex = this.currentAttentionData.image_token_range.start + hoveredPatch.patch_idx;
                    tokenIndexInfo = `, Token Index: ${tokenIndex}`;
                }
                
                // Update info text
                document.getElementById('attentionImageInfo').textContent = 
                    `Hovering Image Patch [Patch: ${hoveredPatch.patch_idx}${tokenIndexInfo}]: Attention ${hoveredPatch.attention.toFixed(4)}`;
                
                // Highlight corresponding position in heatmap (if possible)
                // This would need Plotly hover events to be properly implemented
                
                // Temporarily highlight the patch
                this.highlightImagePatch(canvas, hoveredPatch, patches);
            } else {
                document.getElementById('attentionImageInfo').textContent = 
                    'Hover over image patches to see attention weights';
                // Redraw normal overlay
                this.drawAttentionOverlay(img, canvas, patches);
            }
        };
        
        canvas.addEventListener('mousemove', handleImageHover);
        canvas.addEventListener('mouseleave', () => {
            document.getElementById('attentionImageInfo').textContent = 
                'Hover over image patches to see attention weights';
            this.drawAttentionOverlay(img, canvas, patches);
        });
    }
    
    highlightImagePatch(canvas, hoveredPatch, allPatches) {
        const ctx = canvas.getContext('2d');
        const img = document.getElementById('attentionImage');
        
        // Redraw all patches first
        this.drawAttentionOverlay(img, canvas, allPatches);
        
        // Then highlight the specific patch
        const [x1, y1, x2, y2] = hoveredPatch.bbox;
        const rect = img.getBoundingClientRect();
        const scaleX = rect.width / img.naturalWidth;
        const scaleY = rect.height / img.naturalHeight;
        
        const canvasX1 = x1 * scaleX;
        const canvasY1 = y1 * scaleY;
        const canvasX2 = x2 * scaleX;
        const canvasY2 = y2 * scaleY;
        
        // Draw highlighted border
        ctx.strokeStyle = '#ffff00'; // Yellow highlight
        ctx.lineWidth = 4;
        ctx.strokeRect(canvasX1, canvasY1, canvasX2 - canvasX1, canvasY2 - canvasY1);
    }
    
    highlightImagePatchByIndex(patchIndex) {
        // Find the patch by its index and highlight it
        if (!this.currentAttentionData || !this.currentAttentionData.patches) return;
        
        const targetPatch = this.currentAttentionData.patches.find(patch => patch.patch_idx === patchIndex);
        if (targetPatch) {
            const img = document.getElementById('attentionImage');
            const canvas = document.getElementById('attentionOverlayCanvas');
            if (img && canvas) {
                this.highlightImagePatch(canvas, targetPatch, this.currentAttentionData.patches);
            }
        }
    }
    
    addHeatmapInteractions(chartContainer, attentionData) {
        // Add hover events to the heatmap to sync with other visualizations
        chartContainer.addEventListener('plotly_hover', (data) => {
            if (data.points && data.points.length > 0) {
                const point = data.points[0];
                const xIndex = point.pointNumber[1]; // Column index
                
                // Get the token at this position (now in correct sequence order)
                if (attentionData.text_tokens && xIndex < attentionData.text_tokens.length) {
                    const token = attentionData.text_tokens[xIndex];
                    
                    if (token.is_image_token) {
                        // This is an image token
                        document.getElementById('attentionImageInfo').textContent = 
                            `Heatmap hover: Image Patch ${token.patch_idx} (Token ${token.position}), Attention ${token.attention.toFixed(4)}`;
                        
                        // Highlight the patch in the image
                        this.highlightImagePatchByIndex(token.patch_idx);
                    } else {
                        // This is a text token
                        document.getElementById('attentionImageInfo').textContent = 
                            `Heatmap hover: Text Token "${token.text}" (Index ${token.position}), Attention ${token.attention.toFixed(4)}`;
                        
                        // Highlight the token in the text
                        this.highlightTextToken(xIndex);
                    }
                }
            }
        });
        
        chartContainer.addEventListener('plotly_unhover', () => {
            // Reset highlights
            document.getElementById('attentionImageInfo').textContent = 
                'Hover over image patches to see attention weights';
            
            // Redraw normal overlays
            const img = document.getElementById('attentionImage');
            const canvas = document.getElementById('attentionOverlayCanvas');
            if (img && canvas && this.currentAttentionData) {
                this.drawAttentionOverlay(img, canvas, this.currentAttentionData.patches);
            }
            
            // Reset text highlighting
            this.resetTextHighlighting();
        });
    }
    
    highlightTextToken(tokenIndex) {
        const container = document.getElementById('attentionTextContainer');
        if (!container) return;
        
        const tokens = container.querySelectorAll('.attention-token');
        
        // Reset all tokens
        tokens.forEach((token, idx) => {
            if (idx === tokenIndex) {
                // Highlight this token
                token.style.outline = '3px solid #ffff00';
                token.style.outlineOffset = '2px';
            } else {
                token.style.outline = 'none';
            }
        });
    }
    
    resetTextHighlighting() {
        const container = document.getElementById('attentionTextContainer');
        if (!container) return;

        const tokens = container.querySelectorAll('.attention-token');
        tokens.forEach(token => {
            token.style.outline = 'none';
        });
    }

    setupEnhancedDropdown(config) {
        const { selector, decrementBtn, incrementBtn, minValue, maxValue, onChange } = config;

        if (!selector || !decrementBtn || !incrementBtn) {
            console.log('setupEnhancedDropdown: Missing required elements', { selector, decrementBtn, incrementBtn });
            return;
        }

        console.log('Setting up enhanced dropdown', selector.id);

        // Update button states based on current selection
        const updateButtonStates = () => {
            const currentValue = parseInt(selector.value);
            const hasSelection = selector.value !== '' && !isNaN(currentValue);

            decrementBtn.disabled = !hasSelection || currentValue <= minValue;
            incrementBtn.disabled = !hasSelection || currentValue >= maxValue;
        };

        // Handle decrement button click
        decrementBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Decrement clicked for', selector.id);

            let currentValue = parseInt(selector.value);

            // If no current selection, start from the last valid option
            if (isNaN(currentValue) || selector.value === '') {
                currentValue = maxValue + 1; // Will be decremented to maxValue
            }

            if (currentValue > minValue) {
                const newValue = currentValue - 1;
                selector.value = newValue;

                // Trigger change event manually
                const changeEvent = new Event('change', { bubbles: true });
                selector.dispatchEvent(changeEvent);

                if (onChange) {
                    onChange(newValue);
                }
            }
            updateButtonStates();
        });

        // Handle increment button click
        incrementBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Increment clicked for', selector.id);

            let currentValue = parseInt(selector.value);

            // If no current selection, start from before the first valid option
            if (isNaN(currentValue) || selector.value === '') {
                currentValue = minValue - 1; // Will be incremented to minValue
            }

            if (currentValue < maxValue) {
                const newValue = currentValue + 1;
                selector.value = newValue;

                // Trigger change event manually
                const changeEvent = new Event('change', { bubbles: true });
                selector.dispatchEvent(changeEvent);

                if (onChange) {
                    onChange(newValue);
                }
            }
            updateButtonStates();
        });

        // Handle dropdown change to update button states
        selector.addEventListener('change', updateButtonStates);

        // Handle keyboard navigation
        selector.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                e.preventDefault();
                decrementBtn.click();
            } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                e.preventDefault();
                incrementBtn.click();
            }
        });

        // Enable buttons initially if we have valid range
        decrementBtn.disabled = false;
        incrementBtn.disabled = false;

        // Initial state update
        updateButtonStates();
    }
    
    setupAttentionExplorer(resultsContent, modelInfo) {
        const layerSelector = resultsContent.querySelector('#attentionLayerSelector');
        const headSelector = resultsContent.querySelector('#attentionHeadSelector');
        const analyzeBtn = resultsContent.querySelector('#analyzeAttentionBtn');

        if (!layerSelector || !headSelector || !analyzeBtn) {
            return;
        }

        // Populate layer selector
        layerSelector.innerHTML = '<option value="">Select a layer...</option>';
        for (let i = 0; i < modelInfo.num_layers; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `Layer ${i}`;
            layerSelector.appendChild(option);
        }

        // Populate head selector
        const updateHeadSelector = (numHeads) => {
            headSelector.innerHTML = '<option value="">Select a head...</option>';
            for (let i = 0; i < numHeads; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Head ${i}`;
                headSelector.appendChild(option);
            }
        };

        // Initialize with the model's number of heads
        updateHeadSelector(modelInfo.num_heads);

        // Setup enhanced dropdown controls for layer selector
        this.setupEnhancedDropdown({
            selector: layerSelector,
            decrementBtn: resultsContent.querySelector('#attentionLayerDecrement'),
            incrementBtn: resultsContent.querySelector('#attentionLayerIncrement'),
            minValue: 0,
            maxValue: modelInfo.num_layers - 1,
            onChange: (value) => {
                this.handleAttentionAnalysis(value, headSelector.value);
            }
        });

        // Setup enhanced dropdown controls for head selector
        this.setupEnhancedDropdown({
            selector: headSelector,
            decrementBtn: resultsContent.querySelector('#attentionHeadDecrement'),
            incrementBtn: resultsContent.querySelector('#attentionHeadIncrement'),
            minValue: 0,
            maxValue: modelInfo.num_heads - 1,
            onChange: (value) => {
                // Use instant head switching if we have cached data
                if (this.allHeadsAttentionData && this.currentLayer !== undefined) {
                    this.switchToHead(parseInt(value));
                } else {
                    this.handleAttentionAnalysis(layerSelector.value, value);
                }
            }
        });

        // Enable/disable analyze button based on selection
        const checkSelections = () => {
            const layerSelected = layerSelector.value !== '';
            const headSelected = headSelector.value !== '';
            analyzeBtn.disabled = !(layerSelected && headSelected);
        };

        layerSelector.addEventListener('change', checkSelections);
        headSelector.addEventListener('change', checkSelections);

        // Add event listener for analyze button (only for layer changes)
        analyzeBtn.addEventListener('click', () => {
            this.handleAttentionAnalysis(layerSelector.value, headSelector.value);
        });

        // Store selectors for instant head switching setup
        this.attentionLayerSelector = layerSelector;
        this.attentionHeadSelector = headSelector;
    }
    
    async handleAttentionAnalysis(layer, head) {
        if (!this.modelReady || this.isPredicting) {
            return;
        }

        // Check for valid input based on mode
        if (this.inputMode === 'upload' && !this.uploadedImage) {
            return;
        }
        if (this.inputMode === 'generate' && !this.generatedImage) {
            return;
        }

        this.isPredicting = true;
        const button = document.getElementById('analyzeAttentionBtn');
        const spinner = document.getElementById('attentionSpinner');
        const visualization = document.getElementById('attentionVisualization');

        // Update UI
        button.disabled = true;
        spinner.classList.remove('d-none');

        // Determine filename based on mode
        const imageFilename = this.inputMode === 'upload'
            ? this.uploadedImage.filename
            : this.generatedImage.filename;

        const formData = {
            filename: imageFilename,
            task: document.getElementById('task').value,
            assistant_prefill: document.getElementById('assistantPrefill').value,
            layer: parseInt(layer),
            head: parseInt(head)
        };
        
        try {
            const response = await fetch('/api/attention', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Update current view info
                document.getElementById('attentionCurrentView').textContent = `Layer ${layer}, Head ${head}`;
                document.getElementById('attentionQueryInfo').textContent = result.token_position;
                
                // Show visualization container
                visualization.classList.remove('d-none');
                
                // Store attention data for interactions (current head)
                this.currentAttentionData = result.attention_data;
                
                // Cache ALL heads data for instant switching
                this.allHeadsAttentionData = result.attention_data.all_heads;
                this.currentLayer = parseInt(layer);
                this.currentImageUrl = result.image_url;
                
                // Update all visualizations
                this.updateAttentionImage(result.image_url, result.attention_data);
                this.updateAttentionText(result.attention_data.text_tokens);
                this.updateAttentionHeatmap(result.attention_data);
                
                // Enable instant head switching by updating the head selector behavior
                this.setupInstantHeadSwitching();
                
            } else {
                this.showError(result.error || 'Attention analysis failed');
            }
        } catch (error) {
            console.error('Error during attention analysis:', error);
            this.showError('Failed to connect to server for attention analysis');
        } finally {
            // Reset UI
            button.disabled = false;
            spinner.classList.add('d-none');
            this.isPredicting = false;
        }
    }
    
    updateAttentionImage(imageUrl, attentionData) {
        const img = document.getElementById('attentionImage');
        const canvas = document.getElementById('attentionOverlayCanvas');
        
        if (!img || !canvas) return;
        
        img.src = imageUrl;
        img.onload = () => {
            this.drawAttentionOverlay(img, canvas, attentionData.patches);
            this.addImageInteractions(img, canvas, attentionData.patches);
        };
    }
    
    drawAttentionOverlay(img, canvas, patches) {
        const rect = img.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!patches || patches.length === 0) return;
        
        // Calculate attention statistics for 2 standard deviation threshold
        const attentions = patches.map(p => p.attention);
        const meanAtt = attentions.reduce((sum, att) => sum + att, 0) / attentions.length;
        const variance = attentions.reduce((sum, att) => sum + Math.pow(att - meanAtt, 2), 0) / attentions.length;
        const stdDev = Math.sqrt(variance);
        const threshold = meanAtt + (2 * stdDev);
        
        // Find min/max for alpha calculation
        const minAtt = Math.min(...attentions);
        const maxAtt = Math.max(...attentions);
        const range = maxAtt - minAtt;
        
        // Calculate scale factors
        const scaleX = rect.width / img.naturalWidth;
        const scaleY = rect.height / img.naturalHeight;
        
        // Draw attention rectangles for patches more than 2 standard deviations above mean
        patches.forEach(patch => {
            if (range > 0 && patch.attention > threshold) {
                // Normalize attention to 0-1 range for alpha calculation
                const normalizedAtt = (patch.attention - minAtt) / range;
                const [x1, y1, x2, y2] = patch.bbox;
                
                // Scale coordinates to canvas size
                const canvasX1 = x1 * scaleX;
                const canvasY1 = y1 * scaleY;
                const canvasX2 = x2 * scaleX;
                const canvasY2 = y2 * scaleY;
                
                // Draw rectangle with attention-based alpha
                const alpha = 0.3 + (normalizedAtt * 0.4); // 0.3 to 0.7 alpha range
                ctx.fillStyle = `rgba(255, 0, 0, ${alpha})`;
                ctx.fillRect(canvasX1, canvasY1, canvasX2 - canvasX1, canvasY2 - canvasY1);
                
                // Draw border
                ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
                ctx.lineWidth = 2;
                ctx.strokeRect(canvasX1, canvasY1, canvasX2 - canvasX1, canvasY2 - canvasY1);
            }
        });
    }
    
    updateAttentionText(textTokens) {
        const container = document.getElementById('attentionTextContainer');
        if (!container || !textTokens) return;
        
        // Find min/max attention for normalization
        const attentions = textTokens.map(t => t.attention);
        const minAtt = Math.min(...attentions);
        const maxAtt = Math.max(...attentions);
        const range = maxAtt - minAtt;
        
        // Clear container and add token spans
        container.innerHTML = '';
        
        textTokens.forEach((token, idx) => {
            const span = document.createElement('span');
            span.textContent = token.text;
            span.className = 'attention-token';
            span.style.padding = '2px 1px';
            span.style.cursor = 'pointer';
            span.style.borderRadius = '2px';
            span.style.margin = '0 1px';
            
            // Special styling for image tokens
            if (token.is_image_token) {
                span.classList.add('image-token');
                span.style.backgroundColor = 'rgba(255, 165, 0, 0.3)'; // Orange background for image tokens
                span.style.border = '1px solid orange';
                span.title = `Image Token | Token Index: ${token.position} | Image Patch: ${token.patch_idx} | Text: "${token.text}" | Attention: ${token.attention.toFixed(4)}`;
            } else {
                // Set background based on attention weight for regular text tokens
                if (range > 0) {
                    const normalizedAtt = (token.attention - minAtt) / range;
                    // Apply gamma correction to make differences more visible
                    const gamma = 0.5;
                    const adjustedAtt = Math.pow(normalizedAtt, gamma);
                    const alpha = 0.1 + (adjustedAtt * 0.6); // 0.1 to 0.7 alpha range
                    span.style.backgroundColor = `rgba(0, 100, 200, ${alpha})`;
                    span.style.color = adjustedAtt > 0.5 ? 'white' : 'black';
                }
                span.title = `Text Token | Token Index: ${token.position} | Text: "${token.text}" | Attention: ${token.attention.toFixed(4)}`;
            }
            
            // Add hover effects
            span.addEventListener('mouseenter', () => {
                span.style.outline = '2px solid #0d6efd';
                
                if (token.is_image_token) {
                    // For image tokens, show info and highlight corresponding image patch
                    document.getElementById('attentionImageInfo').textContent = 
                        `Hovering Image Token [Index: ${token.position}, Patch: ${token.patch_idx}]: "${token.text}" (attention: ${token.attention.toFixed(4)})`;
                    
                    // Highlight the corresponding image patch
                    this.highlightImagePatchByIndex(token.patch_idx);
                } else {
                    // For regular text tokens
                    document.getElementById('attentionImageInfo').textContent = 
                        `Hovering Text Token [Index: ${token.position}]: "${token.text}" (attention: ${token.attention.toFixed(4)})`;
                }
            });
            
            span.addEventListener('mouseleave', () => {
                span.style.outline = 'none';
                document.getElementById('attentionImageInfo').textContent = 
                    'Hover over image patches to see attention weights';
                
                // Reset image overlay to normal state if we were highlighting an image token
                if (token.is_image_token) {
                    const img = document.getElementById('attentionImage');
                    const canvas = document.getElementById('attentionOverlayCanvas');
                    if (img && canvas && this.currentAttentionData) {
                        this.drawAttentionOverlay(img, canvas, this.currentAttentionData.patches);
                    }
                }
            });
            
            container.appendChild(span);
        });
    }
    
    updateAttentionHeatmap(attentionData) {
        const chartContainer = document.getElementById('attentionHeatmap');
        if (!chartContainer) return;
        
        // Check if Plotly is available
        if (typeof Plotly === 'undefined') {
            console.error('Plotly is not loaded');
            chartContainer.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle me-2"></i>Chart library not loaded. Please refresh the page.</div>';
            return;
        }
        
        // Prepare data for heatmap
        const textTokens = attentionData.text_tokens || [];
        
        if (textTokens.length === 0) {
            chartContainer.innerHTML = '<div class="alert alert-warning">No attention data available</div>';
            return;
        }
        
        // Create labels and attention values in the ACTUAL token sequence order
        const xLabels = [];
        const attentionValues = [];
        
        // Process tokens in their actual sequence order (not separating patches and text)
        textTokens.forEach((token, idx) => {
            if (token.is_image_token) {
                // This is an image token
                xLabels.push(`img:${token.patch_idx}`);
            } else {
                // This is a text token
                const cleanToken = token.text.replace(/\s/g, '_').replace(/\n/g, '\\n');
                xLabels.push(`t:${token.position}:${cleanToken.slice(0, 8)}`);
            }
            attentionValues.push(token.attention);
        });
        
        // Create 2D array for heatmap (single row for the query token)
        const zValues = [attentionValues];
        const yLabels = ['Last Token Query'];
        
        const data = [{
            z: zValues,
            type: 'heatmap',
            colorscale: 'Viridis',
            hovertemplate: 
                '<b>Position:</b> %{x}<br>' +
                '<b>Query:</b> %{y}<br>' +
                '<b>Attention:</b> %{z:.4f}<extra></extra>',
            showscale: true,
            colorbar: {
                title: 'Attention Weight'
            }
        }];
        
        const layout = {
            title: {
                text: 'Attention Heatmap: Last Token Query',
                x: 0.5,
                font: { size: 14 }
            },
            xaxis: {
                title: 'Key Positions (Image Patches + Text Tokens)',
                tickangle: -45,
                tickfont: { size: 9 },
                showgrid: false
            },
            yaxis: {
                title: 'Query Token',
                showgrid: false
            },
            margin: {
                l: 80,
                r: 60,
                t: 60,
                b: 120
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(chartContainer, data, layout, config);
        
        // Add hover interactions for the heatmap
        this.addHeatmapInteractions(chartContainer, attentionData);
    }
    
    setupInstantHeadSwitching() {
        // Remove any existing head change listeners to avoid duplicates
        if (this.headSwitchListener) {
            this.attentionHeadSelector.removeEventListener('change', this.headSwitchListener);
        }
        
        // Add new head change listener that uses cached data
        this.headSwitchListener = (event) => {
            const selectedHead = parseInt(event.target.value);
            if (selectedHead >= 0 && this.allHeadsAttentionData) {
                this.switchToHead(selectedHead);
            }
        };
        
        this.attentionHeadSelector.addEventListener('change', this.headSwitchListener);
    }
    
    switchToHead(headIndex) {
        // Update current view info
        document.getElementById('attentionCurrentView').textContent = `Layer ${this.currentLayer}, Head ${headIndex}`;
        
        // Get data for this head from cached all heads data
        const headPatches = this.allHeadsAttentionData.patches[headIndex];
        const headTextTokens = this.allHeadsAttentionData.text_tokens[headIndex];
        
        // Create attention data structure for this head
        const headAttentionData = {
            patches: headPatches,
            text_tokens: headTextTokens,
            full_attention: this.allHeadsAttentionData.full_attention[headIndex],
            num_image_patches: this.currentAttentionData.num_image_patches,
            image_token_range: this.currentAttentionData.image_token_range,
            image_dimensions: this.currentAttentionData.image_dimensions
        };
        
        // Update current attention data
        this.currentAttentionData = headAttentionData;
        
        // Update all visualizations with the new head data
        this.updateAttentionImage(this.currentImageUrl, headAttentionData);
        this.updateAttentionText(headTextTokens);
        this.updateAttentionHeatmap(headAttentionData);
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DemoApp();
});
