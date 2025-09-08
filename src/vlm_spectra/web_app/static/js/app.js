// Demo application JavaScript for image upload and prediction
class DemoApp {
    constructor() {
        this.modelReady = false;
        this.isPredicting = false;
        this.uploadedImage = null;
        
        this.initializeEventListeners();
        this.checkDependencies();
        this.checkModelStatus();
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
            document.getElementById('analyzeGenerationBtn').disabled = false;
            document.getElementById('analyzeForwardBtn').disabled = false;
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
                check: () => document.querySelectorAll('link[href*="fontawesome"]').length > 0 || document.querySelectorAll('script[src*="fontawesome"]').length > 0,
                elementId: 'fontawesomeStatus'
            }
        ];
        
        let allLoaded = true;
        const failedDeps = [];
        
        dependencies.forEach(dep => {
            const element = document.getElementById(dep.elementId);
            if (!element) return;
            
            if (dep.check()) {
                element.className = 'badge bg-success';
                element.innerHTML = `<i class="fas fa-check me-1"></i>${dep.name}`;
            } else {
                element.className = 'badge bg-danger';
                element.innerHTML = `<i class="fas fa-times me-1"></i>${dep.name}`;
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
        }
        
        // Recheck dependencies after a short delay in case they're still loading
        if (!allLoaded) {
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
            
            if (status.ready) {
                statusElement.className = 'alert alert-success';
                statusElement.innerHTML = '<i class="fas fa-check-circle me-2"></i>Model ready!';
                this.modelReady = true;
                if (this.uploadedImage) {
                    generationBtn.disabled = false;
                    forwardBtn.disabled = false;
                }
            } else if (status.error) {
                statusElement.className = 'alert alert-danger';
                statusElement.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i>Error: ${status.error}`;
                generationBtn.disabled = true;
                forwardBtn.disabled = true;
            } else {
                statusElement.className = 'alert alert-info';
                statusElement.innerHTML = '<i class="spinner-border spinner-border-sm me-2"></i>Model loading...';
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
        if (!this.modelReady || this.isPredicting || !this.uploadedImage) {
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
        
        const formData = {
            filename: this.uploadedImage.filename,
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
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DemoApp();
});