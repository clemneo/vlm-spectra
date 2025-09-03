// Demo application JavaScript for image upload and prediction
class DemoApp {
    constructor() {
        this.modelReady = false;
        this.isPredicting = false;
        this.uploadedImage = null;
        
        this.initializeEventListeners();
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
            task: document.getElementById('task').value
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
    }
    
    displayForwardResults(result) {
        const container = document.getElementById('resultsContainer');
        const template = document.getElementById('forwardResultsTemplate');
        
        // Clone template
        const resultsContent = template.cloneNode(true);
        resultsContent.classList.remove('d-none');
        resultsContent.id = '';
        
        // Populate data (no image display)
        resultsContent.querySelector('#imageSizeForward').textContent = `${result.image_size[0]}×${result.image_size[1]}`;
        resultsContent.querySelector('#taskTextForward').textContent = result.task;
        resultsContent.querySelector('#tokenPosition').textContent = result.token_position || 'Next token';
        resultsContent.querySelector('#inferenceTimeForward').textContent = `${result.inference_time}s`;
        
        // Populate token predictions table
        const tbody = resultsContent.querySelector('#tokenPredictions');
        tbody.innerHTML = '';
        
        if (result.top_tokens) {
            result.top_tokens.forEach((token, index) => {
                const row = document.createElement('tr');
                // Replace whitespace characters with underscores for visibility
                const displayToken = token.token.replace(/\s/g, '_');
                row.innerHTML = `
                    <td class="fw-bold">${index + 1}</td>
                    <td><code>${this.escapeHtml(displayToken)}</code></td>
                    <td><span class="badge bg-primary">${(token.probability * 100).toFixed(2)}%</span></td>
                    <td class="font-monospace small">${token.logit.toFixed(4)}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        // Replace container content
        container.innerHTML = '';
        container.appendChild(resultsContent);
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