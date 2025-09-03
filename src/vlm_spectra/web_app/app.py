import os
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from vlm_spectra.web_app.model_manager import ModelManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

model_manager = ModelManager()

@app.route('/')
def index():
    """Main demo interface"""
    return render_template('index.html')

@app.route('/api/model/status')
def model_status():
    """Get current model loading status"""
    return jsonify({
        'loading': model_manager.is_loading,
        'ready': model_manager.is_ready,
        'error': model_manager.error_message
    })

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/predict', methods=['POST'])
def predict_from_uploaded_image():
    """Run model prediction on uploaded image"""
    if not model_manager.is_ready:
        return jsonify({'error': 'Model not ready yet'}), 503
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        task = data.get('task', 'Click on the relevant element.')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        result = model_manager.predict_from_image(
            image_path=image_path,
            task=task
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/uploads/<filename>')
def serve_uploaded_image(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting demo server...")
    print("Model loading in background...")
    
    # Start model loading in background thread
    model_thread = threading.Thread(target=model_manager.load_model)
    model_thread.daemon = True
    model_thread.start()
    
    print("Demo will be available at http://localhost:55556")
    app.run(host='0.0.0.0', port=55556, debug=True)