import os
import time
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from vlm_spectra.web_app.model_manager import ModelManager

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(script_dir, 'static'), template_folder=os.path.join(script_dir, 'templates'))
app.config['SECRET_KEY'] = 'demo-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(script_dir, 'static', 'uploads')
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
        'error': model_manager.error_message,
        'model_id': model_manager.current_model_id,
        'model_name': model_manager.get_current_model_name(),
        'model_label': model_manager.get_current_model_label(),
        'pending_model_id': model_manager.pending_model_id,
        'pending_model_label': model_manager.model_options.get(model_manager.pending_model_id, {}).get('label')
    })

@app.route('/api/model/options')
def model_options():
    """List available model options"""
    return jsonify(model_manager.get_model_options())

@app.route('/api/model/select', methods=['POST'])
def model_select():
    """Select and load a model"""
    data = request.get_json() or {}
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({'error': 'No model_id provided'}), 400

    try:
        status = model_manager.start_model_loading(model_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if status == "unavailable":
        return jsonify({'error': 'Model is not available in this environment'}), 400

    return jsonify({
        'status': status,
        'model_id': model_id,
        'model_label': model_manager.model_options[model_id]['label']
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
        assistant_prefill = data.get('assistant_prefill', '')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        result = model_manager.predict_from_image(
            image_path=image_path,
            task=task,
            assistant_prefill=assistant_prefill
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forward', methods=['POST'])
def forward_pass_analysis():
    """Run forward pass analysis on uploaded image"""
    if not model_manager.is_ready:
        return jsonify({'error': 'Model not ready yet'}), 503
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        task = data.get('task', 'Click on the relevant element.')
        assistant_prefill = data.get('assistant_prefill', '')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        result = model_manager.forward_pass_analysis(
            image_path=image_path,
            task=task,
            assistant_prefill=assistant_prefill
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dla', methods=['POST'])
def direct_logit_attribution_analysis():
    """Run direct logit attribution analysis on uploaded image"""
    if not model_manager.is_ready:
        return jsonify({'error': 'Model not ready yet'}), 503
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        task = data.get('task', 'Click on the relevant element.')
        assistant_prefill = data.get('assistant_prefill', '')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        result = model_manager.direct_logit_attribution(
            image_path=image_path,
            task=task,
            assistant_prefill=assistant_prefill
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attention', methods=['POST'])
def attention_analysis():
    """Run attention analysis for a specific layer and head"""
    if not model_manager.is_ready:
        return jsonify({'error': 'Model not ready yet'}), 503
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        task = data.get('task', 'Click on the relevant element.')
        assistant_prefill = data.get('assistant_prefill', '')
        layer = data.get('layer', 0)
        head = data.get('head', 0)
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Validate layer and head parameters
        try:
            layer = int(layer)
            head = int(head)
        except (ValueError, TypeError):
            return jsonify({'error': 'Layer and head must be integers'}), 400
        
        if layer < 0 or head < 0:
            return jsonify({'error': 'Layer and head must be non-negative'}), 400
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        result = model_manager.attention_analysis(
            image_path=image_path,
            task=task,
            layer=layer,
            head=head,
            assistant_prefill=assistant_prefill
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
    model_manager.start_model_loading()
    
    print("Demo will be available at http://localhost:55556")
    app.run(host='0.0.0.0', port=55556, debug=False)
