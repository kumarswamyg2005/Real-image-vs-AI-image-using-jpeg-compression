"""
Web App using Bitstream Forensics AI Detector
Bitstream forensics (DCT/quantization analysis) + Camera Signature Analysis
Accuracy: 97.2% (trained on 348k images, 2.6% false positive rate)
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from ensemble_detector import EnsembleAIDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Ensemble AI Detector
print("Initializing Bitstream AI Detector...")
print("Loading Random Forest model (trained on 348k images)...")
detector = EnsembleAIDetector()
print("Ready to accept requests!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save file with unique name
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Get prediction with details from ensemble
            result = detector.predict(filepath, return_details=True)
            
            # Format response with breakdown - convert all values to JSON-serializable types
            response = {
                'is_ai': bool(result['is_ai']),
                'label': str(result['label']),
                'confidence': float(result['confidence']),
                'ai_score': float(result['ai_probability'] * 100),
                'real_score': float(result['real_probability'] * 100),
                'model': str(result['model']),
                'method': str(result['method']),
                'image_url': f'/uploads/{unique_filename}',
                
                # Add model breakdown for transparency
                'breakdown': {
                    'bitstream': {
                        'ai_prob': float(result['model_breakdown']['bitstream']['ai_probability'] * 100),
                        'weight': float(result['model_breakdown']['bitstream']['weight'] * 100),
                        'contribution': float(result['model_breakdown']['bitstream']['contribution'] * 100)
                    },
                    'camera': {
                        'is_camera': bool(result['model_breakdown']['camera_signature']['is_camera_likely']),
                        'confidence': float(result['model_breakdown']['camera_signature']['confidence']),
                        'weight': float(result['model_breakdown']['camera_signature']['weight'] * 100),
                        'reasons': [str(r) for r in result['model_breakdown']['camera_signature']['reasons']]
                    }
                },
                'camera_override': bool(result.get('camera_override_applied', False)),
                'computational_photography': bool(result.get('computational_photography_detected', False)),
                'note': str(result['note']) if result.get('note') else None
            }
            
            # Keep the file for display (cleanup can be done later or periodically)
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 BITSTREAM AI IMAGE DETECTOR")
    print("="*80)
    print("\n📡 Starting server on http://localhost:5001")
    print("   Using Bitstream Forensics + Camera Signature:")
    print("   • Random Forest (97.2% accuracy, 348k training images)")
    print("   • DCT Coefficient Analysis (70 forensic features)")
    print("   • Quantization Pattern Detection")
    print("   • Camera Signature Analysis")
    print("\n   🎯 Performance: 97.4% real | 97.0% AI | 2.6% false positives")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
