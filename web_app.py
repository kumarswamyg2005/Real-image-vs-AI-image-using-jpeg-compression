"""
Web App using Bitstream Forensics AI Detector
==============================================
Bitstream forensics (DCT/quantization analysis)
+ AI Model Attribution + Camera Signature Analysis

Accuracy: 94.17% (trained on 939k images, three-way split, 4.1% false positive rate)
Supported formats: JPEG, PNG, WebP, HEIC/HEIF, TIFF, BMP

API Documentation: /api/docs (Swagger UI)
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from ensemble_detector import EnsembleAIDetector
from bitstream_features import SUPPORTED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max (HEIC/TIFF can be large)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ai_detector.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# JWT configuration (change in production!)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'change-this-secret-key-in-production')

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database and JWT
from models import init_db
from auth import init_jwt
init_db(app)
init_jwt(app)

# Register API blueprint (Swagger UI at /api/docs)
from api import api_bp
app.register_blueprint(api_bp)

# Initialize Ensemble AI Detector
print("Initializing AI Detector (Bitstream + SPN + Attribution)...")
detector = EnsembleAIDetector()
app.detector = detector  # Store for API access
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

    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            'error': f'Unsupported file format "{ext}". '
                     f'Supported: {", ".join(sorted(SUPPORTED_EXTENSIONS))}'
        }), 400

    if file:
        # Save file with unique name
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            # Get prediction with details from ensemble
            result = detector.predict(filepath, return_details=True)

            mb = result['model_breakdown']
            spn = mb.get('spn', {})
            attr = result.get('attribution', {})

            # Format response — convert all values to JSON-serializable types
            response = {
                'is_ai': bool(result['is_ai']),
                'label': str(result['label']),
                'confidence': float(result['confidence']),
                'ai_score': float(result['ai_probability'] * 100),
                'real_score': float(result['real_probability'] * 100),
                'model': str(result['model']),
                'method': str(result['method']),
                'image_url': f'/uploads/{unique_filename}',
                'image_format': str(result.get('image_format', 'unknown')),

                # Model breakdown for transparency
                'breakdown': {
                    'bitstream': {
                        'ai_prob': float(mb['bitstream']['ai_probability'] * 100),
                        'individual_probs': [float(p * 100) for p in mb['bitstream'].get('individual_probs', [])],
                        'ensemble_seeds': [int(s) for s in mb['bitstream'].get('ensemble_seeds', [])],
                        'weight': float(mb['bitstream']['weight'] * 100),
                        'contribution': float(mb['bitstream']['contribution'] * 100)
                    },
                    'spn': {
                        'ai_prob': float(spn.get('ai_probability', 0.5) * 100),
                        'prnu_score': float(spn.get('prnu_score', 0.0)),
                        'is_camera_noise': bool(spn.get('is_camera_noise', False)),
                        'weight': float(spn.get('weight', 0.0) * 100),
                        'contribution': float(spn.get('contribution', 0.0) * 100)
                    },
                    'camera': {
                        'is_camera': bool(mb['camera_signature']['is_camera_likely']),
                        'confidence': float(mb['camera_signature']['confidence']),
                        'weight': float(mb['camera_signature']['weight'] * 100),
                        'reasons': [str(r) for r in mb['camera_signature']['reasons']]
                    }
                },

                # AI Model Attribution
                'attribution': {
                    'top_model': str(attr.get('top_model', 'Unknown')),
                    'top_confidence': float(attr.get('top_confidence', 0.0)),
                    'is_ai_generated': bool(attr.get('is_ai_generated', False)),
                    'scores': {str(k): float(v) for k, v in attr.get('scores', {}).items()},
                    'reasoning': [str(r) for r in attr.get('reasoning', [])]
                },

                'note': str(result['note']) if result.get('note') else None
            }

            return jsonify(response)

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()

            # Clean up on error
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/heatmap', methods=['POST'])
def heatmap():
    data = request.get_json(silent=True)
    if not data or 'filepath' not in data:
        return jsonify({'error': 'Missing filepath'}), 400

    # Security: accept only the bare filename, resolve within uploads folder
    filename  = os.path.basename(data['filepath'].lstrip('/'))
    if not filename:
        return jsonify({'error': 'Invalid filepath'}), 400

    uploads_abs = os.path.abspath(app.config['UPLOAD_FOLDER'])
    safe_abs    = os.path.abspath(os.path.join(uploads_abs, filename))

    # Path-traversal guard
    if not safe_abs.startswith(uploads_abs + os.sep):
        return jsonify({'error': 'Invalid filepath'}), 400

    if not os.path.isfile(safe_abs):
        return jsonify({'error': 'File not found'}), 404

    try:
        from ai_region_heatmap import generate_heatmap
        result = generate_heatmap(safe_abs)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Heatmap generation failed: {str(e)}'}), 500


if __name__ == '__main__':
    import socket

    def _find_free_port(preferred: int = 5001, max_tries: int = 20) -> int:
        """Return *preferred* if free, else the next available port."""
        for port in range(preferred, preferred + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    return port
        # fall back to OS-assigned port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port = _find_free_port(5001)

    print("\n" + "="*80)
    print("🚀 BITSTREAM AI IMAGE DETECTOR")
    print("="*80)
    print(f"\n📡 Starting server on http://localhost:{port}")
    if port != 5001:
        print(f"   ⚠️  Port 5001 was busy — using port {port} instead")
    print("   Modules:")
    print("   • Bitstream Forensics v3 (94.17% accuracy, 939k images, 104 features, 4.1% FPR)")
    print("   • SPN Sensor Pattern Noise (camera fingerprinting)")
    print("   • AI Model Attribution (DALL-E / Midjourney / Stable Diffusion / Firefly)")
    print("   • Camera Signature Analysis")
    print("   Formats: JPEG · PNG · WebP · HEIC/HEIF · TIFF · BMP")
    print("\n   🎯 Performance: 95.9% real (TNR) | 92.5% AI (TPR) | 4.1% false positives")
    print(f"\n   📚 API Documentation: http://localhost:{port}/api/docs")
    print("   🔐 Default admin: admin / admin123 (change in production!)")
    print("\n" + "="*80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
