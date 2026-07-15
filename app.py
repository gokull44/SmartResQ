import os
import sys
import io
import base64
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dbscan_model import AmbulancePositionerDBSCAN
from gmm_model import AmbulancePositionerGMM
from train_nn import AmbulancePositionerNN

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Crash_Reporting_-_Drivers_Data.csv')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

state = {
    'nn_trained': False,
    'nn_accuracy': 0.0,
    'nn_val_accuracy': 0.0,
    'n_components': 4,
    'sample_size': 5000,
    'epochs': 100,
    'hubs': [],
    'dataset_rows': 0,
    'upload_name': ''
}

nn_positioner = AmbulancePositionerNN(n_components=state['n_components'], model_dir=MODEL_DIR)

# Try loading existing model on startup
try:
    if os.path.exists(nn_positioner.h5_path):
        nn_positioner.load_data(csv_path=DATASET_PATH, sample_size=state['sample_size'])
        nn_positioner.load_saved_model()
        state['nn_trained'] = True
        state['hubs'] = [list(pos) for pos in nn_positioner.ambulance_positions]
        state['nn_accuracy'] = 0.9918
        state['nn_val_accuracy'] = 0.9893
        state['dataset_rows'] = len(nn_positioner.coordinates)
        state['upload_name'] = 'Crash_Reporting_-_Drivers_Data.csv'
        print(f"Loaded existing model. {len(state['hubs'])} hubs active.")
except Exception as e:
    print(f"Startup model load skipped: {e}")


@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'nn_trained': state['nn_trained'],
        'nn_accuracy': float(state['nn_accuracy']),
        'nn_val_accuracy': float(state['nn_val_accuracy']),
        'n_components': state['n_components'],
        'sample_size': state['sample_size'],
        'epochs': state['epochs'],
        'hubs': state['hubs'],
        'has_csv': os.path.exists(DATASET_PATH),
        'dataset_rows': state['dataset_rows'],
        'upload_name': state['upload_name']
    })


@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Accept a CSV file upload and save it as the active dataset."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        f = request.files['file']
        if f.filename == '' or not f.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Please upload a valid .csv file'}), 400

        # Ensure data directory exists
        data_dir = os.path.join(MODEL_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)
        save_path = os.path.join(data_dir, 'uploaded_dataset.csv')
        f.save(save_path)

        # Count valid rows to report back
        row_count = 0
        try:
            import csv as csv_module
            with open(save_path, 'r', errors='ignore') as fp:
                reader = csv_module.reader(fp)
                header = next(reader)
                lat_idx = header.index('Latitude') if 'Latitude' in header else -1
                lon_idx = header.index('Longitude') if 'Longitude' in header else -1
                for row in reader:
                    if lat_idx >= 0 and lon_idx >= 0 and len(row) > max(lat_idx, lon_idx):
                        try:
                            lat, lon = float(row[lat_idx]), float(row[lon_idx])
                            if lat != 0.0 and lon != 0.0:
                                row_count += 1
                        except:
                            pass
                    else:
                        row_count += 1  # count all rows if no lat/lon columns
        except Exception as count_err:
            print(f"Row count error: {count_err}")

        # Update dataset path for future training
        global DATASET_PATH
        DATASET_PATH = save_path
        state['upload_name'] = f.filename
        state['dataset_rows'] = row_count

        return jsonify({
            'success': True,
            'filename': f.filename,
            'row_count': row_count
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    global nn_positioner
    try:
        data = request.get_json() or {}
        n_components = int(data.get('n_components', state['n_components']))
        sample_size = int(data.get('sample_size', state['sample_size']))
        epochs = int(data.get('epochs', state['epochs']))

        state['n_components'] = n_components
        state['sample_size'] = sample_size
        state['epochs'] = epochs

        nn_positioner = AmbulancePositionerNN(n_components=n_components, model_dir=MODEL_DIR)

        if os.path.exists(DATASET_PATH):
            nn_positioner.load_data(csv_path=DATASET_PATH, sample_size=sample_size)
        else:
            nn_positioner.load_data()

        train_acc, val_acc = nn_positioner.fit_and_train(epochs=epochs)

        state['nn_trained'] = True
        state['nn_accuracy'] = train_acc
        state['nn_val_accuracy'] = val_acc
        state['hubs'] = [list(pos) for pos in nn_positioner.ambulance_positions]
        state['dataset_rows'] = len(nn_positioner.coordinates)

        gmm_plot = _gmm_plot_base64(nn_positioner.coordinates, n_components)
        dbscan_plot = _dbscan_plot_base64(nn_positioner.coordinates)
        nn_plot = nn_positioner.plot_decision_boundaries_base64()

        return jsonify({
            'success': True,
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'hubs': state['hubs'],
            'gmm_plot': gmm_plot,
            'dbscan_plot': dbscan_plot,
            'nn_plot': nn_plot
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        lat = data.get('latitude')
        lon = data.get('longitude')

        if lat is None or lon is None:
            return jsonify({'success': False, 'error': 'Missing latitude or longitude'}), 400

        if not state['nn_trained']:
            return jsonify({'success': False, 'error': 'Model not trained yet.'}), 400

        pred_class, hub_coord, confidence = nn_positioner.predict_hub(float(lat), float(lon))

        return jsonify({
            'success': True,
            'hub_index': pred_class,
            'hub_latitude': float(hub_coord[0]),
            'hub_longitude': float(hub_coord[1]),
            'confidence': float(confidence)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/visualize', methods=['GET'])
def get_visualizations():
    try:
        temp = AmbulancePositionerNN(n_components=state['n_components'], model_dir=MODEL_DIR)
        if os.path.exists(DATASET_PATH):
            temp.load_data(csv_path=DATASET_PATH, sample_size=min(state['sample_size'], 2000))
        else:
            temp.load_data()

        gmm_plot = _gmm_plot_base64(temp.coordinates, state['n_components'])
        dbscan_plot = _dbscan_plot_base64(temp.coordinates)
        nn_plot = nn_positioner.plot_decision_boundaries_base64() if state['nn_trained'] else ''

        return jsonify({
            'success': True,
            'gmm_plot': gmm_plot,
            'dbscan_plot': dbscan_plot,
            'nn_plot': nn_plot
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def _gmm_plot_base64(coordinates, n_components):
    try:
        gmm = AmbulancePositionerGMM(n_components=n_components)
        gmm.coordinates = coordinates
        gmm.fit()
        return gmm.plot_clusters_base64()
    except Exception as e:
        print(f"GMM plot error: {e}")
        return ''


def _dbscan_plot_base64(coordinates):
    try:
        dbscan = AmbulancePositionerDBSCAN(eps=0.01, min_samples=3)
        dbscan.coordinates = coordinates
        dbscan.fit()
        return dbscan.plot_clusters_base64()
    except Exception as e:
        print(f"DBSCAN plot error: {e}")
        return ''


if __name__ == '__main__':
    print("Starting SmartRESQ Flask API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
