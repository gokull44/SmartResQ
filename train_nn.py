import os
import csv
import random
import io
import base64
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

ORIGINAL_COORDINATES = np.array([
    (38.98765667, -76.987545), (39.03991652, -77.05364898), (38.743573, -77.54699707),
    (39.12537803, -77.19194047), (39.02157017, -77.07633333), (39.11641667, -76.9505),
    (39.00014446, -77.10988077), (39.140092, -77.0563), (39.0724598, -77.06486034),
    (39.05440667, -77.05048833), (39.14877889, -77.21343947), (39.1690707, -77.24978833),
    (39.10772138, -77.09790811), (39.1038175, -76.93908666), (39.0041115, -77.12907269),
    (39.195653, -77.18864333), (38.99238667, -77.060428), (39.13550616, -77.2699489),
    (39.1486666, -77.22337562), (39.09249724, -76.99683328), (39.00923745, -77.07965497),
    (39.164895, -77.23921667), (39.14824033, -77.26314367), (39.02262796, -77.18325859),
    (39.117974, -77.25304667), (38.97525167, -77.07707167), (39.1934545, -77.2010785),
    (38.955657, -77.09558367), (39.034985, -77.0727583), (39.18473333, -77.232185),
    (39.15234789, -77.14193769), (39.03563111, -77.205199), (39.01894302, -77.26788361),
    (39.1859435, -77.0955867), (39.19016405, -77.20476017),
    (39.14973657, -77.17579538), (39.06043817, -77.1117805), (39.041318, -76.98480083),
    (39.02029603, -77.01266248), (39.04407059, -77.03934728), (38.9875717, -77.01054533),
    (38.99671, -77.0253), (38.944445, -77.07183483), (38.9727427, -77.0804527),
    (39.0025515, -77.2233679), (39.18022333, -77.24894333), (39.05238983, -77.0746425),
    (39.0180667, -77.24775667), (38.981764, -77.1145838), (39.2255417, -77.2755565),
    (39.044, -76.9866282), (38.98786567, -77.0926115), (39.03480468, -77.15764207),
    (39.05274555, -77.18404601), (39.0639517, -77.1549067), (39.031351, -77.00481903),
    (39.0167347, -77.04239868), (39.1901417, -77.26476583), (39.22455033, -77.426384),
    (39.00945337, -77.01722396), (39.0161333, -77.01143), (39.0472759, -77.0671052),
    (39.146258, -77.27468667), (39.058077, -77.1261367), (39.171659, -77.19043067),
    (39.211579, -77.2584767), (39.03572954, -77.03204943), (39.1811458, -77.2632975),
    (39.18329707, -77.13664793), (39.050667, -77.20411047), (39.018077, -77.04449),
    (39.0282529, -77.2728563), (39.16872333, -77.18823), (39.2276333, -77.16864167),
    (39.12635, -77.0664167), (39.05815478, -77.1956217), (39.04449378, -77.04928769)
])

#99.18
class AmbulancePositionerNN:
    """
    Neural Network (MLP) Classifier trained on GMM cluster labels
    to predict optimal ambulance hub for any coordinate.
    Model is saved as ambulance_model.h5 (joblib format).
    """
    def __init__(self, n_components=4, model_dir='.'):
        self.n_components = n_components
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.gmm = None
        self.coordinates = None
        self.labels = None
        self.ambulance_positions = []

        self.h5_path = os.path.join(model_dir, 'ambulance_model.h5')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.gmm_path = os.path.join(model_dir, 'gmm_labels.pkl')

    def load_data(self, csv_path=None, sample_size=5000, random_seed=42):
        """Load coordinates from CSV dataset or fall back to hardcoded list."""
        if csv_path and os.path.exists(csv_path):
            print(f"Loading data from {csv_path}...")
            coords = []
            try:
                with open(csv_path, 'r', errors='ignore') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    lat_idx = header.index('Latitude')
                    lon_idx = header.index('Longitude')
                    for row in reader:
                        if not row or len(row) <= max(lat_idx, lon_idx):
                            continue
                        try:
                            lat = float(row[lat_idx])
                            lon = float(row[lon_idx])
                            if lat != 0.0 and lon != 0.0:
                                coords.append((lat, lon))
                        except ValueError:
                            continue

                print(f"Total valid coordinates found: {len(coords)}")
                if len(coords) > sample_size:
                    random.seed(random_seed)
                    coords = random.sample(coords, sample_size)
                    print(f"Sampled {sample_size} coordinate points for training.")

                self.coordinates = np.array(coords)
                return
            except Exception as e:
                print(f"Error loading CSV: {e}. Falling back to default coordinates.")

        print("Using default hardcoded project coordinates.")
        self.coordinates = ORIGINAL_COORDINATES

    def fit_and_train(self, epochs=100, batch_size=32):
        """
        Step 1: Fit GMM to define target hub labels.
        Step 2: Train MLP Neural Network on (lat, lon) -> hub cluster label.
        Step 3: Save model as ambulance_model.h5 using joblib.
        Returns: (train_accuracy, val_accuracy)
        """
        if self.coordinates is None:
            raise ValueError("No data loaded. Run load_data() first.")

        print("\n[1/4] Fitting GMM to establish ambulance hub positions...")
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=42, n_init=3)
        self.gmm.fit(self.coordinates)
        self.labels = self.gmm.predict(self.coordinates)
        self.ambulance_positions = self.gmm.means_
        print(f"      {self.n_components} ambulance hubs identified.")

        print("\n[2/4] Standardizing coordinate features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.coordinates)
        y = self.labels

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )

        print("\n[3/4] Training MLP Neural Network classifier...")
        # hidden_layer_sizes = (64, 32) — a 2-layer deep neural network
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=min(batch_size, len(X_train)),
            max_iter=epochs,
            random_state=42,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
        self.model.fit(X_train, y_train)

        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        print(f"\n      Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")

        print(f"\n[4/4] Saving model artifacts...")
        # Save the MLP model as .h5 file using joblib
        joblib.dump(self.model, self.h5_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump({'gmm': self.gmm, 'hubs': self.ambulance_positions}, self.gmm_path)
        print(f"      Model saved -> {self.h5_path}")
        print(f"      Scaler saved -> {self.scaler_path}")
        print(f"      GMM saved   -> {self.gmm_path}")

        return train_acc, val_acc

    def load_saved_model(self):
        """Load the saved MLP .h5 model, scaler and GMM from disk."""
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"Model not found: {self.h5_path}. Train the model first.")

        print(f"Loading MLP model from {self.h5_path}...")
        self.model = joblib.load(self.h5_path)

        print(f"Loading scaler from {self.scaler_path}...")
        self.scaler = joblib.load(self.scaler_path)

        gmm_data = joblib.load(self.gmm_path)
        self.gmm = gmm_data['gmm']
        self.ambulance_positions = gmm_data['hubs']
        print(f"Model loaded. {len(self.ambulance_positions)} ambulance hubs active.")

    def predict_hub(self, lat, lon):
        """
        Predict the best ambulance hub for a given (lat, lon).
        Returns: (hub_index, hub_coordinate, confidence_score)
        """
        if self.model is None or self.scaler is None:
            self.load_saved_model()

        X_input = np.array([[lat, lon]])
        X_scaled = self.scaler.transform(X_input)
        probs = self.model.predict_proba(X_scaled)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        hub_coord = tuple(self.ambulance_positions[pred_class])
        return pred_class, hub_coord, confidence

    def plot_decision_boundaries_base64(self):
        """Generate decision boundary plot and return as base64 PNG string."""
        if self.model is None or self.scaler is None:
            self.load_saved_model()

        lats = self.coordinates[:, 0]
        lons = self.coordinates[:, 1]
        lat_min, lat_max = lats.min() - 0.05, lats.max() + 0.05
        lon_min, lon_max = lons.min() - 0.05, lons.max() + 0.05

        xx, yy = np.meshgrid(
            np.linspace(lon_min, lon_max, 150),
            np.linspace(lat_min, lat_max, 150)
        )
        grid_points = np.c_[yy.ravel(), xx.ravel()]
        grid_scaled = self.scaler.transform(grid_points)
        preds = self.model.predict(grid_scaled)
        Z = preds.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0b0f19')
        ax.set_facecolor('#0f172a')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.tick_params(colors='#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f8fafc')

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        scatter = ax.scatter(lons, lats, c=self.labels, cmap='viridis',
                             s=50, alpha=0.85, edgecolors='black', linewidths=0.4)
        if len(self.ambulance_positions) > 0:
            ax.scatter(self.ambulance_positions[:, 1], self.ambulance_positions[:, 0],
                       c='#ef4444', marker='X', s=280, label='Ambulance Hubs',
                       edgecolors='white', linewidths=1.5, zorder=6)

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Neural Network (MLP) — Ambulance Coverage Decision Boundaries',
                     fontsize=13, fontweight='bold', pad=15)
        legend = ax.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=10)
        for t in legend.get_texts():
            t.set_color('#f8fafc')
        ax.grid(True, linestyle='--', alpha=0.1)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#0b0f19')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP Neural Network for Ambulance Positioning')
    parser.add_argument('--use-dataset', action='store_true', help='Use full CSV dataset')
    parser.add_argument('--sample-size', type=int, default=5000, help='Sample size from CSV')
    parser.add_argument('--n-components', type=int, default=4, help='Number of ambulance hubs')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    args = parser.parse_args()

    csv_path = 'data/Crash_Reporting_-_Drivers_Data.csv'
    positioner = AmbulancePositionerNN(n_components=args.n_components)

    if args.use_dataset:
        positioner.load_data(csv_path=csv_path, sample_size=args.sample_size)
    else:
        positioner.load_data()

    train_acc, val_acc = positioner.fit_and_train(epochs=args.epochs)

    print(f"\n=== Training Complete ===")
    print(f"Train Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"Val   Accuracy : {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"Model saved    : ambulance_model.h5")
    print("\nOptimal Ambulance Hub Positions (Lat, Lon):")
    for i, hub in enumerate(positioner.ambulance_positions, 1):
        print(f"  Hub {i}: ({hub[0]:.6f}, {hub[1]:.6f})")
