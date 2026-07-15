import os
import io
import csv
import base64
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Fallback coordinates from the original GMM script
ORIGINAL_COORDINATES = np.array([
    # Dataset 1
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
    # Dataset 2
    (39.14973657, -77.17579538), (39.06043817, -77.1117805), (39.041318, -76.98480083),
    (39.02029603, -77.01266248), (39.04407059, -77.03934728), (38.9875717, -77.01054533),
    (38.99671, -77.0253), (38.944445, -77.07183483), (38.9727427, -77.0804527),
    (39.0025515, -77.2233679), (39.18022333, -77.24894333), (39.05238983, -77.0746425),
    (39.0180667, -77.24775667), (38.981764, -77.1145838), (39.2255417, -77.2755565),
    (39.0446, -76.9866282), (38.98786567, -77.0926115), (39.03480468, -77.15764207),
    (39.05274555, -77.18404601), (39.0639517, -77.1549067), (39.031351, -77.00481903),
    (39.0167347, -77.04239868), (39.1901417, -77.26476583), (39.22455033, -77.426384),
    (39.00945337, -77.01722396), (39.0161333, -77.01143), (39.0472759, -77.0671052),
    (39.146258, -77.27468667), (39.058077, -77.1261367), (39.171659, -77.19043067),
    (39.211579, -77.2584767), (39.03572954, -77.03204943), (39.1811458, -77.2632975),
    (39.18329707, -77.13664793), (39.050667, -77.20411047), (39.018077, -77.04449),
    (39.0282529, -77.2728563), (39.16872333, -77.18823), (39.2276333, -77.16864167),
    (39.12635, -77.0664167), (39.05815478, -77.1956217), (39.04449378, -77.04928769)
])

class AmbulancePositionerGMM:
    """
    Optimal Ambulance Positioning model utilizing Gaussian Mixture Models (GMM)
    to model the density of road accident locations and place ambulances 
    at the cluster means (centroids).
    """
    def __init__(self, n_components=4, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.coordinates = None
        self.labels = None
        self.ambulance_positions = []

    def load_data(self, csv_path=None, sample_size=2000, random_seed=42):
        """
        Loads dataset coordinates. If no path is provided or load fails, 
        falls back to original GMM coordinates.
        """
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
                        lat_val, lon_val = row[lat_idx], row[lon_idx]
                        if lat_val and lon_val:
                            try:
                                lat, lon = float(lat_val), float(lon_val)
                                if lat != 0.0 and lon != 0.0:
                                    coords.append((lat, lon))
                            except ValueError:
                                continue
                
                print(f"Total valid coordinates found: {len(coords)}")
                if len(coords) > sample_size:
                    random.seed(random_seed)
                    coords = random.sample(coords, sample_size)
                    print(f"Sampled {sample_size} coordinate points for GMM.")
                
                self.coordinates = np.array(coords)
                return
            except Exception as e:
                print(f"Error loading CSV dataset: {e}. Falling back to default coordinates.")
        
        print("Using original hardcoded GMM project coordinates (85 points).")
        self.coordinates = ORIGINAL_COORDINATES

    def fit(self, coordinates=None):
        """
        Fits the GMM model on coordinates and extracts cluster centers (means).
        """
        if coordinates is not None:
            self.coordinates = coordinates
            
        if self.coordinates is None:
            raise ValueError("No coordinates loaded. Run load_data() first.")

        self.gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.gmm.fit(self.coordinates)
        self.labels = self.gmm.predict(self.coordinates)
        self.ambulance_positions = self.gmm.means_
        return self.ambulance_positions

    def plot_clusters(self, save_path=None):
        """
        Generates and optionally saves a beautiful high-resolution cluster plot.
        """
        if self.coordinates is None or self.labels is None:
            raise ValueError("Model must be fitted before plotting.")
            
        plt.figure(figsize=(12, 7), facecolor='#0b0f19')
        ax = plt.axes()
        ax.set_facecolor('#0f172a')
        
        # Plot styling
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        ax.tick_params(colors='#94a3b8', which='both')
        ax.yaxis.label.set_color('#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f8fafc')
        
        # Plot accident locations grouped by GMM component
        scatter = plt.scatter(self.coordinates[:, 1], self.coordinates[:, 0], 
                              c=self.labels, cmap='viridis', s=55, alpha=0.8, edgecolors='black', linewidths=0.5, label='Accidents')
        
        # Add colorbar for GMM components
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Gaussian Component / Cluster Label', color='#94a3b8', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='#94a3b8', labelcolor='#94a3b8')
        
        # Plot ambulance positions (GMM centers)
        if len(self.ambulance_positions) > 0:
            plt.scatter(self.ambulance_positions[:, 1], self.ambulance_positions[:, 0], 
                        c='#ef4444', marker='X', s=250, label='Ambulance Hubs', edgecolors='white', linewidths=1.5, zorder=5)
            
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title('Optimal Ambulance Positioning using Gaussian Mixture Models (GMM)', fontsize=14, fontweight='bold', pad=15)
        
        # Legend styling
        legend = plt.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=10, loc='best')
        for text in legend.get_texts():
            text.set_color('#f8fafc')
            
        plt.grid(True, linestyle='--', alpha=0.1)
        plt.tight_layout()
        
        if save_path:
            # Ensure output folder exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, facecolor='#0b0f19')
            print(f"Plot saved successfully to {save_path}")
        
        plt.close()

    def plot_clusters_base64(self):
        """
        Generates the cluster plot and returns it as a base64-encoded PNG string.
        Used by the API server to send plots to the web frontend.
        """
        if self.coordinates is None or self.labels is None:
            raise ValueError("Model must be fitted before plotting.")

        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0b0f19')
        ax.set_facecolor('#0f172a')

        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        ax.tick_params(colors='#94a3b8', which='both')
        ax.yaxis.label.set_color('#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f8fafc')

        scatter = ax.scatter(self.coordinates[:, 1], self.coordinates[:, 0],
                             c=self.labels, cmap='viridis', s=55, alpha=0.8,
                             edgecolors='black', linewidths=0.5, label='Accidents')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster Label', color='#94a3b8', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='#94a3b8', labelcolor='#94a3b8')

        if len(self.ambulance_positions) > 0:
            ax.scatter(self.ambulance_positions[:, 1], self.ambulance_positions[:, 0],
                       c='#ef4444', marker='X', s=250, label='Ambulance Hubs',
                       edgecolors='white', linewidths=1.5, zorder=5)

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('GMM Clustering — Simulation Result', fontsize=14, fontweight='bold', pad=15)

        legend = ax.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=10, loc='best')
        for text in legend.get_texts():
            text.set_color('#f8fafc')

        ax.grid(True, linestyle='--', alpha=0.1)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor='#0b0f19')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimal Ambulance Positioning using GMM')
    parser.add_argument('--use-dataset', action='store_true', help='Use the full CSV dataset instead of hardcoded coordinates')
    parser.add_argument('--sample-size', type=int, default=2000, help='Sample size to extract from the large CSV dataset')
    parser.add_argument('--n-components', type=int, default=4, help='Number of GMM clusters (Ambulance locations)')
    parser.add_argument('--save-plot', type=str, default='web/assets/gmm_plot.png', help='Path to save cluster plot')
    
    args = parser.parse_args()
    
    csv_dataset_path = 'data/Crash_Reporting_-_Drivers_Data.csv'
    
    positioner = AmbulancePositionerGMM(n_components=args.n_components)
    
    if args.use_dataset:
        positioner.load_data(csv_path=csv_dataset_path, sample_size=args.sample_size)
    else:
        positioner.load_data()
        
    ambulances = positioner.fit()
    positioner.plot_clusters(save_path=args.save_plot)
    
    print("\nOptimal Ambulance Hub Positions (Latitude, Longitude):")
    for i, pos in enumerate(ambulances, 1):
        print(f"Hub {i}: ({pos[0]:.6f}, {pos[1]:.6f})")
