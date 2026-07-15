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
from sklearn.cluster import DBSCAN

# Fallback coordinates from the original project script
ORIGINAL_COORDINATES = np.array([
    (38.98765667, -76.987545), (39.03991652, -77.05364898), (38.98765667, -76.987545),
    (39.04822639, -76.98444849), (39.05903733, -77.0474325), (39.22623, -77.06966),
    (38.99930583, -77.0171455), (39.14042283, -77.21027052), (39.0931506, -77.19653287),
    (39.17389983, -77.2614475), (38.99552833, -77.03636), (39.2214725, -77.05976517),
    (38.97398, -77.00462833), (39.14632167, -77.17034667), (39.05553814, -76.95665433),
    (39.189535, -77.250415), (39.04283833, -77.05219667), (39.19241833, -77.26999333),
    (39.103395, -77.04762667), (39.00821467, -76.97983367), (39.06375333, -77.23017667),
    (39.04583717, -77.19172983), (39.08136873, -77.14656839), (39.14066403, -77.21422407),
    (39.05740667, -77.07353333), (39.169816, -77.1616985), (39.21642833, -77.23893667),
    (39.018128, -77.1206015), (39.07839488, -77.12989608), (39.09467017, -77.13461367),
    (39.01787867, -77.00721517), (39.15882667, -77.45195333), (39.0016375, -77.05377383),
    (39.12309553, -77.23018403), (39.15244667, -77.19625333), (39.08983, -77.250825),
    (39.24673667, -77.27470167), (39.16215667, -77.19221683), (39.05570667, -77.08211),
    (39.15753833, -77.20340833), (39.09071833, -77.04512383), (39.15997333, -77.20149333),
    (39.14009943, -77.19565158), (39.08049617, -77.15270517), (39.10628667, -77.152135),
    (39.05619539, -76.94432604), (39.08531683, -77.14843217), (39.02352392, -77.02909558),
    (39.1119955, -77.18702867), (39.1107449, -76.9335761)
])

class AmbulancePositionerDBSCAN:
    """
    Optimal Ambulance Positioning model utilizing Density-Based Spatial Clustering 
    of Applications with Noise (DBSCAN) to identify crash clusters and place 
    ambulances at cluster centroids.
    """
    def __init__(self, eps=0.01, min_samples=3, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.dbscan = None
        self.coordinates = None
        self.labels = None
        self.ambulance_positions = []

    def load_data(self, csv_path=None, sample_size=2000, random_seed=42):
        """
        Loads dataset coordinates. If no path is provided or load fails, 
        falls back to original hardcoded methodology coordinates.
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
                    print(f"Sampled {sample_size} coordinate points for DBSCAN.")
                
                self.coordinates = np.array(coords)
                return
            except Exception as e:
                print(f"Error loading CSV dataset: {e}. Falling back to default coordinates.")
        
        print("Using original hardcoded project coordinates (50 points).")
        self.coordinates = ORIGINAL_COORDINATES

    def fit(self, coordinates=None):
        """
        Fits the DBSCAN model on coordinates and computes cluster centers.
        """
        if coordinates is not None:
            self.coordinates = coordinates
        
        if self.coordinates is None:
            raise ValueError("No coordinates loaded. Run load_data() first.")

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.dbscan.fit(self.coordinates)
        self.labels = self.dbscan.labels_
        
        unique_labels = set(self.labels)
        self.ambulance_positions = []
        
        # Calculate cluster centroids as ambulance positions
        for label in unique_labels:
            if label != -1:
                cluster_points = self.coordinates[self.labels == label]
                cluster_center = np.mean(cluster_points, axis=0)
                self.ambulance_positions.append(cluster_center)
                
        self.ambulance_positions = np.array(self.ambulance_positions)
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
        
        unique_labels = set(self.labels)
        
        # Custom color palette for clean professional appearance
        colors = plt.cm.plasma(np.linspace(0, 0.8, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = (self.labels == label)
            if label == -1:
                # Noise points
                plt.scatter(self.coordinates[mask, 1], self.coordinates[mask, 0], 
                            c='#475569', s=35, label='Noise (Outliers)', alpha=0.5, edgecolors='none')
            else:
                plt.scatter(self.coordinates[mask, 1], self.coordinates[mask, 0], 
                            color=color, s=55, label=f'Cluster {label}', edgecolors='black', linewidths=0.5)
        
        # Plot ambulance positions (centers)
        if len(self.ambulance_positions) > 0:
            plt.scatter(self.ambulance_positions[:, 1], self.ambulance_positions[:, 0], 
                        c='#ef4444', marker='X', s=250, label='Ambulance Hubs', edgecolors='white', linewidths=1.5, zorder=5)
            
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title('Optimal Ambulance Positioning using DBSCAN Clustering', fontsize=14, fontweight='bold', pad=15)
        
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

        unique_labels = set(self.labels)
        colors = plt.cm.plasma(np.linspace(0, 0.8, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = (self.labels == label)
            if label == -1:
                ax.scatter(self.coordinates[mask, 1], self.coordinates[mask, 0],
                           c='#475569', s=35, label='Noise (Outliers)', alpha=0.5, edgecolors='none')
            else:
                ax.scatter(self.coordinates[mask, 1], self.coordinates[mask, 0],
                           color=color, s=55, label=f'Cluster {label}', edgecolors='black', linewidths=0.5)

        if len(self.ambulance_positions) > 0:
            ax.scatter(self.ambulance_positions[:, 1], self.ambulance_positions[:, 0],
                       c='#ef4444', marker='X', s=250, label='Ambulance Hubs',
                       edgecolors='white', linewidths=1.5, zorder=5)

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('DBSCAN Clustering — Simulation Result', fontsize=14, fontweight='bold', pad=15)

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
    parser = argparse.ArgumentParser(description='Optimal Ambulance Positioning using DBSCAN')
    parser.add_argument('--use-dataset', action='store_true', help='Use the full CSV dataset instead of hardcoded coordinates')
    parser.add_argument('--sample-size', type=int, default=2000, help='Sample size to extract from the large CSV dataset')
    parser.add_argument('--eps', type=float, default=0.01, help='DBSCAN epsilon parameter')
    parser.add_argument('--min-samples', type=int, default=3, help='DBSCAN min_samples parameter')
    parser.add_argument('--save-plot', type=str, default='web/assets/dbscan_plot.png', help='Path to save cluster plot')
    
    args = parser.parse_args()
    
    csv_dataset_path = 'data/Crash_Reporting_-_Drivers_Data.csv'
    
    positioner = AmbulancePositionerDBSCAN(eps=args.eps, min_samples=args.min_samples)
    
    if args.use_dataset:
        positioner.load_data(csv_path=csv_dataset_path, sample_size=args.sample_size)
    else:
        positioner.load_data()
        
    ambulances = positioner.fit()
    positioner.plot_clusters(save_path=args.save_plot)
    
    print("\nOptimal Ambulance Hub Positions (Latitude, Longitude):")
    for i, pos in enumerate(ambulances, 1):
        print(f"Hub {i}: ({pos[0]:.6f}, {pos[1]:.6f})")
