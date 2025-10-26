"""
Sea Surface Temperature Spherical Harmonic Expansion and Visualization Module
Designed for ERA5 Sea Surface Temperature Data Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
import torch
import torch.nn as nn
from scipy.special import lpmn, factorial
from src.plot.sst import COLOR_MAP_ERROR

class SphericalHarmonicAnalysis:
    """Sea Surface Temperature Spherical Harmonic Expansion Analysis Class"""
    
    def __init__(self, max_degree=10):
        """
        Initialize spherical harmonic analyzer
        
        Args:
            max_degree: Maximum degree for spherical harmonic expansion
        """
        self.max_degree = max_degree
        self.coefficients = {}
        self.reconstructed_data = None
        
    def compute_spherical_harmonics(self, l, m, theta, phi):
        """
        Compute spherical harmonic function Y_l^m(theta, phi)
        
        Args:
            l: Degree (order)
            m: Order (azimuthal quantum number)
            theta: Colatitude (0 to π)
            phi: Longitude (0 to 2π)
            
        Returns:
            Spherical harmonic function values
        """
        # Normalization factor
        norm_factor = np.sqrt((2*l + 1) * factorial(l - abs(m)) / 
                             (4 * np.pi * factorial(l + abs(m))))
        
        # Compute associated Legendre polynomials
        if abs(m) <= l:
            P_lm = lpmn(abs(m), l, np.cos(theta))[0][abs(m), l]
        else:
            P_lm = np.zeros_like(theta)
        
        # Spherical harmonic function
        if m == 0:
            Y_lm = norm_factor * P_lm
        elif m > 0:
            Y_lm = norm_factor * P_lm * np.cos(m * phi)
        else:  # m < 0
            Y_lm = norm_factor * P_lm * np.sin(abs(m) * phi)
            
        return Y_lm
    
    def _convert_to_numpy(self, data):
        """Convert PyTorch tensor to NumPy array"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def expand_sst_data(self, sst_data, lon, lat):
        """
        Expand sea surface temperature data into spherical harmonics
        
        Args:
            sst_data: Sea surface temperature data [lat, lon]
            lon: Longitude array
            lat: Latitude array
            
        Returns:
            Dictionary of expansion coefficients
        """
        # Convert to NumPy arrays
        sst_data = self._convert_to_numpy(sst_data)
        lon = self._convert_to_numpy(lon)
        lat = self._convert_to_numpy(lat)
        
        # Convert to spherical coordinates
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        
        # Create grids
        lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)
        
        # Convert to spherical coordinates (theta: colatitude, phi: longitude)
        theta = np.pi/2 - lat_grid  # Colatitude
        phi = lon_grid  # Longitude
        
        # Calculate area element (spherical)
        dA = np.sin(theta) * np.abs(lon_rad[1] - lon_rad[0]) * np.abs(lat_rad[1] - lat_rad[0])
        
        coefficients = {}
        
        print(f"Starting spherical harmonic expansion, maximum degree: {self.max_degree}")
        
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                # Compute spherical harmonic function
                Y_lm = self.compute_spherical_harmonics(l, m, theta, phi)
                
                # Calculate expansion coefficients
                # a_lm = ∫∫ f(θ,φ) Y_l^m*(θ,φ) sin(θ) dθ dφ
                integrand = sst_data * np.conj(Y_lm) * np.sin(theta)
                a_lm = np.sum(integrand * dA)
                
                coefficients[(l, m)] = a_lm
        
        self.coefficients = coefficients
        return coefficients
    
    def reconstruct_sst(self, lon, lat, max_degree=None):
        """
        Reconstruct sea surface temperature data from spherical harmonic coefficients
        
        Args:
            lon: Longitude array
            lat: Latitude array
            max_degree: Maximum degree for reconstruction
            
        Returns:
            Reconstructed sea surface temperature data
        """
        if max_degree is None:
            max_degree = self.max_degree
            
        # Convert to NumPy arrays
        lon = self._convert_to_numpy(lon)
        lat = self._convert_to_numpy(lat)
            
        # Convert to spherical coordinates
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        
        # Create grids
        lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)
        
        # Convert to spherical coordinates
        theta = np.pi/2 - lat_grid
        phi = lon_grid
        
        # Reconstruct data
        reconstructed = np.zeros_like(theta, dtype=complex)
        
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                if (l, m) in self.coefficients:
                    Y_lm = self.compute_spherical_harmonics(l, m, theta, phi)
                    reconstructed += self.coefficients[(l, m)] * Y_lm
        
        # Take real part
        self.reconstructed_data = np.real(reconstructed)
        return self.reconstructed_data
    
    def plot_spherical_harmonics(self, l, m, lon_range=(-180, 180), lat_range=(-80, 80), 
                                resolution=1.0):
        """
        Visualize individual spherical harmonic function
        
        Args:
            l: Degree
            m: Order
            lon_range: Longitude range
            lat_range: Latitude range
            resolution: Resolution
        """
        # Create grids
        lon = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
        lat = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
        
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Convert to spherical coordinates
        theta = np.pi/2 - np.deg2rad(lat_grid)
        phi = np.deg2rad(lon_grid)
        
        # Compute spherical harmonic function
        Y_lm = self.compute_spherical_harmonics(l, m, theta, phi)
        
        # Create figure
        fig = plt.figure(figsize=(12, 8), dpi=300)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Set map extent
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot spherical harmonic function
        im = ax.contourf(lon_grid, lat_grid, np.real(Y_lm), 
                        levels=20, cmap='RdBu_r', 
                        transform=ccrs.PlateCarree(), extend='both')
        
        # Add contour lines
        ax.contour(lon_grid, lat_grid, np.real(Y_lm), 
                  levels=10, colors='black', alpha=0.3, linewidths=0.5,
                  transform=ccrs.PlateCarree())
        
        # Set title and labels
        ax.set_title(f'Spherical Harmonic Y_{l}^{m} (Degree={l}, Order={m})', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, fraction=0.05, shrink=0.8)
        cbar.set_label('Spherical Harmonic Value', fontsize=12)
        
        # Set grid lines
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                    alpha=0.5, linestyle='--')
        
        plt.tight_layout()
        
        return fig, ax
    
    def plot_expansion_analysis(self, original_sst, reconstructed_sst, lon, lat):
        """
        Plot spherical harmonic expansion analysis
        
        Args:
            original_sst: Original sea surface temperature data
            reconstructed_sst: Reconstructed sea surface temperature data
            lon: Longitude array
            lat: Latitude array
        """
        # Convert to NumPy arrays
        original_sst = self._convert_to_numpy(original_sst)
        reconstructed_sst = self._convert_to_numpy(reconstructed_sst)
        lon = self._convert_to_numpy(lon)
        lat = self._convert_to_numpy(lat)
        
        # Calculate error
        error = original_sst - reconstructed_sst
        
        # Create figure
        fig = plt.figure(figsize=(20, 12), dpi=300)
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                             wspace=0.3, hspace=0.3)
        
        # Create grids
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 1. Original data
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax1.set_global()
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        im1 = ax1.contourf(lon_grid, lat_grid, original_sst, 
                          levels=20, cmap='jet', 
                          transform=ccrs.PlateCarree(), extend='both')
        ax1.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        
        # 2. Reconstructed data
        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ax2.set_global()
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax2.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        im2 = ax2.contourf(lon_grid, lat_grid, reconstructed_sst, 
                          levels=20, cmap='jet', 
                          transform=ccrs.PlateCarree(), extend='both')

        ax2.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        
        # 3. Error distribution
        ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        ax3.set_global()
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax3.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax3.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # 计算误差范围，确保色标以0为中心
        abs_max_error = max(abs(np.nanmin(error)), abs(np.nanmax(error)))
        error_levels = np.linspace(-abs_max_error, abs_max_error, 30)
        
        im3 = ax3.contourf(lon_grid, lat_grid, error, 
                          levels=error_levels, cmap=COLOR_MAP_ERROR, 
                          transform=ccrs.PlateCarree(), extend='both')

        ax3.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
        
        # 4. Coefficient energy spectrum
        ax4 = fig.add_subplot(gs[1, :])
        
        # Calculate energy for each degree
        energy_by_degree = {}
        for (l, m), coeff in self.coefficients.items():
            if l not in energy_by_degree:
                energy_by_degree[l] = 0
            energy_by_degree[l] += np.abs(coeff)**2
        
        degrees = sorted(energy_by_degree.keys())
        energies = [energy_by_degree[d] for d in degrees]
        
        ax4.semilogy(degrees, energies, 'bo-', linewidth=2, markersize=6)
        ax4.set_xlabel('Spherical Harmonic Degree', fontsize=12)
        ax4.set_ylabel('Energy (Log Scale)', fontsize=12)
        ax4.set_title('Spherical Harmonic Expansion Energy Spectrum', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        stats_text = f'RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C\nMax Error: {max_error:.3f}°C'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbars
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', 
                            pad=0.05, fraction=0.05, shrink=0.8)
        cbar1.set_label('Temperature (°C)', fontsize=10)
        
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', 
                            pad=0.05, fraction=0.05, shrink=0.8)
        cbar2.set_label('Temperature (°C)', fontsize=10)
        
        cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', 
                            pad=0.05, fraction=0.05, shrink=0.8)
        cbar3.set_label('Error (°C)', fontsize=10)
        
        plt.suptitle('Sea Surface Temperature Spherical Harmonic Expansion Analysis', fontsize=18, fontweight='bold', y=0.95)
        
        return fig
    
    def plot_individual_harmonics(self, max_degree=3):
        """
        Plot visualization of first few spherical harmonic functions
        
        Args:
            max_degree: Maximum degree to display
        """
        n_harmonics = sum(2*l + 1 for l in range(max_degree + 1))
        cols = 4
        rows = (n_harmonics + cols - 1) // cols
        
        fig = plt.figure(figsize=(20, 5*rows), dpi=300)
        
        plot_idx = 1
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                ax = fig.add_subplot(rows, cols, plot_idx, projection=ccrs.PlateCarree())
                ax.set_global()
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
                ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
                
                # Create grids
                lon = np.arange(-180, 181, 2)
                lat = np.arange(-80, 81, 2)
                lon_grid, lat_grid = np.meshgrid(lon, lat)
                
                # Convert to spherical coordinates
                theta = np.pi/2 - np.deg2rad(lat_grid)
                phi = np.deg2rad(lon_grid)
                
                # Compute spherical harmonic function
                Y_lm = self.compute_spherical_harmonics(l, m, theta, phi)
                
                # Plot
                im = ax.contourf(lon_grid, lat_grid, np.real(Y_lm), 
                               levels=15, cmap='RdBu_r', 
                               transform=ccrs.PlateCarree(), extend='both')
                
                ax.set_title(f'Y_{l}^{m}', fontsize=12, fontweight='bold')
                ax.gridlines(alpha=0.3, linestyle='--')
                
                plot_idx += 1
        
        plt.suptitle(f'Spherical Harmonic Visualization (Degrees 0-{max_degree})', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        return fig

# Main analysis function
def analyze_sst_spherical_harmonics(sst_data, lon, lat, max_degree=10):
    """
    Main function for sea surface temperature spherical harmonic expansion analysis
    
    Args:
        sst_data: Sea surface temperature data [lat, lon]
        lon: Longitude array
        lat: Latitude array
        max_degree: Maximum expansion degree
    """
    
    # Create analyzer
    analyzer = SphericalHarmonicAnalysis(max_degree=max_degree)
    
    print("Starting spherical harmonic expansion...")
    # Expand data
    coefficients = analyzer.expand_sst_data(sst_data, lon, lat)
    
    print("Reconstructing sea surface temperature data...")
    # Reconstruct data
    reconstructed = analyzer.reconstruct_sst(lon, lat)
    
    print("Generating visualization plots...")
    # Generate analysis plots
    analyzer.plot_expansion_analysis(sst_data, reconstructed, lon, lat)
    
    # Generate spherical harmonic visualization
    analyzer.plot_individual_harmonics(max_degree=3)
    
    return analyzer, coefficients, reconstructed