"""
Spatial Interpolation Module
Implements IDW and Kriging for hyper-local AQI estimation
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class SpatialInterpolator:
    def __init__(self, method='idw', power=2):
        """
        Initialize spatial interpolator
        
        Args:
            method: 'idw' or 'kriging'
            power: Power parameter for IDW (default=2)
        """
        self.method = method
        self.power = power
        self.variogram_params = None
        
    def idw_interpolation(self, known_points, known_values, query_points, power=None):
        """
        Inverse Distance Weighting interpolation
        
        Args:
            known_points: Array of shape (n, 2) with [lat, lon]
            known_values: Array of shape (n,) with AQI values
            query_points: Array of shape (m, 2) with query [lat, lon]
            power: Distance decay power (default=self.power)
        
        Returns:
            interpolated_values: Array of shape (m,)
        """
        if power is None:
            power = self.power
            
        # Calculate distances between query points and known points
        distances = cdist(query_points, known_points, metric='euclidean')
        
        # Avoid division by zero for points at exact sensor locations
        # Add small epsilon to distances
        epsilon = 1e-10
        distances = distances + epsilon
        
        # Calculate weights (inverse distance to the power)
        weights = 1 / (distances ** power)
        
        # Normalize weights
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_normalized = weights / weights_sum
        
        # Interpolate values
        interpolated_values = (weights_normalized * known_values).sum(axis=1)
        
        return interpolated_values
    
    def spherical_variogram(self, h, nugget, sill, range_param):
        """
        Spherical variogram model
        
        Args:
            h: Distance (lag)
            nugget: Nugget effect (measurement error)
            sill: Sill (maximum variance)
            range_param: Range (distance where correlation becomes zero)
        """
        gamma = np.zeros_like(h)
        
        # For h < range
        mask = h < range_param
        gamma[mask] = nugget + (sill - nugget) * (
            1.5 * (h[mask] / range_param) - 0.5 * (h[mask] / range_param) ** 3
        )
        
        # For h >= range
        gamma[~mask] = sill
        
        return gamma
    
    def fit_variogram(self, known_points, known_values):
        """
        Fit variogram model to data (simplified version)
        """
        # Calculate pairwise distances
        distances = cdist(known_points, known_points, metric='euclidean')
        
        # Calculate pairwise semi-variances
        n = len(known_values)
        semivariances = np.zeros_like(distances)
        
        for i in range(n):
            for j in range(i+1, n):
                semivariances[i, j] = 0.5 * (known_values[i] - known_values[j]) ** 2
                semivariances[j, i] = semivariances[i, j]
        
        # Flatten and remove zeros (self-distances)
        mask = distances > 0
        h = distances[mask].flatten()
        gamma = semivariances[mask].flatten()
        
        # Initial parameter guesses
        nugget_init = np.percentile(gamma, 10)
        sill_init = np.percentile(gamma, 90)
        range_init = np.percentile(h, 50)
        
        # Simplified parameter estimation (using percentiles)
        self.variogram_params = {
            'nugget': nugget_init,
            'sill': sill_init,
            'range': range_init
        }
        
        return self.variogram_params
    
    def kriging_interpolation(self, known_points, known_values, query_points):
        """
        Simple Kriging interpolation
        
        Args:
            known_points: Array of shape (n, 2) with [lat, lon]
            known_values: Array of shape (n,) with AQI values
            query_points: Array of shape (m, 2) with query [lat, lon]
        
        Returns:
            interpolated_values: Array of shape (m,)
            variances: Kriging variance (uncertainty) for each point
        """
        # Fit variogram if not already done
        if self.variogram_params is None:
            self.fit_variogram(known_points, known_values)
        
        n = len(known_points)
        m = len(query_points)
        
        # Build kriging matrix
        distances_known = cdist(known_points, known_points, metric='euclidean')
        K = self.spherical_variogram(
            distances_known,
            self.variogram_params['nugget'],
            self.variogram_params['sill'],
            self.variogram_params['range']
        )
        
        # Add Lagrange multiplier row/column
        K_aug = np.ones((n+1, n+1))
        K_aug[:n, :n] = K
        K_aug[n, n] = 0
        
        interpolated_values = np.zeros(m)
        variances = np.zeros(m)
        
        # Interpolate for each query point
        for i, query_point in enumerate(query_points):
            # Calculate distances from query point to known points
            distances_query = cdist(
                query_point.reshape(1, -1),
                known_points,
                metric='euclidean'
            ).flatten()
            
            # Calculate covariances
            k = self.spherical_variogram(
                distances_query,
                self.variogram_params['nugget'],
                self.variogram_params['sill'],
                self.variogram_params['range']
            )
            
            # Augment with Lagrange constraint
            k_aug = np.append(k, 1)
            
            # Solve for weights
            try:
                weights = np.linalg.solve(K_aug, k_aug)
                
                # Interpolate
                interpolated_values[i] = np.dot(weights[:n], known_values)
                
                # Calculate variance (kriging error)
                variances[i] = np.dot(weights, k_aug)
                
            except np.linalg.LinAlgError:
                # Fallback to IDW if kriging fails
                interpolated_values[i] = self.idw_interpolation(
                    known_points,
                    known_values,
                    query_point.reshape(1, -1)
                )[0]
                variances[i] = 0
        
        return interpolated_values, variances
    
    def interpolate(self, known_points, known_values, query_points):
        """
        Main interpolation method
        
        Args:
            known_points: DataFrame or array with latitude, longitude
            known_values: Series or array with AQI values
            query_points: DataFrame or array with latitude, longitude
        
        Returns:
            interpolated_values: Interpolated AQI at query points
            (variances): Only for kriging method
        """
        # Convert to numpy arrays if needed
        if isinstance(known_points, pd.DataFrame):
            known_points = known_points[['latitude', 'longitude']].values
        if isinstance(known_values, pd.Series):
            known_values = known_values.values
        if isinstance(query_points, pd.DataFrame):
            query_points = query_points[['latitude', 'longitude']].values
            
        # Ensure 2D arrays
        if len(known_points.shape) == 1:
            known_points = known_points.reshape(-1, 2)
        if len(query_points.shape) == 1:
            query_points = query_points.reshape(-1, 2)
        
        if self.method == 'idw':
            return self.idw_interpolation(known_points, known_values, query_points)
        
        elif self.method == 'kriging':
            return self.kriging_interpolation(known_points, known_values, query_points)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")

def create_interpolation_grid(lat_range, lon_range, grid_size=50):
    """
    Create a grid of points for interpolation visualization
    
    Args:
        lat_range: Tuple (min_lat, max_lat)
        lon_range: Tuple (min_lon, max_lon)
        grid_size: Number of points in each dimension
    
    Returns:
        grid_points: Array of shape (grid_size^2, 2)
        lat_grid: 2D array for plotting
        lon_grid: 2D array for plotting
    """
    lat = np.linspace(lat_range[0], lat_range[1], grid_size)
    lon = np.linspace(lon_range[0], lon_range[1], grid_size)
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    
    return grid_points, lat_grid, lon_grid

def interpolate_city_aqi(sensor_data, method='idw', grid_size=50):
    """
    Interpolate AQI across entire city area
    
    Args:
        sensor_data: DataFrame with columns: latitude, longitude, aqi
        method: 'idw' or 'kriging'
        grid_size: Resolution of interpolation grid
    
    Returns:
        DataFrame with interpolated AQI values and coordinates
    """
    # Get sensor locations and values
    known_points = sensor_data[['latitude', 'longitude']].values
    known_values = sensor_data['aqi'].values
    
    # Create interpolation grid
    lat_range = (sensor_data['latitude'].min() - 0.01, 
                 sensor_data['latitude'].max() + 0.01)
    lon_range = (sensor_data['longitude'].min() - 0.01,
                 sensor_data['longitude'].max() + 0.01)
    
    grid_points, lat_grid, lon_grid = create_interpolation_grid(
        lat_range, lon_range, grid_size
    )
    
    # Interpolate
    interpolator = SpatialInterpolator(method=method)
    
    if method == 'kriging':
        interpolated_values, variances = interpolator.interpolate(
            known_points, known_values, grid_points
        )
        
        result_df = pd.DataFrame({
            'latitude': grid_points[:, 0],
            'longitude': grid_points[:, 1],
            'aqi_interpolated': interpolated_values,
            'kriging_variance': variances
        })
    else:
        interpolated_values = interpolator.interpolate(
            known_points, known_values, grid_points
        )
        
        result_df = pd.DataFrame({
            'latitude': grid_points[:, 0],
            'longitude': grid_points[:, 1],
            'aqi_interpolated': interpolated_values
        })
    
    return result_df, lat_grid, lon_grid

if __name__ == '__main__':
    # Test spatial interpolation
    np.random.seed(42)
    
    # Create sample sensor data
    n_sensors = 10
    known_points = np.random.rand(n_sensors, 2) * 0.1 + [19.0, 72.8]  # Mumbai area
    known_values = np.random.rand(n_sensors) * 100 + 50  # AQI 50-150
    
    # Create query points
    query_points = np.random.rand(5, 2) * 0.1 + [19.0, 72.8]
    
    # Test IDW
    print("Testing IDW interpolation...")
    idw = SpatialInterpolator(method='idw', power=2)
    idw_results = idw.interpolate(known_points, known_values, query_points)
    print(f"IDW Results: {idw_results}")
    
    # Test Kriging
    print("\nTesting Kriging interpolation...")
    kriging = SpatialInterpolator(method='kriging')
    kriging_results, variances = kriging.interpolate(known_points, known_values, query_points)
    print(f"Kriging Results: {kriging_results}")
    print(f"Variances: {variances}")
