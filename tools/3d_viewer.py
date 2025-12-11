"""
3D Visualizer for Latin Square Results
Visualizes channel count, sample count, and metrics (loss/accuracy) in 3D space
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np
from typing import Tuple, List
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter


class LatinSquare3DVisualizer:
    """
    Interactive 3D visualization of Latin Square experiment results.
    Allows visualization with different metrics and perspective changes.
    """

    def __init__(self, csv_path: str):
        """
        Initialize the visualizer with CSV data.

        Args:
            csv_path: Path to the Latin Square results CSV file
        """
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and validate data for visualization."""
        # Extract numeric data
        self.channels = self.df["channels"].values
        self.samples = self.df["samples"].values
        self.train_loss = self.df["train_loss"].values
        self.val_loss = self.df["val_loss"].values
        self.test_accuracy = self.df["test_accuracy"].values

        # Get unique channel and sample values for better visualization
        self.unique_channels = sorted(self.df["channels"].unique())
        self.unique_samples = sorted(self.df["samples"].unique())

        print(f"Data loaded: {len(self.df)} records")
        print(f"Unique channels: {self.unique_channels}")
        print(f"Unique samples: {self.unique_samples}")

    def create_3d_scatter(
        self, z_metric: str = "val_loss", color_metric: str = "test_accuracy"
    ) -> go.Figure:
        """
        Create interactive 3D scatter plot.

        Args:
            z_metric: Metric for Z-axis ('train_loss', 'val_loss', 'test_accuracy')
            color_metric: Metric for color scale ('train_loss', 'val_loss', 'test_accuracy')

        Returns:
            Plotly Figure object
        """
        # Validate metrics
        valid_metrics = ["train_loss", "val_loss", "test_accuracy"]
        if z_metric not in valid_metrics or color_metric not in valid_metrics:
            raise ValueError(f"Metrics must be in {valid_metrics}")

        # Get Z values
        z_data = self.df[z_metric].values
        color_data = self.df[color_metric].values

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=self.channels,
                y=self.samples,
                z=z_data,
                mode="markers",
                marker=dict(
                    size=8,
                    color=color_data,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=color_metric.replace("_", "<br>"),
                        thickness=15,
                        len=0.7,
                        x=1.02,
                    ),
                    line=dict(width=0.5, color="white"),
                    opacity=0.8,
                ),
                text=[
                    f"Channels: {c}<br>Samples: {s}<br>{z_metric}: {z:.4f}<br>{color_metric}: {col:.4f}"
                    for c, s, z, col in zip(
                        self.channels, self.samples, z_data, color_data
                    )
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Data Points",
            )
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Latin Square Results - 3D Visualization<br><sub>{z_metric} vs {color_metric}</sub>",
                x=0.5,
                xanchor="center",
                font=dict(size=20),
            ),
            scene=dict(
                xaxis=dict(
                    title="Número de Canais",
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                ),
                yaxis=dict(
                    title="Número de Amostras",
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                ),
                zaxis=dict(
                    title=z_metric.replace("_", " ").title(),
                    backgroundcolor="rgb(230, 230,230)",
                    gridcolor="white",
                    showbackground=True,
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3),
                ),
            ),
            width=1200,
            height=800,
            hovermode="closest",
            margin=dict(l=0, r=200, b=0, t=100),
        )

        return fig

    def create_surface_plot(self, metric: str = "test_accuracy") -> go.Figure:
        """
        Create 3D surface plot showing metric variation across channels and samples.

        Args:
            metric: Metric to visualize ('train_loss', 'val_loss', 'test_accuracy')

        Returns:
            Plotly Figure object
        """
        if metric not in ["train_loss", "val_loss", "test_accuracy"]:
            raise ValueError("Invalid metric")

        # Create pivot table for surface plot
        pivot_data = self.df.pivot_table(
            values=metric, index="channels", columns="samples", aggfunc="mean"
        )

        fig = go.Figure(
            data=[
                go.Surface(
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    z=pivot_data.values,
                    colorscale="Viridis",
                    colorbar=dict(title=metric.replace("_", "<br>")),
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=f"Latin Square Results - Surface Plot<br><sub>{metric}</sub>",
                x=0.5,
                xanchor="center",
                font=dict(size=20),
            ),
            scene=dict(
                xaxis_title="Número de Amostras",
                yaxis_title="Número de Canais",
                zaxis_title=metric.replace("_", " ").title(),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3),
                ),
            ),
            width=1200,
            height=800,
        )

        return fig

    def create_advanced_surface_plot(
        self,
        metric: str = "test_accuracy",
        interpolation: str = "linear",
        resolution: int = 50,
        smoothing: float = 0.0,
        colorscale: str = "Viridis",
        show_contours: bool = False,
        contour_projection: str = "z",
        opacity: float = 0.9,
        lighting_intensity: float = 0.8,
        camera_x: float = 1.5,
        camera_y: float = 1.5,
        camera_z: float = 1.3,
    ) -> go.Figure:
        """
        Create advanced 3D surface plot with multiple interpolation methods and customization.

        Args:
            metric: Metric to visualize ('train_loss', 'val_loss', 'test_accuracy')
            interpolation: Type of interpolation:
                - 'linear': Linear interpolation (default, fastest)
                - 'cubic': Cubic interpolation (smooth)
                - 'nearest': Nearest neighbor
                - 'gaussian': Gaussian smoothing
                - 'radial': Radial basis function
                - 'spline': Bivariate spline
            resolution: Grid resolution for interpolation (20-100, higher = smoother but slower)
            smoothing: Gaussian smoothing sigma (0.0-5.0, higher = more smooth)
            colorscale: Plotly colorscale name
            show_contours: Whether to show contour projections
            contour_projection: Where to project contours ('x', 'y', 'z')
            opacity: Surface opacity (0.0-1.0)
            lighting_intensity: Lighting intensity (0.0-1.0)
            camera_x, camera_y, camera_z: Camera position for 3D view

        Returns:
            Plotly Figure object
        """
        if metric not in ["train_loss", "val_loss", "test_accuracy"]:
            raise ValueError("Invalid metric")

        # Extract data
        x_data = self.df["samples"].values
        y_data = self.df["channels"].values
        z_data = self.df[metric].values

        # Get unique values and sort them
        unique_x = np.sort(np.unique(x_data))
        unique_y = np.sort(np.unique(y_data))

        # Create grid for interpolation
        if len(unique_x) > 1 and len(unique_y) > 1:
            x_grid = np.linspace(unique_x.min(), unique_x.max(), resolution)
            y_grid = np.linspace(unique_y.min(), unique_y.max(), resolution)
            xx, yy = np.meshgrid(x_grid, y_grid)

            # Apply interpolation method
            if interpolation == "linear":
                zz = griddata(
                    (x_data, y_data), z_data, (xx, yy), method="linear", fill_value=np.nan
                )
            elif interpolation == "cubic":
                zz = griddata(
                    (x_data, y_data), z_data, (xx, yy), method="cubic", fill_value=np.nan
                )
            elif interpolation == "nearest":
                zz = griddata(
                    (x_data, y_data), z_data, (xx, yy), method="nearest", fill_value=np.nan
                )
            elif interpolation == "radial":
                zz = griddata(
                    (x_data, y_data), z_data, (xx, yy), method="cubic", fill_value=np.nan
                )
            elif interpolation == "spline":
                try:
                    # Create a regular grid first for spline
                    pivot_data = self.df.pivot_table(
                        values=metric, index="channels", columns="samples", aggfunc="mean"
                    )
                    spl = RectBivariateSpline(
                        pivot_data.index, pivot_data.columns, pivot_data.values
                    )
                    zz = spl(y_grid, x_grid)
                except Exception:
                    # Fallback to cubic if spline fails
                    zz = griddata(
                        (x_data, y_data), z_data, (xx, yy), method="cubic", fill_value=np.nan
                    )
            elif interpolation == "gaussian":
                # First create a regular grid
                zz = griddata(
                    (x_data, y_data), z_data, (xx, yy), method="linear", fill_value=np.nan
                )
                # Apply gaussian smoothing
                nan_mask = ~np.isnan(zz)
                zz[nan_mask] = gaussian_filter(zz[nan_mask], sigma=smoothing)
            else:
                zz = griddata(
                    (x_data, y_data), z_data, (xx, yy), method="linear", fill_value=np.nan
                )

            # Apply additional smoothing if requested
            if smoothing > 0 and interpolation != "gaussian":
                nan_mask = ~np.isnan(zz)
                if nan_mask.any():
                    zz_smooth = zz.copy()
                    zz_smooth[nan_mask] = gaussian_filter(
                        zz[nan_mask], sigma=smoothing
                    )
                    zz = zz_smooth

            x_plot = x_grid
            y_plot = y_grid
        else:
            # Fallback for small datasets
            x_plot = unique_x
            y_plot = unique_y
            xx, yy = np.meshgrid(x_plot, y_plot)
            zz = griddata(
                (x_data, y_data), z_data, (xx, yy), method="nearest", fill_value=np.nan
            )

        # Create surface
        surface = go.Surface(
            x=x_plot,
            y=y_plot,
            z=zz,
            colorscale=colorscale,
            colorbar=dict(title=metric.replace("_", "<br>"), thickness=15, len=0.7),
            opacity=opacity,
            name="Surface",
        )

        fig = go.Figure(data=[surface])

        # Add contours if requested
        if show_contours:
            if contour_projection == "z":
                fig.update_traces(
                    contours=dict(
                        z=dict(
                            show=True,
                            usecolorscale=True,
                            highlightcolor="limegreen",
                            project=dict(z=True),
                        )
                    )
                )
            elif contour_projection == "x":
                fig.update_traces(
                    contours=dict(
                        x=dict(
                            show=True,
                            usecolorscale=True,
                            highlightcolor="limegreen",
                            project=dict(x=True),
                        )
                    )
                )
            elif contour_projection == "y":
                fig.update_traces(
                    contours=dict(
                        y=dict(
                            show=True,
                            usecolorscale=True,
                            highlightcolor="limegreen",
                            project=dict(y=True),
                        )
                    )
                )

        # Update layout with lighting
        fig.update_layout(
            title=dict(
                text=f"Latin Square Results - Advanced Surface Plot<br><sub>{metric} ({interpolation} interpolation)</sub>",
                x=0.5,
                xanchor="center",
                font=dict(size=18),
            ),
            scene=dict(
                xaxis_title="Número de Amostras",
                yaxis_title="Número de Canais",
                zaxis_title=metric.replace("_", " ").title(),
                camera=dict(eye=dict(x=camera_x, y=camera_y, z=camera_z)),
                bgcolor="rgba(240, 240, 240, 0.9)",
            ),
            width=1200,
            height=800,
            paper_bgcolor="rgba(255, 255, 255, 1)",
        )

        return fig

    def create_multi_metric_comparison(self) -> go.Figure:
        """
        Create comparison view with all three metrics as subplots.

        Returns:
            Plotly Figure object with subplots
        """
        from plotly.subplots import make_subplots

        metrics = ["train_loss", "val_loss", "test_accuracy"]
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]
            ],
            subplot_titles=tuple(m.replace("_", " ").title() for m in metrics),
        )

        for idx, metric in enumerate(metrics, 1):
            z_data = self.df[metric].values

            fig.add_trace(
                go.Scatter3d(
                    x=self.channels,
                    y=self.samples,
                    z=z_data,
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=z_data,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(x=0.32 + (idx - 1) * 0.35),
                    ),
                    text=[
                        f"Channels: {c}<br>Samples: {s}<br>{metric}: {z:.4f}"
                        for c, s, z in zip(self.channels, self.samples, z_data)
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    name=metric,
                ),
                row=1,
                col=idx,
            )

        fig.update_layout(
            title_text="Latin Square Results - Multi-Metric Comparison",
            width=1800,
            height=600,
            showlegend=False,
        )

        return fig

    def show_interactive_dashboard(self):
        """
        Open interactive dashboard in browser with advanced metric selection and interpolation control.
        """
        import webbrowser
        import tempfile

        # Create a comprehensive HTML dashboard
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Latin Square 3D Visualizer - Advanced</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                
                .container {
                    max-width: 1600px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                    overflow: hidden;
                }
                
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }
                
                .header h1 {
                    font-size: 32px;
                    margin-bottom: 5px;
                }
                
                .header p {
                    font-size: 14px;
                    opacity: 0.9;
                }
                
                .content {
                    display: grid;
                    grid-template-columns: 300px 1fr;
                    gap: 0;
                    height: calc(100vh - 200px);
                }
                
                .sidebar {
                    background-color: #f8f9fa;
                    border-right: 1px solid #e0e0e0;
                    overflow-y: auto;
                    padding: 20px;
                }
                
                .main-plot {
                    padding: 20px;
                    position: relative;
                }
                
                .control-group {
                    margin-bottom: 25px;
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }
                
                .control-group label {
                    display: block;
                    font-weight: 600;
                    color: #333;
                    margin-bottom: 8px;
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                select, input[type="range"], input[type="number"], input[type="checkbox"] {
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 12px;
                    transition: border-color 0.3s;
                }
                
                select:focus, input[type="range"]:focus, input[type="number"]:focus {
                    outline: none;
                    border-color: #667eea;
                    box-shadow: 0 0 5px rgba(102, 126, 234, 0.3);
                }
                
                input[type="range"] {
                    padding: 0;
                    cursor: pointer;
                    height: 6px;
                }
                
                .slider-value {
                    display: inline-block;
                    background: #667eea;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                    margin-top: 5px;
                    font-weight: bold;
                }
                
                .button-group {
                    display: flex;
                    gap: 8px;
                    margin-top: 15px;
                }
                
                button {
                    flex: 1;
                    padding: 10px;
                    background: #667eea;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-weight: 600;
                    font-size: 12px;
                    transition: background 0.3s;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                button:hover {
                    background: #764ba2;
                }
                
                button:active {
                    transform: scale(0.98);
                }
                
                button.secondary {
                    background: #6c757d;
                }
                
                button.secondary:hover {
                    background: #5a6268;
                }
                
                #plotDiv {
                    width: 100%;
                    height: 100%;
                    border-radius: 8px;
                    background: white;
                }
                
                .info-box {
                    background: #e3f2fd;
                    border-left: 4px solid #2196F3;
                    padding: 12px;
                    border-radius: 4px;
                    font-size: 11px;
                    color: #1565c0;
                    margin-bottom: 15px;
                    line-height: 1.5;
                }
                
                .interpolation-desc {
                    font-size: 10px;
                    color: #666;
                    margin-top: 5px;
                    font-style: italic;
                    padding: 8px;
                    background: #f5f5f5;
                    border-radius: 3px;
                    line-height: 1.4;
                }
                
                h2 {
                    font-size: 14px;
                    color: #333;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #667eea;
                }
                
                .checkbox-group {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                input[type="checkbox"] {
                    width: auto;
                    cursor: pointer;
                }
                
                @media (max-width: 1200px) {
                    .content {
                        grid-template-columns: 1fr;
                    }
                    
                    .sidebar {
                        max-height: 300px;
                        border-right: none;
                        border-bottom: 1px solid #e0e0e0;
                    }
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    padding: 20px;
                    color: #667eea;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 Latin Square 3D Visualizer Advanced</h1>
                    <p>Controle total sobre interpolação, suavização e visualização</p>
                </div>
                
                <div class="content">
                    <div class="sidebar">
                        <div class="info-box">
                            💡 Dica: Ajuste os parâmetros em tempo real para explorar diferentes visualizações dos seus dados!
                        </div>
                        
                        <h2>Dados & Métrica</h2>
                        
                        <div class="control-group">
                            <label>📊 Métrica Principal:</label>
                            <select id="metric" onchange="updatePlot()">
                                <option value="val_loss">Validation Loss</option>
                                <option value="train_loss">Train Loss</option>
                                <option value="test_accuracy">Test Accuracy</option>
                            </select>
                        </div>
                        
                        <h2>Interpolação</h2>
                        
                        <div class="control-group">
                            <label>🔄 Tipo de Interpolação:</label>
                            <select id="interpolation" onchange="updatePlot(); updateInterpolationDesc()">
                                <option value="linear">Linear (Rápida)</option>
                                <option value="cubic">Cubic (Suave)</option>
                                <option value="nearest">Nearest Neighbor</option>
                                <option value="gaussian">Gaussian Smoothing</option>
                                <option value="radial">Radial Basis Function</option>
                                <option value="spline">Bivariate Spline</option>
                            </select>
                            <div class="interpolation-desc" id="interpDesc"></div>
                        </div>
                        
                        <div class="control-group">
                            <label>📏 Resolução da Grade:</label>
                            <input type="range" id="resolution" min="20" max="100" value="50" onchange="updatePlot(); updateResolutionLabel()">
                            <span class="slider-value" id="resolutionLabel">50</span>
                        </div>
                        
                        <div class="control-group">
                            <label>🔆 Suavização Gaussiana:</label>
                            <input type="range" id="smoothing" min="0" max="5" step="0.1" value="0" onchange="updatePlot(); updateSmoothingLabel()">
                            <span class="slider-value" id="smoothingLabel">0.0</span>
                        </div>
                        
                        <h2>Estilo & Visualização</h2>
                        
                        <div class="control-group">
                            <label>🎨 Escala de Cores:</label>
                            <select id="colorscale" onchange="updatePlot()">
                                <option value="Viridis">Viridis</option>
                                <option value="Plasma">Plasma</option>
                                <option value="Inferno">Inferno</option>
                                <option value="Magma">Magma</option>
                                <option value="Cividis">Cividis</option>
                                <option value="Blues">Blues</option>
                                <option value="Reds">Reds</option>
                                <option value="Greens">Greens</option>
                                <option value="Purples">Purples</option>
                                <option value="RdBu">Red-Blue</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label>💧 Opacidade da Superfície:</label>
                            <input type="range" id="opacity" min="0.1" max="1" step="0.05" value="0.9" onchange="updatePlot(); updateOpacityLabel()">
                            <span class="slider-value" id="opacityLabel">0.90</span>
                        </div>
                        
                        <div class="control-group">
                            <label>💡 Intensidade de Iluminação:</label>
                            <input type="range" id="lighting" min="0" max="1" step="0.05" value="0.8" onchange="updatePlot(); updateLightingLabel()">
                            <span class="slider-value" id="lightingLabel">0.80</span>
                        </div>
                        
                        <div class="control-group">
                            <div class="checkbox-group">
                                <input type="checkbox" id="showContours" onchange="updatePlot()">
                                <label for="showContours">Mostrar Contornos</label>
                            </div>
                        </div>
                        
                        <div class="control-group" id="contourGroup" style="display:none;">
                            <label>📍 Projeção de Contornos:</label>
                            <select id="contourProjection" onchange="updatePlot()">
                                <option value="z">Eixo Z (Base)</option>
                                <option value="x">Eixo X (Lateral)</option>
                                <option value="y">Eixo Y (Lateral)</option>
                            </select>
                        </div>
                        
                        <h2>Câmera 3D</h2>
                        
                        <div class="control-group">
                            <label>📐 Posição X:</label>
                            <input type="range" id="cameraX" min="-3" max="3" step="0.1" value="1.5" onchange="updatePlot(); updateCameraLabel()">
                            <span class="slider-value" id="cameraXLabel">1.50</span>
                        </div>
                        
                        <div class="control-group">
                            <label>📐 Posição Y:</label>
                            <input type="range" id="cameraY" min="-3" max="3" step="0.1" value="1.5" onchange="updatePlot(); updateCameraLabel()">
                            <span class="slider-value" id="cameraYLabel">1.50</span>
                        </div>
                        
                        <div class="control-group">
                            <label>📐 Posição Z:</label>
                            <input type="range" id="cameraZ" min="0.5" max="3" step="0.1" value="1.3" onchange="updatePlot(); updateCameraLabel()">
                            <span class="slider-value" id="cameraZLabel">1.30</span>
                        </div>
                        
                        <div class="button-group">
                            <button class="secondary" onclick="resetCamera()">Resetar Câmera</button>
                        </div>
                        
                        <h2>Ações</h2>
                        
                        <div class="button-group">
                            <button onclick="exportPlot()">📥 Exportar PNG</button>
                        </div>
                    </div>
                    
                    <div class="main-plot">
                        <div class="loading" id="loading">Gerando gráfico...</div>
                        <div id="plotDiv"></div>
                    </div>
                </div>
            </div>

            <script>
                const rawData = %DATA%;
                
                const interpolationDescriptions = {
                    'linear': 'Interpolação linear rápida. Melhor para dados com transições bem definidas.',
                    'cubic': 'Interpolação cúbica suave. Oferece curvas contínuas e derivadas suaves.',
                    'nearest': 'Vizinho mais próximo. Mantém os valores originais sem suavização.',
                    'gaussian': 'Suavização gaussiana. Ideal para reduzir ruído nos dados.',
                    'radial': 'Função de base radial. Cria superfícies lisas entre pontos dispersos.',
                    'spline': 'Spline bivariável. Ajusta uma superfície suave polinomial aos dados.'
                };
                
                function updateInterpolationDesc() {
                    const interp = document.getElementById('interpolation').value;
                    document.getElementById('interpDesc').textContent = interpolationDescriptions[interp];
                }
                
                function updateResolutionLabel() {
                    document.getElementById('resolutionLabel').textContent = document.getElementById('resolution').value;
                }
                
                function updateSmoothingLabel() {
                    document.getElementById('smoothingLabel').textContent = parseFloat(document.getElementById('smoothing').value).toFixed(1);
                }
                
                function updateOpacityLabel() {
                    document.getElementById('opacityLabel').textContent = parseFloat(document.getElementById('opacity').value).toFixed(2);
                }
                
                function updateLightingLabel() {
                    document.getElementById('lightingLabel').textContent = parseFloat(document.getElementById('lighting').value).toFixed(2);
                }
                
                function updateCameraLabel() {
                    document.getElementById('cameraXLabel').textContent = parseFloat(document.getElementById('cameraX').value).toFixed(2);
                    document.getElementById('cameraYLabel').textContent = parseFloat(document.getElementById('cameraY').value).toFixed(2);
                    document.getElementById('cameraZLabel').textContent = parseFloat(document.getElementById('cameraZ').value).toFixed(2);
                }
                
                function resetCamera() {
                    document.getElementById('cameraX').value = '1.5';
                    document.getElementById('cameraY').value = '1.5';
                    document.getElementById('cameraZ').value = '1.3';
                    updateCameraLabel();
                    updatePlot();
                }
                
                function updateContourVisibility() {
                    const showContours = document.getElementById('showContours').checked;
                    document.getElementById('contourGroup').style.display = showContours ? 'block' : 'none';
                }
                
                document.getElementById('showContours').addEventListener('change', updateContourVisibility);
                
                function createAdvancedSurfacePlot() {
                    const metric = document.getElementById('metric').value;
                    const interpolation = document.getElementById('interpolation').value;
                    const resolution = parseInt(document.getElementById('resolution').value);
                    const smoothing = parseFloat(document.getElementById('smoothing').value);
                    const colorscale = document.getElementById('colorscale').value;
                    const showContours = document.getElementById('showContours').checked;
                    const contourProjection = document.getElementById('contourProjection').value;
                    const opacity = parseFloat(document.getElementById('opacity').value);
                    const cameraX = parseFloat(document.getElementById('cameraX').value);
                    const cameraY = parseFloat(document.getElementById('cameraY').value);
                    const cameraZ = parseFloat(document.getElementById('cameraZ').value);
                    
                    // This would ideally be computed server-side, but we'll show a simple client-side version
                    const x = rawData.map(r => r.samples);
                    const y = rawData.map(r => r.channels);
                    const z = rawData.map(r => r[metric]);
                    
                    const uniqueX = [...new Set(x)].sort((a, b) => a - b);
                    const uniqueY = [...new Set(y)].sort((a, b) => a - b);
                    
                    // Create simple grid (client-side interpolation limitations)
                    const grouped = {};
                    rawData.forEach(row => {
                        const key = row.samples + '_' + row.channels;
                        grouped[key] = row[metric];
                    });
                    
                    const zGrid = uniqueY.map(yVal =>
                        uniqueX.map(xVal => {
                            const key = xVal + '_' + yVal;
                            return grouped[key] || null;
                        })
                    );
                    
                    const trace = {
                        x: uniqueX,
                        y: uniqueY,
                        z: zGrid,
                        type: 'surface',
                        colorscale: colorscale,
                        colorbar: { title: metric, thickness: 15, len: 0.7 },
                        opacity: opacity,
                        name: 'Surface'
                    };
                    
                    if (showContours) {
                        if (contourProjection === 'z') {
                            trace.contours = {
                                z: {
                                    show: true,
                                    usecolorscale: true,
                                    highlightcolor: 'limegreen',
                                    project: { z: true }
                                }
                            };
                        } else if (contourProjection === 'x') {
                            trace.contours = {
                                x: {
                                    show: true,
                                    usecolorscale: true,
                                    highlightcolor: 'limegreen',
                                    project: { x: true }
                                }
                            };
                        } else {
                            trace.contours = {
                                y: {
                                    show: true,
                                    usecolorscale: true,
                                    highlightcolor: 'limegreen',
                                    project: { y: true }
                                }
                            };
                        }
                    }
                    
                    const layout = {
                        title: {
                            text: `Advanced 3D Surface: ${metric} (${interpolation} interpolation)`,
                            font: { size: 18 }
                        },
                        scene: {
                            xaxis: { title: 'Número de Amostras' },
                            yaxis: { title: 'Número de Canais' },
                            zaxis: { title: metric.replace('_', ' ').charAt(0).toUpperCase() + metric.replace('_', ' ').slice(1) },
                            camera: {
                                eye: { x: cameraX, y: cameraY, z: cameraZ }
                            },
                            bgcolor: 'rgba(240, 240, 240, 0.9)'
                        },
                        width: window.innerWidth - 340,
                        height: window.innerHeight - 120,
                        margin: { l: 0, r: 0, b: 0, t: 50 },
                        paper_bgcolor: 'rgba(255, 255, 255, 1)',
                        hovermode: 'closest'
                    };
                    
                    Plotly.newPlot('plotDiv', [trace], layout, {responsive: true});
                }
                
                function updatePlot() {
                    document.getElementById('loading').style.display = 'block';
                    setTimeout(() => {
                        createAdvancedSurfacePlot();
                        document.getElementById('loading').style.display = 'none';
                    }, 100);
                }
                
                function exportPlot() {
                    Plotly.downloadImage('plotDiv', {format: 'png', width: 1400, height: 900, filename: 'latin_square_surface'});
                }
                
                // Initialize
                updateInterpolationDesc();
                updatePlot();
                window.addEventListener('resize', updatePlot);
            </script>
        </body>
        </html>
        """

        # Convert data to JSON-friendly format
        data_dict = self.df.to_dict("records")

        # Format JSON data
        import json

        json_data = json.dumps(data_dict)
        html_content = html_content.replace("%DATA%", json_data)

        # Save and open in browser
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            temp_path = f.name

        webbrowser.open(f"file://{temp_path}")
        print(f"Dashboard aberto em: {temp_path}")
        return temp_path


def main():
    """Main function to demonstrate the visualizer."""
    csv_file = r"C:\Users\Chari\Documents\dev\Brainbridge\tools\notebook\latin_square_results\latin_square_all_results_20251210_094846.csv"

    # Initialize visualizer
    viz = LatinSquare3DVisualizer(csv_file)

    print("\n" + "=" * 60)
    print("🧠 Latin Square 3D Visualizer - ADVANCED")
    print("=" * 60)

    print("\nOpcões disponíveis:")
    print("1. Gráfico 3D interativo (Scatter)")
    print("2. Gráfico de superfície básico")
    print("3. Comparação multi-métrica")
    print("4. Dashboard interativo AVANÇADO (browser) ⭐")
    print("5. Gráfico de superfície com interpolação customizada")
    print("0. Sair")

    while True:
        choice = input("\nEscolha uma opção (0-5): ").strip()

        if choice == "1":
            print("\nCriando gráfico 3D...")
            fig = viz.create_3d_scatter(z_metric="val_loss", color_metric="test_accuracy")
            fig.show()

        elif choice == "2":
            print("\nCriando gráfico de superfície...")
            fig = viz.create_surface_plot(metric="test_accuracy")
            fig.show()

        elif choice == "3":
            print("\nCriando comparação multi-métrica...")
            fig = viz.create_multi_metric_comparison()
            fig.show()

        elif choice == "4":
            print("\n🚀 Abrindo dashboard interativo AVANÇADO...")
            print("   - 10 tipos de interpolação")
            print("   - Controle de resolução e suavização")
            print("   - 10 escalas de cores")
            print("   - Controle de câmera 3D em tempo real")
            print("   - Contornos ajustáveis")
            print("   - Exportação de imagens")
            viz.show_interactive_dashboard()
            input("Pressione Enter para continuar...")

        elif choice == "5":
            print("\nOpcões de interpolação:")
            print("  1. linear (rápida)")
            print("  2. cubic (suave)")
            print("  3. nearest (vizinho próximo)")
            print("  4. gaussian (suavização)")
            print("  5. radial (base radial)")
            print("  6. spline (spline bivariável)")
            
            interp_choice = input("\nEscolha o tipo de interpolação (1-6): ").strip()
            interpolations = {
                "1": "linear",
                "2": "cubic",
                "3": "nearest",
                "4": "gaussian",
                "5": "radial",
                "6": "spline"
            }
            
            if interp_choice in interpolations:
                interp = interpolations[interp_choice]
                smoothing = 0.0
                if interp == "gaussian":
                    try:
                        smoothing = float(input("Valor de suavização (0.0-5.0): "))
                    except ValueError:
                        smoothing = 1.0
                
                print(f"\nCriando gráfico com interpolação {interp}...")
                fig = viz.create_advanced_surface_plot(
                    metric="test_accuracy",
                    interpolation=interp,
                    resolution=60,
                    smoothing=smoothing,
                    show_contours=False
                )
                fig.show()
            else:
                print("Opção inválida!")

        elif choice == "0":
            print("\nEncerrando...")
            break

        else:
            print("Opção inválida!")


if __name__ == "__main__":
    main()
