"""
3D Visualizer for Latin Square Results - VERSÃO SIMPLIFICADA
Visualiza channel count, sample count e metrics em 3D com interpolação de superfície
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from scipy.interpolate import Rbf

class LatinSquare3DVisualizer:
    """Visualizador simples 3D para resultados Latin Square."""

    def __init__(self, csv_path: str):
        """
        Inicializa o visualizador com dados do CSV.
        
        Args:
            csv_path: Caminho para o arquivo CSV
        """
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        print(f"✅ Dados carregados: {len(self.df)} registros")
        print(f"Colunas: {self.df.columns.tolist()}\n")

    def create_3d_scatter(self, metric: str = "val_loss") -> go.Figure:
        """
        Cria gráfico 3D scatter simples.
        
        Args:
            metric: Métrica para Z (val_loss, train_loss, test_accuracy)
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=self.df['samples'],
            y=self.df['channels'],
            z=self.df[metric],
            mode='markers',
            marker=dict(
                size=8,
                color=self.df['test_accuracy'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Test Accuracy'),
                line=dict(width=0.5, color='white')
            ),
            text=[f"Ch: {c}, Samples: {s}, {metric}: {m:.4f}, Acc: {a:.4f}" 
                  for c, s, m, a in zip(self.df['channels'], self.df['samples'], 
                                        self.df[metric], self.df['test_accuracy'])],
            hovertemplate='%{text}<extra></extra>'
        )])

        fig.update_layout(
            title=f'3D Scatter - {metric}',
            scene=dict(
                xaxis_title='Samples',
                yaxis_title='Channels',
                zaxis_title=metric
            ),
            width=1000,
            height=800
        )
        return fig

    def create_surface(self, metric: str = "val_loss", resolution: int = 30) -> go.Figure:
        """
        Cria superfície 3D com interpolação spline.
        
        Args:
            metric: Métrica para Z (val_loss, train_loss, test_accuracy)
            resolution: Resolução da grid (mais alto = mais suave)
        
        Returns:
            Plotly Figure
        """
        # Extrair dados
        x = self.df['samples'].values.astype(float)
        y = self.df['channels'].values.astype(float)
        z = self.df[metric].values.astype(float)
        
        # Criar grid para interpolação
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolar usando RBF (Radial Basis Function) - mais suave que cubic
        rbf = Rbf(x, y, z, function='thin_plate', smooth=0.1)
        zi = rbf(xi_grid, yi_grid)
        
        # Criar superfície
        fig = go.Figure(data=[go.Surface(
            x=xi,
            y=yi,
            z=zi,
            colorscale='Viridis',
            colorbar=dict(title=metric)
        )])
        
        fig.update_layout(
            title=f'Surface 3D - {metric} (Interpolação Spline)',
            scene=dict(
                xaxis_title='Samples',
                yaxis_title='Channels',
                zaxis_title=metric
            ),
            width=1000,
            height=800
        )
        return fig

    def create_surface_with_scatter(self, metric: str = "val_loss", resolution: int = 30) -> go.Figure:
        """
        Cria superfície 3D com pontos de dados sobrepostos.
        
        Args:
            metric: Métrica para Z
            resolution: Resolução da grid
        
        Returns:
            Plotly Figure
        """
        # Dados originais
        x = self.df['samples'].values.astype(float)
        y = self.df['channels'].values.astype(float)
        z = self.df[metric].values.astype(float)
        acc = self.df['test_accuracy'].values.astype(float)
        
        # Grid para interpolação
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolar usando RBF
        rbf = Rbf(x, y, z, function='thin_plate', smooth=0.1)
        zi = rbf(xi_grid, yi_grid)
        
        # Criar figura com superfície
        fig = go.Figure(data=[
            # Superfície interpolada
            go.Surface(
                x=xi,
                y=yi,
                z=zi,
                colorscale='Viridis',
                name='Superfície',
                opacity=0.7,
                showscale=True,
                colorbar=dict(title=metric, x=1.02)
            ),
            # Pontos de dados originais
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=6,
                    color=acc,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title='Test Accuracy', x=1.15)
                ),
                text=[f"Ch: {c}, Samples: {s}, {metric}: {m:.4f}, Acc: {a:.4f}" 
                      for c, s, m, a in zip(y, x, z, acc)],
                hovertemplate='%{text}<extra></extra>',
                name='Dados'
            )
        ])
        
        fig.update_layout(
            title=f'Surface + Data - {metric}',
            scene=dict(
                xaxis_title='Samples',
                yaxis_title='Channels',
                zaxis_title=metric
            ),
            width=1100,
            height=800
        )
        return fig


def main():
    """Menu principal."""
    csv_file = r"C:\Users\Chari\Documents\dev\BrainBridge\tools\notebook\latin_square_results\latin_square_all_results_20251211_004451.csv"
    
    viz = LatinSquare3DVisualizer(csv_file)
    
    print("=" * 50)
    print("3D VISUALIZER - LATIN SQUARE (Simplificado)")
    print("=" * 50)
    print("\n1. Scatter 3D")
    print("2. Superfície com Interpolação")
    print("3. Superfície + Pontos de Dados")
    print("0. Sair\n")
    
    while True:
        choice = input("Escolha (0-3): ").strip()
        
        if choice == "1":
            print("\n📊 Criando scatter 3D...")
            metric = input("Métrica (val_loss/train_loss/test_accuracy) [val_loss]: ").strip() or "val_loss"
            fig = viz.create_3d_scatter(metric=metric)
            fig.show()
            
        elif choice == "2":
            print("\n📊 Criando superfície...")
            metric = input("Métrica (val_loss/train_loss/test_accuracy) [val_loss]: ").strip() or "val_loss"
            resolution = input("Resolução (20-100) [50]: ").strip()
            resolution = int(resolution) if resolution else 50
            fig = viz.create_surface(metric=metric, resolution=resolution)
            fig.show()
            
        elif choice == "3":
            print("\n📊 Criando superfície + dados...")
            metric = input("Métrica (val_loss/train_loss/test_accuracy) [val_loss]: ").strip() or "val_loss"
            resolution = input("Resolução (20-100) [50]: ").strip()
            resolution = int(resolution) if resolution else 50
            fig = viz.create_surface_with_scatter(metric=metric, resolution=resolution)
            fig.show()
            
        elif choice == "0":
            print("\nEncerrando...")
            break
        else:
            print("❌ Opção inválida!")


if __name__ == "__main__":
    main()
