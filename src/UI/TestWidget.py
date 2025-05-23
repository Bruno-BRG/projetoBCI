# Importações de terceiros
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QGroupBox,
    QInputDialog
)
import os
from datetime import datetime

# Importações locais
from model.MultiSubjectTest import MultiSubjectTest

class TestWidget(QWidget):
    """Widget para testes de modelo multi-paciente"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layout principal com padding
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Grupo de plotagem para curvas de treinamento
        plot_group = QGroupBox("Métricas de Treinamento")
        plot_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Grupo de status
        status_group = QGroupBox("Status do Teste")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Pronto para iniciar os testes")
        self.progress_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Grupo de controle
        control_group = QGroupBox("Controles de Teste")
        control_layout = QHBoxLayout()
        self.start_test_button = QPushButton("Iniciar Teste Multi-Paciente")
        self.start_test_button.clicked.connect(self.start_test)
        control_layout.addWidget(self.start_test_button)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        self.setLayout(layout)
        
        # Inicializar sistema de teste
        self.test_system = None
        self.figure.clear()
        self.canvas.draw()

    def update_plot(self, history):
        """Atualizar o gráfico com novas métricas de treinamento"""
        self.figure.clear()
        
        # Criar dois subplots
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # Plotar perdas
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], 'b-', label='Perda de Treinamento')
        ax1.plot(epochs, history['val_losses'], 'r-', label='Perda de Validação')
        ax1.set_title('Perda de Treinamento e Validação')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Perda')
        ax1.grid(True)
        ax1.legend()
        
        # Plotar acurácias
        ax2.plot(epochs, history['train_accs'], 'b-', label='Acurácia de Treinamento')
        ax2.plot(epochs, history['val_accs'], 'r-', label='Acurácia de Validação')
        ax2.set_title('Acurácia de Treinamento e Validação')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.grid(True)
        ax2.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()

    def start_test(self):
        """Iniciar o processo de teste multi-paciente"""
        self.start_test_button.setEnabled(False)
        self.status_label.setText("Preparando conjuntos de dados...")
        self.progress_label.setText("Isso pode levar alguns minutos...")
        
        # Perguntar pelo nome do modelo
        name, ok = QInputDialog.getText(self, "Nome do Modelo", "Digite o nome para o modelo multi-paciente:", text=datetime.now().strftime("multipaciente_%Y%m%d_%H%M%S"))
        if not ok or not name.strip():
            self.status_label.setText("Teste cancelado: nenhum nome de modelo fornecido")
            self.start_test_button.setEnabled(True)
            return
            
        # Ask about fast mode
        fast_mode_reply = QMessageBox.question(self, 'Modo Rápido', 
                                    'Deseja usar o modo rápido? (Recomendado para CPU)',
                                    QMessageBox.Yes | QMessageBox.No, 
                                    QMessageBox.Yes)
        fast_mode = fast_mode_reply == QMessageBox.Yes
        
        model_filename = f"{name}.pth"
        os.makedirs('checkpoints', exist_ok=True)
        model_path = os.path.join('checkpoints', model_filename)
        try:
            # Inicializar sistema de teste com o caminho de salvamento fornecido
            self.test_system = MultiSubjectTest(
                train_samples=40, 
                test_samples=20, 
                model_path=model_path,
                fast_mode=fast_mode
            )
             
            # Set number of epochs based on fast mode
            num_epochs = 30 if fast_mode else 100
            
            # Executar treinamento e avaliação
            history = self.test_system.train_and_evaluate(
                num_epochs=num_epochs,
                batch_size=32 if fast_mode else 10,  # Larger batch size for CPU
                learning_rate=5e-4
            )
            
            # Atualizar gráfico com os resultados
            self.update_plot(history)
            
            # Atualizar status
            final_train_acc = history['train_accs'][-1]
            final_val_acc = history['val_accs'][-1]
            self.status_label.setText("Teste concluído com sucesso!")
            self.progress_label.setText(
                f"Resultados Finais:\n"
                f"Acurácia de Treinamento: {final_train_acc:.2%}\n"
                f"Acurácia de Validação: {final_val_acc:.2%}\n"
                f"Modelo salvo em: {model_path}"
            )
         
        except Exception as e:
            self.status_label.setText("Erro durante o teste")
            self.progress_label.setText(str(e))
        finally:
            self.start_test_button.setEnabled(True)
