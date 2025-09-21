#!/usr/bin/env python3
"""
Demo do Checkbox ESP32 - Interface de Usuário
Demonstra como o checkbox controla o envio serial para ESP32
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QLabel, QGroupBox
from PyQt5.QtCore import Qt

# Adicionar o diretório do projeto ao Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bci.network.esp32_serial_communication import get_esp32_communicator

class ESP32DemoWindow(QMainWindow):
    """Janela de demonstração dos controles ESP32"""
    
    def __init__(self):
        super().__init__()
        self.esp32_communicator = get_esp32_communicator()
        self.esp32_connected = False
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface de demonstração"""
        self.setWindowTitle("Demo - Controles ESP32")
        self.setGeometry(100, 100, 600, 400)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Título
        title_label = QLabel("Demo - Controles ESP32 com Checkbox")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Grupo ESP32
        esp32_group = QGroupBox("Comunicação Serial ESP32")
        esp32_layout = QVBoxLayout()
        
        # Linha 1 - Status e Conexão
        row1 = QHBoxLayout()
        
        self.status_label = QLabel("ESP32: Desconectado")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.connect_btn = QPushButton("Conectar ESP32")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.connect_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        
        row1.addWidget(self.status_label)
        row1.addWidget(self.connect_btn)
        row1.addStretch()
        
        # Linha 2 - Checkbox de Controle
        row2 = QHBoxLayout()
        
        self.auto_send_checkbox = QCheckBox("Envio Serial Automático")
        self.auto_send_checkbox.setChecked(False)
        self.auto_send_checkbox.setToolTip("Quando marcado, comandos TRIGGER são enviados automaticamente para ESP32")
        self.auto_send_checkbox.stateChanged.connect(self.on_checkbox_changed)
        
        self.checkbox_status = QLabel("❌ Envio Serial: DESABILITADO")
        self.checkbox_status.setStyleSheet("color: red; font-weight: bold;")
        
        row2.addWidget(self.auto_send_checkbox)
        row2.addWidget(self.checkbox_status)
        row2.addStretch()
        
        # Linha 3 - Testes Manuais
        row3 = QHBoxLayout()
        
        test_label = QLabel("Teste Manual:")
        self.test_left_btn = QPushButton("🤚 Trigger Esquerdo")
        self.test_left_btn.clicked.connect(lambda: self.test_trigger('esquerda'))
        self.test_left_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.test_left_btn.setEnabled(False)
        
        self.test_right_btn = QPushButton("✋ Trigger Direito")
        self.test_right_btn.clicked.connect(lambda: self.test_trigger('direita'))
        self.test_right_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.test_right_btn.setEnabled(False)
        
        row3.addWidget(test_label)
        row3.addWidget(self.test_left_btn)
        row3.addWidget(self.test_right_btn)
        row3.addStretch()
        
        # Linha 4 - Simulação de Triggers Automáticos
        row4 = QHBoxLayout()
        
        auto_label = QLabel("Simulação Automática:")
        self.auto_left_btn = QPushButton("Simular T1 (Esquerda)")
        self.auto_left_btn.clicked.connect(lambda: self.simulate_automatic_trigger('esquerda'))
        self.auto_left_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.auto_right_btn = QPushButton("Simular T2 (Direita)")
        self.auto_right_btn.clicked.connect(lambda: self.simulate_automatic_trigger('direita'))
        self.auto_right_btn.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold;")
        
        row4.addWidget(auto_label)
        row4.addWidget(self.auto_left_btn)
        row4.addWidget(self.auto_right_btn)
        row4.addStretch()
        
        # Adicionar ao layout do grupo
        esp32_layout.addLayout(row1)
        esp32_layout.addLayout(row2)
        esp32_layout.addLayout(row3)
        esp32_layout.addLayout(row4)
        
        esp32_group.setLayout(esp32_layout)
        layout.addWidget(esp32_group)
        
        # Área de log
        log_group = QGroupBox("Log de Atividades")
        log_layout = QVBoxLayout()
        
        self.log_label = QLabel("Aguardando ações...")
        self.log_label.setStyleSheet("font-family: monospace; padding: 10px; background-color: #f0f0f0; border: 1px solid #ccc;")
        self.log_label.setWordWrap(True)
        self.log_label.setMinimumHeight(150)
        
        log_layout.addWidget(self.log_label)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        central_widget.setLayout(layout)
        
    def toggle_connection(self):
        """Toggle da conexão ESP32"""
        if not self.esp32_connected:
            # Conectar
            if self.esp32_communicator.connect():
                self.esp32_connected = True
                self.status_label.setText("ESP32: Conectado (COM4)")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
                self.connect_btn.setText("Desconectar ESP32")
                self.connect_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
                self.test_left_btn.setEnabled(True)
                self.test_right_btn.setEnabled(True)
                self.add_log("✅ ESP32 conectado com sucesso!")
            else:
                self.add_log("❌ Falha ao conectar ESP32 - Verifique COM4")
        else:
            # Desconectar
            self.esp32_communicator.disconnect()
            self.esp32_connected = False
            self.status_label.setText("ESP32: Desconectado")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.connect_btn.setText("Conectar ESP32")
            self.connect_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
            self.test_left_btn.setEnabled(False)
            self.test_right_btn.setEnabled(False)
            self.add_log("🔌 ESP32 desconectado")
    
    def on_checkbox_changed(self, state):
        """Callback quando checkbox muda"""
        if state == Qt.Checked:
            self.checkbox_status.setText("✅ Envio Serial: HABILITADO")
            self.checkbox_status.setStyleSheet("color: green; font-weight: bold;")
            self.add_log("📋 Checkbox marcado - Envio serial automático HABILITADO")
        else:
            self.checkbox_status.setText("❌ Envio Serial: DESABILITADO")
            self.checkbox_status.setStyleSheet("color: red; font-weight: bold;")
            self.add_log("📋 Checkbox desmarcado - Envio serial automático DESABILITADO")
    
    def test_trigger(self, direction):
        """Teste manual de trigger"""
        if self.esp32_connected:
            if direction == 'esquerda':
                success = self.esp32_communicator.send_trigger_left()
            else:
                success = self.esp32_communicator.send_trigger_right()
            
            side = "esquerdo" if direction == 'esquerda' else "direito"
            if success:
                self.add_log(f"🧪 Teste manual - Trigger {side} enviado com SUCESSO")
            else:
                self.add_log(f"🧪 Teste manual - FALHA ao enviar trigger {side}")
        else:
            self.add_log("⚠️ ESP32 não conectado - teste manual cancelado")
    
    def simulate_automatic_trigger(self, direction):
        """Simula o comportamento do sistema BCI com checkbox"""
        side = "esquerdo" if direction == 'esquerda' else "direito"
        
        # Esta é a lógica exata do sistema BCI
        if self.esp32_connected and self.auto_send_checkbox.isChecked():
            if direction == 'esquerda':
                success = self.esp32_communicator.send_trigger_left()
            else:
                success = self.esp32_communicator.send_trigger_right()
            
            if success:
                self.add_log(f"🤖 Trigger automático {side} - ENVIADO (checkbox habilitado)")
            else:
                self.add_log(f"🤖 Trigger automático {side} - FALHA no envio")
        else:
            # Explicar por que não foi enviado
            if not self.esp32_connected:
                self.add_log(f"🤖 Trigger automático {side} - NÃO ENVIADO (ESP32 desconectado)")
            elif not self.auto_send_checkbox.isChecked():
                self.add_log(f"🤖 Trigger automático {side} - NÃO ENVIADO (checkbox desmarcado)")
    
    def add_log(self, message):
        """Adiciona mensagem ao log"""
        current_log = self.log_label.text()
        if current_log == "Aguardando ações...":
            new_log = message
        else:
            # Manter apenas as últimas 10 linhas
            lines = current_log.split('\n')
            lines.append(message)
            if len(lines) > 10:
                lines = lines[-10:]
            new_log = '\n'.join(lines)
        
        self.log_label.setText(new_log)

def main():
    """Função principal"""
    app = QApplication(sys.argv)
    
    print("🚀 Demo - Controles ESP32 com Checkbox")
    print("=" * 50)
    print("Esta demonstração mostra como o checkbox controla o envio serial")
    print("Funcionalidades:")
    print("1. Conectar/Desconectar ESP32")
    print("2. Checkbox para habilitar/desabilitar envio automático")
    print("3. Testes manuais (sempre funcionam se conectado)")
    print("4. Simulação de triggers automáticos (respeitam checkbox)")
    print("=" * 50)
    
    window = ESP32DemoWindow()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())