from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel, 
                           QLineEdit, QDateEdit, QTextEdit, QPushButton, 
                           QGroupBox, QTableWidget, QTableWidgetItem, 
                           QHeaderView, QMessageBox, QHBoxLayout, 
                           QComboBox, QSpinBox, QScrollArea)
from PyQt5.QtCore import QDate
from PyQt5.QtGui import QFont
from typing import Optional
from brainbridge_v2.interface_adapters.controllers.patient_controller import PatientController
from brainbridge_v2.presentation.gui.styles import Theme

class PatientRegistrationWidget(QWidget):
    """Widget para cadastro de pacientes"""
    
    def __init__(self, patient_controller: PatientController, parent=None):
        super().__init__(parent)
        self.patient_controller = patient_controller
        self.setup_ui()
        self.load_patients()
        
    def setup_ui(self):
        """Configura a interface - Formulário à esquerda, Tabela à direita"""
        # Layout principal horizontal
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)
        
        # ========== PAINEL ESQUERDO - FORMULÁRIO ==========
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        # Formulário de cadastro
        form_group = QGroupBox("📝 Novo Paciente")
        form_group.setFont(QFont("Arial", 10, QFont.Bold))
        form_layout = QVBoxLayout()
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(12, 16, 12, 12)
        
        # Estilos para labels
        label_style = f"color: {Theme.PRIMARY_DARK_GREEN}; font-weight: bold; font-size: 9pt;"
        
        # Campo: Nome Completo
        name_label = QLabel("Nome Completo:")
        name_label.setStyleSheet(label_style)
        form_layout.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Digite o nome do paciente...")
        self.name_edit.setMinimumHeight(32)
        form_layout.addWidget(self.name_edit)
        
        # Linha: Idade e Sexo
        row1 = QHBoxLayout()
        row1.setSpacing(12)
        
        # Idade
        age_container = QVBoxLayout()
        age_label = QLabel("Idade:")
        age_label.setStyleSheet(label_style)
        age_container.addWidget(age_label)
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        self.age_spin.setValue(30)
        self.age_spin.setMinimumHeight(32)
        self.age_spin.setMaximumWidth(80)
        age_container.addWidget(self.age_spin)
        row1.addLayout(age_container)
        
        # Sexo
        sex_container = QVBoxLayout()
        sex_label = QLabel("Sexo:")
        sex_label.setStyleSheet(label_style)
        sex_container.addWidget(sex_label)
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Masculino", "Feminino", "Outro"])
        self.sex_combo.setMinimumHeight(32)
        sex_container.addWidget(self.sex_combo)
        row1.addLayout(sex_container, 1)
        
        form_layout.addLayout(row1)
        
        # Linha: Mão Afetada e Tempo desde evento
        row2 = QHBoxLayout()
        row2.setSpacing(12)
        
        # Mão Afetada
        hand_container = QVBoxLayout()
        hand_label = QLabel("Mão Afetada:")
        hand_label.setStyleSheet(label_style)
        hand_container.addWidget(hand_label)
        self.hand_combo = QComboBox()
        self.hand_combo.addItems(["Esquerda", "Direita", "Ambas", "Nenhuma"])
        self.hand_combo.setMinimumHeight(32)
        hand_container.addWidget(self.hand_combo)
        row2.addLayout(hand_container, 1)
        
        # Tempo desde evento
        time_container = QVBoxLayout()
        time_label = QLabel("Tempo (meses):")
        time_label.setStyleSheet(label_style)
        time_container.addWidget(time_label)
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 1000)
        self.time_spin.setValue(0)
        self.time_spin.setMinimumHeight(32)
        self.time_spin.setMaximumWidth(100)
        time_container.addWidget(self.time_spin)
        row2.addLayout(time_container)
        
        form_layout.addLayout(row2)
        
        # Campo: Observações
        notes_label = QLabel("Observações:")
        notes_label.setStyleSheet(label_style)
        form_layout.addWidget(notes_label)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMinimumHeight(60)
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.setPlaceholderText("Notas adicionais sobre o paciente...")
        form_layout.addWidget(self.notes_edit)
        
        # Botão de cadastro
        self.register_btn = QPushButton("✓ Cadastrar Paciente")
        self.register_btn.setMinimumHeight(36)
        self.register_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.register_btn.clicked.connect(self.register_patient)
        form_layout.addWidget(self.register_btn)
        
        # Espaçador para empurrar conteúdo para cima
        form_layout.addStretch()
        
        form_group.setLayout(form_layout)
        left_layout.addWidget(form_group)
        
        # Definir largura fixa para o painel esquerdo
        left_panel.setFixedWidth(320)
        
        # ========== PAINEL DIREITO - TABELA ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        patients_group = QGroupBox("👥 Pacientes Cadastrados")
        patients_group.setFont(QFont("Arial", 10, QFont.Bold))
        patients_layout = QVBoxLayout()
        patients_layout.setSpacing(8)
        patients_layout.setContentsMargins(8, 16, 8, 8)
        
        self.patients_table = QTableWidget()
        self.patients_table.setColumnCount(7)
        self.patients_table.setHorizontalHeaderLabels([
            "ID", "Nome", "Idade", "Sexo", "Mão Afetada", "Tempo", "Cadastro"
        ])
        self.patients_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.patients_table.setAlternatingRowColors(True)
        
        # Configurar header para expandir colunas
        header = self.patients_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.Stretch)           # Nome (expande)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Idade
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Sexo
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Mão
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Tempo
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Data
        
        # Ajustar altura das linhas
        self.patients_table.verticalHeader().setDefaultSectionSize(28)
        self.patients_table.verticalHeader().setVisible(False)
        
        patients_layout.addWidget(self.patients_table)
        patients_group.setLayout(patients_layout)
        right_layout.addWidget(patients_group)
        
        # ========== ADICIONAR PAINÉIS AO LAYOUT PRINCIPAL ==========
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # stretch=1 para expandir
        
        self.setLayout(main_layout)
        
    def register_patient(self):
        """Registra um novo paciente"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "⚠ Erro de Validação", "Nome é obrigatório!")
            return
            
        age = self.age_spin.value()
        sex = self.sex_combo.currentText()
        affected_hand = self.hand_combo.currentText()
        time_since_event = self.time_spin.value()
        notes = self.notes_edit.toPlainText()
        
        try:
            patient_id = self.patient_controller.register_patient(
                {
                    "name": name,
                    "age": age,
                    "sex": sex,
                    "affected_hand": affected_hand,
                    "time_since_event": time_since_event,
                    "notes": notes,
                }
            )
            
            QMessageBox.information(self, "✓ Sucesso", 
                                  f"Paciente {name} cadastrado com ID {patient_id}")
            
            # Limpar formulário
            self.name_edit.clear()
            self.age_spin.setValue(30)
            self.sex_combo.setCurrentIndex(0)
            self.hand_combo.setCurrentIndex(0)
            self.time_spin.setValue(0)
            self.notes_edit.clear()
            
            # Recarregar tabela
            self.load_patients()
            
        except Exception as e:
            QMessageBox.critical(self, "❌ Erro", f"Erro ao cadastrar paciente: {e}")
    
    def load_patients(self):
        """Carrega a lista de pacientes"""
        try:
            patients = self.patient_controller.list_patients()
            
            self.patients_table.setRowCount(len(patients))
            
            for row, patient in enumerate(patients):
                self.patients_table.setItem(row, 0, QTableWidgetItem(str(patient["id"])))
                self.patients_table.setItem(row, 1, QTableWidgetItem(patient["name"]))
                self.patients_table.setItem(row, 2, QTableWidgetItem(str(patient["age"])))
                self.patients_table.setItem(row, 3, QTableWidgetItem(patient["sex"]))
                self.patients_table.setItem(row, 4, QTableWidgetItem(patient["affected_hand"]))
                self.patients_table.setItem(row, 5, QTableWidgetItem(str(patient["time_since_event"])))
                created_at = patient.get("created_at", "")
                self.patients_table.setItem(row, 6, QTableWidgetItem(created_at[:10] if created_at else ""))
            
        except Exception as e:
            QMessageBox.critical(self, "❌ Erro", f"Erro ao carregar pacientes: {e}")
    
    def get_selected_patient(self) -> Optional[int]:
        """Retorna o ID do paciente selecionado"""
        current_row = self.patients_table.currentRow()
        if current_row >= 0:
            return int(self.patients_table.item(current_row, 0).text())
        return None
