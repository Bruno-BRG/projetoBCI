"""
Tema visual para BrainBridge - Verde claro e ebrance com acentos em verde escuro
Paleta profissional para sistema BCI
"""

class Theme:
    """Definição de cores e estilos da aplicação"""
    
    # Cores principais - Paleta Verde Profissional
    PRIMARY_DARK_GREEN = "#1B4D2E"      # Verde escuro (acentos e destaques)
    PRIMARY_GREEN = "#2D7A4A"            # Verde principal (botões primários)
    SECONDARY_GREEN = "#4CAF7F"          # Verde médio (hover e interações)
    LIGHT_GREEN = "#A8D8C8"              # Verde claro (bordas e backgrounds)
    VERY_LIGHT_GREEN = "#E8F5F0"         # Verde muito claro (backgrounds gerais)
    
    # Paleta Neutra - Ebrance
    WHITE = "#FFFFFF"
    CREAM = "#F5F5F0"                    # Ebrance claro (fundo principal)
    DARK_TEXT = "#2C3E50"                # Texto escuro
    LIGHT_GRAY = "#ECEFF1"
    BORDER_COLOR = "#B0BEC5"
    
    # Cores para feedback
    SUCCESS_COLOR = "#27AE60"
    WARNING_COLOR = "#F39C12"
    ERROR_COLOR = "#E74C3C"
    INFO_COLOR = "#3498DB"
    
    # Tamanhos
    BORDER_RADIUS = 6
    PADDING_SMALL = "4px"
    PADDING_DEFAULT = "8px"
    PADDING_LARGE = "12px"
    
    @staticmethod
    def get_stylesheet():
        """Retorna o stylesheet completo da aplicação com design refinado"""
        return f"""
        /* ========== WIDGETS PRINCIPAIS ========== */
        QMainWindow {{
            background-color: {Theme.CREAM};
        }}
        
        QWidget {{
            background-color: {Theme.CREAM};
            color: {Theme.DARK_TEXT};
        }}
        
        /* ========== TÍTULO E HEADERS ========== */
        QLabel {{
            color: {Theme.DARK_TEXT};
        }}
        
        QMainWindow QLabel[title="true"] {{
            color: {Theme.PRIMARY_DARK_GREEN};
            font-size: 18pt;
            font-weight: bold;
            padding: 10px;
        }}
        
        /* ========== INPUTS - DESIGN REFINADO ========== */
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {{
            background-color: {Theme.WHITE};
            color: {Theme.DARK_TEXT};
            border: 1px solid {Theme.LIGHT_GREEN};
            border-radius: 4px;
            padding: 5px 8px;
            font-size: 10pt;
            selection-background-color: {Theme.PRIMARY_GREEN};
            selection-color: white;
            margin: 1px;
            min-height: 24px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, 
        QDateEdit:focus, QTimeEdit:focus {{
            border: 1px solid {Theme.PRIMARY_GREEN};
            background-color: {Theme.VERY_LIGHT_GREEN};
            outline: none;
        }}
        
        QLineEdit:hover, QTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover,
        QDateEdit:hover, QTimeEdit:hover {{
            border: 1px solid {Theme.SECONDARY_GREEN};
            background-color: {Theme.WHITE};
        }}
        
        /* ========== COMBOBOX ========== */
        QComboBox {{
            background-color: {Theme.WHITE};
            color: {Theme.DARK_TEXT};
            border: 1px solid {Theme.LIGHT_GREEN};
            border-radius: 4px;
            padding: 5px 8px;
            font-size: 10pt;
            margin: 1px;
            min-height: 24px;
        }}
        
        QComboBox:focus {{
            border: 1px solid {Theme.PRIMARY_GREEN};
            background-color: {Theme.VERY_LIGHT_GREEN};
        }}
        
        QComboBox:hover {{
            border: 1px solid {Theme.SECONDARY_GREEN};
        }}
        
        QComboBox::drop-down {{
            border: none;
            background-color: {Theme.VERY_LIGHT_GREEN};
            width: 25px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {Theme.PRIMARY_GREEN};
            margin-right: 8px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {Theme.WHITE};
            color: {Theme.DARK_TEXT};
            selection-background-color: {Theme.PRIMARY_GREEN};
            selection-color: white;
            border: 1px solid {Theme.LIGHT_GREEN};
            border-radius: 4px;
        }}
        
        /* ========== BUTTONS - DESIGN PREMIUM ========== */
        QPushButton {{
            background-color: {Theme.PRIMARY_GREEN};
            color: {Theme.WHITE};
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
            font-size: 10pt;
            min-height: 26px;
            outline: none;
        }}
        
        QPushButton:hover {{
            background-color: {Theme.SECONDARY_GREEN};
        }}
        
        QPushButton:pressed {{
            background-color: {Theme.PRIMARY_DARK_GREEN};
        }}
        
        QPushButton:disabled {{
            background-color: {Theme.BORDER_COLOR};
            color: {Theme.LIGHT_GRAY};
        }}
        
        /* Botões secundários */
        QPushButton#secondaryButton {{
            background-color: {Theme.LIGHT_GREEN};
            color: {Theme.PRIMARY_DARK_GREEN};
            font-weight: bold;
        }}
        
        QPushButton#secondaryButton:hover {{
            background-color: {Theme.SECONDARY_GREEN};
            color: {Theme.WHITE};
        }}
        
        /* Botões de ação destrutiva */
        QPushButton#destructiveButton {{
            background-color: {Theme.ERROR_COLOR};
        }}
        
        QPushButton#destructiveButton:hover {{
            background-color: #C0392B;
        }}
        
        /* ========== GROUPBOX ========== */
        QGroupBox {{
            color: {Theme.PRIMARY_DARK_GREEN};
            border: 1px solid {Theme.LIGHT_GREEN};
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 8px;
            font-weight: bold;
            font-size: 10pt;
            background-color: {Theme.VERY_LIGHT_GREEN};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            background-color: {Theme.VERY_LIGHT_GREEN};
        }}
        
        /* ========== TABS ========== */
        QTabWidget::pane {{
            border: 1px solid {Theme.LIGHT_GREEN};
            background-color: {Theme.WHITE};
            border-radius: 4px;
        }}
        
        QTabBar::tab {{
            background-color: {Theme.VERY_LIGHT_GREEN};
            color: {Theme.DARK_TEXT};
            padding: 8px 20px;
            margin-right: 2px;
            border: 1px solid {Theme.LIGHT_GREEN};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            font-weight: bold;
            font-size: 10pt;
        }}
        
        QTabBar::tab:hover {{
            background-color: {Theme.LIGHT_GREEN};
        }}
        
        QTabBar::tab:selected {{
            background-color: {Theme.PRIMARY_GREEN};
            color: {Theme.WHITE};
            border: 1px solid {Theme.PRIMARY_GREEN};
        }}
        
        /* ========== TABELAS ========== */
        QTableWidget {{
            background-color: {Theme.WHITE};
            alternate-background-color: {Theme.VERY_LIGHT_GREEN};
            gridline-color: {Theme.LIGHT_GREEN};
            border: 1px solid {Theme.LIGHT_GREEN};
            border-radius: 4px;
            font-size: 9pt;
        }}
        
        QTableWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {Theme.LIGHT_GREEN};
        }}
        
        QTableWidget::item:selected {{
            background-color: {Theme.SECONDARY_GREEN};
            color: {Theme.WHITE};
        }}
        
        QHeaderView::section {{
            background-color: {Theme.PRIMARY_GREEN};
            color: {Theme.WHITE};
            padding: 4px;
            border: none;
            font-weight: bold;
        }}
        
        /* ========== SCROLLBAR ========== */
        QScrollBar:vertical {{
            background-color: {Theme.VERY_LIGHT_GREEN};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {Theme.LIGHT_GREEN};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {Theme.SECONDARY_GREEN};
        }}
        
        QScrollBar:horizontal {{
            background-color: {Theme.VERY_LIGHT_GREEN};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {Theme.LIGHT_GREEN};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {Theme.SECONDARY_GREEN};
        }}
        
        /* ========== STATUSBAR ========== */
        QStatusBar {{
            background-color: {Theme.PRIMARY_DARK_GREEN};
            color: {Theme.WHITE};
            border-top: 2px solid {Theme.LIGHT_GREEN};
            padding: 5px;
        }}
        
        /* ========== DIALOGS ========== */
        QDialog {{
            background-color: {Theme.CREAM};
        }}
        
        QMessageBox {{
            background-color: {Theme.CREAM};
        }}
        
        QMessageBox QLabel {{
            color: {Theme.DARK_TEXT};
        }}
        
        /* ========== SPINBOX / ARROWS ========== */
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            background-color: {Theme.VERY_LIGHT_GREEN};
            border: 1px solid {Theme.LIGHT_GREEN};
            border-bottom: none;
            width: 20px;
        }}
        
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: {Theme.VERY_LIGHT_GREEN};
            border: 1px solid {Theme.LIGHT_GREEN};
            border-top: none;
            width: 20px;
        }}
        
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {Theme.LIGHT_GREEN};
        }}
        
        /* ========== CHECKBOX & RADIO ========== */
        QCheckBox, QRadioButton {{
            color: {Theme.DARK_TEXT};
            spacing: 8px;
            margin: 2px;
        }}
        
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 18px;
            height: 18px;
        }}
        
        QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {{
            background-color: {Theme.WHITE};
            border: 2px solid {Theme.LIGHT_GREEN};
            border-radius: 3px;
        }}
        
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {Theme.PRIMARY_GREEN};
            border: 2px solid {Theme.PRIMARY_GREEN};
            border-radius: 3px;
        }}
        
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
            border: 2px solid {Theme.SECONDARY_GREEN};
        }}
        
        /* ========== PROGRESSBAR ========== */
        QProgressBar {{
            background-color: {Theme.VERY_LIGHT_GREEN};
            border: 2px solid {Theme.LIGHT_GREEN};
            border-radius: {Theme.BORDER_RADIUS}px;
            text-align: center;
            color: {Theme.DARK_TEXT};
            padding: 2px;
        }}
        
        QProgressBar::chunk {{
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {Theme.PRIMARY_GREEN}, stop:1 {Theme.SECONDARY_GREEN});
            border-radius: 4px;
        }}
        
        /* ========== TOOLTIPS ========== */
        QToolTip {{
            background-color: {Theme.PRIMARY_DARK_GREEN};
            color: {Theme.WHITE};
            border: 1px solid {Theme.PRIMARY_GREEN};
            border-radius: 4px;
            padding: 4px;
        }}
        
        /* ========== MENUBAR E MENUS ========== */
        QMenuBar {{
            background-color: {Theme.CREAM};
            color: {Theme.DARK_TEXT};
            border-bottom: 2px solid {Theme.LIGHT_GREEN};
        }}
        
        QMenuBar::item:selected {{
            background-color: {Theme.LIGHT_GREEN};
        }}
        
        QMenu {{
            background-color: {Theme.WHITE};
            color: {Theme.DARK_TEXT};
            border: 1px solid {Theme.LIGHT_GREEN};
        }}
        
        QMenu::item:selected {{
            background-color: {Theme.PRIMARY_GREEN};
            color: {Theme.WHITE};
        }}
        """


def apply_theme(app):
    """Aplica o tema à aplicação Qt de forma profissional"""
    app.setStyle('Fusion')
    app.setStyleSheet(Theme.get_stylesheet())
