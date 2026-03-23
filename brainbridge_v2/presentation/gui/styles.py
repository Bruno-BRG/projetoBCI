"""
Tema visual para BrainBridge - Réplica pixel-perfect do bci_system.html
Paleta: Azul escuro (#0a0e27), Painel (#111640), Azul (#3b5bdb)
"""


class Theme:
    """Definição de cores e estilos - Idêntico ao bci_system.html"""

    # Cores principais do HTML
    BG_DARK = "#0a0e27"           # body background
    PANEL_BG = "#111640"          # main-panel background
    BLUE = "#3b5bdb"              # cor primária (tabs, bordas, botões)
    BLUE_HOVER = "#364fc7"        # hover azul
    BLUE_TRANS = "rgba(59, 91, 219, 0.3)"  # tab hover

    # Cores de botões
    BTN_BG = "#e2e8f0"           # .btn background
    BTN_BG_HOVER = "#cbd5e0"     # .btn:hover
    BTN_TEXT = "#1a202c"          # .btn color
    BTN_BORDER = "#4a5568"       # .btn border
    BTN_GREEN = "#38a169"        # .btn-green
    BTN_GREEN_HOVER = "#2f855a"  # .btn-green:hover
    BTN_DARK = "#2d3748"         # .btn-dark
    BTN_DARK_HOVER = "#4a5568"   # .btn-dark:hover

    # Status
    GREEN = "#48bb78"            # conectado/sucesso
    ORANGE = "#f6ad55"           # desconectado/aviso
    GRAY = "#a0aec0"             # off/standby
    LIGHT_BLUE = "#63b3ed"       # mão direita

    # Texto
    WHITE = "#ffffff"
    TEXT_DARK = "#1a202c"
    TEXT_LIGHT_GRAY = "#e2e8f0"

    # Tabela
    TABLE_HEADER_BG = "#e2e8f0"
    TABLE_BODY_BG = "#f7fafc"
    TABLE_BORDER = "#cbd5e0"

    # Scrollbar
    SCROLLBAR_TRACK = "#1a1f4e"
    SCROLLBAR_THUMB = "#3b5bdb"

    # Calibração
    CALIB_BG = "rgba(45, 55, 72, 0.4)"
    CALIB_BORDER = "#4a5568"

    # Tamanhos (mantidos para compatibilidade)
    BORDER_RADIUS = 6
    PADDING_SMALL = "4px"
    PADDING_DEFAULT = "8px"
    PADDING_LARGE = "12px"

    # Aliases para compatibilidade com código existente
    PRIMARY_DARK_GREEN = BLUE
    PRIMARY_GREEN = BTN_GREEN
    SECONDARY_GREEN = GREEN
    LIGHT_GREEN = "#A8D8C8"
    VERY_LIGHT_GREEN = PANEL_BG
    CREAM = BG_DARK
    DARK_TEXT = WHITE
    LIGHT_GRAY = TEXT_LIGHT_GRAY
    BORDER_COLOR = BTN_BORDER
    SUCCESS_COLOR = GREEN
    WARNING_COLOR = ORANGE
    ERROR_COLOR = "#E74C3C"
    INFO_COLOR = LIGHT_BLUE

    @staticmethod
    def get_stylesheet():
        """Retorna o stylesheet completo - Tema azul escuro do bci_system.html"""
        return f"""
        /* ========== WIDGETS PRINCIPAIS ========== */
        QMainWindow {{
            background-color: {Theme.BG_DARK};
        }}

        QWidget {{
            background-color: {Theme.BG_DARK};
            color: {Theme.WHITE};
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        /* ========== LABELS ========== */
        QLabel {{
            color: {Theme.WHITE};
            background-color: transparent;
        }}

        /* ========== INPUTS ========== */
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {{
            background-color: {Theme.BTN_BG};
            color: {Theme.TEXT_DARK};
            border: 1px solid {Theme.BTN_BORDER};
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 13px;
            font-weight: 500;
            min-height: 24px;
        }}

        QLineEdit:focus, QTextEdit:focus {{
            border: 1px solid {Theme.BLUE};
        }}

        /* ========== COMBOBOX ========== */
        QComboBox {{
            background-color: {Theme.BTN_BG};
            color: {Theme.TEXT_DARK};
            border: 1px solid {Theme.BTN_BORDER};
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 13px;
            font-weight: 500;
            min-height: 24px;
        }}

        QComboBox::drop-down {{
            border: none;
            background-color: {Theme.BTN_BG};
            width: 25px;
        }}

        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {Theme.BTN_BORDER};
            margin-right: 8px;
        }}

        QComboBox QAbstractItemView {{
            background-color: {Theme.BTN_BG};
            color: {Theme.TEXT_DARK};
            selection-background-color: {Theme.BLUE};
            selection-color: {Theme.WHITE};
            border: 1px solid {Theme.BTN_BORDER};
        }}

        /* ========== BUTTONS ========== */
        QPushButton {{
            background-color: {Theme.BTN_BG};
            color: {Theme.TEXT_DARK};
            border: 1px solid {Theme.BTN_BORDER};
            border-radius: 5px;
            padding: 7px 18px;
            font-weight: 600;
            font-size: 13px;
            min-height: 26px;
        }}

        QPushButton:hover {{
            background-color: {Theme.BTN_BG_HOVER};
        }}

        QPushButton:pressed {{
            background-color: {Theme.BTN_BORDER};
        }}

        QPushButton:disabled {{
            background-color: {Theme.BTN_DARK};
            color: {Theme.GRAY};
        }}

        /* ========== GROUPBOX ========== */
        QGroupBox {{
            color: {Theme.WHITE};
            border: 2px solid {Theme.BTN_BORDER};
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
            font-weight: bold;
            font-size: 10pt;
            background-color: {Theme.CALIB_BG};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            color: {Theme.WHITE};
        }}

        /* ========== TABS ========== */
        QTabWidget::pane {{
            border: 2px solid {Theme.BLUE};
            border-radius: 0 8px 8px 8px;
            background-color: {Theme.PANEL_BG};
        }}

        QTabBar::tab {{
            background-color: transparent;
            color: {Theme.WHITE};
            padding: 8px 18px;
            border: 2px solid {Theme.BLUE};
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: 600;
            font-size: 14px;
        }}

        QTabBar::tab:hover {{
            background-color: {Theme.BLUE_TRANS};
        }}

        QTabBar::tab:selected {{
            background-color: {Theme.BLUE};
            color: {Theme.WHITE};
        }}

        /* ========== TABELAS ========== */
        QTableWidget {{
            background-color: {Theme.TABLE_BODY_BG};
            alternate-background-color: {Theme.WHITE};
            gridline-color: {Theme.TABLE_BORDER};
            border: 1px solid {Theme.TABLE_BORDER};
            font-size: 13px;
            color: {Theme.TEXT_DARK};
        }}

        QTableWidget::item {{
            padding: 7px 12px;
            border: 1px solid {Theme.TABLE_BORDER};
            color: {Theme.TEXT_DARK};
        }}

        QTableWidget::item:selected {{
            background-color: {Theme.BLUE};
            color: {Theme.WHITE};
        }}

        QHeaderView::section {{
            background-color: {Theme.TABLE_HEADER_BG};
            color: {Theme.TEXT_DARK};
            padding: 8px 12px;
            border: 1px solid {Theme.TABLE_BORDER};
            font-weight: 800;
            font-size: 13px;
            text-transform: uppercase;
        }}

        /* ========== SCROLLBAR ========== */
        QScrollBar:vertical {{
            background-color: {Theme.SCROLLBAR_TRACK};
            width: 10px;
            border-radius: 5px;
        }}

        QScrollBar::handle:vertical {{
            background-color: {Theme.SCROLLBAR_THUMB};
            border-radius: 5px;
            min-height: 20px;
        }}

        QScrollBar:horizontal {{
            background-color: {Theme.SCROLLBAR_TRACK};
            height: 10px;
            border-radius: 5px;
        }}

        QScrollBar::handle:horizontal {{
            background-color: {Theme.SCROLLBAR_THUMB};
            border-radius: 5px;
            min-width: 20px;
        }}

        QScrollBar::add-line, QScrollBar::sub-line {{
            height: 0px;
            width: 0px;
        }}

        /* ========== STATUSBAR ========== */
        QStatusBar {{
            background-color: {Theme.PANEL_BG};
            color: {Theme.WHITE};
            border-top: 2px solid {Theme.BLUE};
            padding: 5px;
        }}

        /* ========== DIALOGS ========== */
        QDialog {{
            background-color: {Theme.BG_DARK};
        }}

        QMessageBox {{
            background-color: {Theme.BG_DARK};
        }}

        QMessageBox QLabel {{
            color: {Theme.WHITE};
        }}

        /* ========== TEXTAREA ========== */
        QTextEdit {{
            background-color: {Theme.BTN_BG};
            color: {Theme.TEXT_DARK};
            border: 1px solid {Theme.BTN_BORDER};
            border-radius: 4px;
            padding: 10px;
            font-size: 13px;
            font-weight: 500;
        }}

        /* ========== CHECKBOX ========== */
        QCheckBox, QRadioButton {{
            color: {Theme.WHITE};
            spacing: 8px;
        }}

        QCheckBox::indicator, QRadioButton::indicator {{
            width: 18px;
            height: 18px;
        }}

        QCheckBox::indicator:unchecked {{
            background-color: {Theme.BTN_BG};
            border: 2px solid {Theme.BTN_BORDER};
            border-radius: 3px;
        }}

        QCheckBox::indicator:checked {{
            background-color: {Theme.BLUE};
            border: 2px solid {Theme.BLUE};
            border-radius: 3px;
        }}

        /* ========== PROGRESSBAR ========== */
        QProgressBar {{
            background-color: {Theme.BTN_DARK};
            border: 2px solid {Theme.BTN_BORDER};
            border-radius: 6px;
            text-align: center;
            color: {Theme.WHITE};
        }}

        QProgressBar::chunk {{
            background-color: {Theme.BLUE};
            border-radius: 4px;
        }}

        /* ========== TOOLTIPS ========== */
        QToolTip {{
            background-color: {Theme.PANEL_BG};
            color: {Theme.WHITE};
            border: 1px solid {Theme.BLUE};
            border-radius: 4px;
            padding: 4px;
        }}

        /* ========== MENUBAR ========== */
        QMenuBar {{
            background-color: {Theme.BG_DARK};
            color: {Theme.WHITE};
            border-bottom: 2px solid {Theme.BLUE};
        }}

        QMenuBar::item:selected {{
            background-color: {Theme.BLUE};
        }}

        QMenu {{
            background-color: {Theme.PANEL_BG};
            color: {Theme.WHITE};
            border: 1px solid {Theme.BLUE};
        }}

        QMenu::item:selected {{
            background-color: {Theme.BLUE};
            color: {Theme.WHITE};
        }}

        /* ========== SPINBOX ========== */
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            background-color: {Theme.BTN_BG};
            border: 1px solid {Theme.BTN_BORDER};
            width: 20px;
        }}

        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: {Theme.BTN_BG};
            border: 1px solid {Theme.BTN_BORDER};
            width: 20px;
        }}
        """


def apply_theme(app):
    """Aplica o tema à aplicação Qt"""
    app.setStyle('Fusion')
    app.setStyleSheet(Theme.get_stylesheet())
