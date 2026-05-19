; ==========================================================================
; BrainBridge – Inno Setup Script
; Gera um instalador profissional Windows a partir da pasta dist/BrainBridge
; ==========================================================================
; Compile com: ISCC.exe installer.iss
; Ou use o script build_installer.py

#define MyAppName "BrainBridge"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "BrainBridge Team"
#define MyAppURL "https://github.com/brainbridge"
#define MyAppExeName "BrainBridge.exe"

[Setup]
AppId={{B7A1D2E3-F456-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Pasta de saída do instalador
OutputDir=Output
OutputBaseFilename=BrainBridge_Setup_{#MyAppVersion}
; Ícone do instalador (descomente e ajuste quando tiver um .ico)
; SetupIconFile=brainbridge.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Requer Windows 10+
MinVersion=10.0
; Instalar para todos os usuários requer elevação
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
; Desinstalação
UninstallDisplayName={#MyAppName}
; UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "portuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na &Área de Trabalho"; GroupDescription: "Ícones adicionais:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "Criar atalho na &Barra de Tarefas"; GroupDescription: "Ícones adicionais:"; Flags: unchecked

[Files]
; Copia toda a pasta dist/BrainBridge para o diretório de instalação
Source: "dist\BrainBridge\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Executar {#MyAppName}"; Flags: nowait postinstall skipifsilent shellexec
