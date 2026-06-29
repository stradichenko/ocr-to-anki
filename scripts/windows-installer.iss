; Inno Setup installer script for OCR to Anki
; Usage:
;   iscc scripts/windows-installer.iss \
;     /DMyAppVersion=0.4.15 \
;     /DSourceDir=C:\path\to\output\release\ocr-to-anki-v0.4.15-windows-x86_64

#define MyAppName "OCR to Anki"
#define MyAppPublisher "stradichenko"
#define MyAppURL "https://github.com/stradichenko/ocr-to-anki"
#define MyAppExeName "ocr_to_anki.exe"

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#ifndef SourceDir
  #define SourceDir "..\output\release\ocr-to-anki-v" + MyAppVersion + "-windows-x86_64"
#endif

[Setup]
AppId={{0d472c4b-e72a-44dc-ac16-8d010450b706}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\output\release
OutputBaseFilename=ocr-to-anki-v{#MyAppVersion}-windows-x86_64
SetupIconFile=..\app\windows\runner\resources\app_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Recursively install the Flutter bundle + backend source
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
