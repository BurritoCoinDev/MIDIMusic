; Inno Setup script for MIDIMusic.
;
; The installer ships the application only. PyTorch and model weights are
; fetched afterwards from inside the app, because the right torch build depends
; on the user's GPU and the weights are measured in gigabytes.

#define AppName "MIDIMusic"
#define AppVersion "0.1.0"
#define AppPublisher "MIDIMusic"
#define AppExeName "MIDIMusic.exe"
#define AppURL "https://github.com/BurritoCoinDev/MIDIMusic"

[Setup]
AppId={{7F3C2A54-9E1B-4D6A-9C42-4B8E5D1A7C33}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir=Output
OutputBaseFilename=MIDIMusic-{#AppVersion}-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The app is 64-bit only; Qt and the audio libraries have no 32-bit story here.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per-user install by default so no administrator prompt is needed. Model
; downloads land in the user's profile anyway.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExeName}
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\..\dist\MIDIMusic\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\MIDIMusic\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "..\..\THIRD_PARTY_LICENSES.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Caches and logs are ours to clean up. Generated music and downloaded models
; are the user's, and are deliberately left alone.
Type: filesandordirs; Name: "{localappdata}\{#AppName}\cache"
Type: filesandordirs; Name: "{localappdata}\{#AppName}\logs"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    { The app provisions PyTorch itself from the Models tab, so setup only has
      to tell the user that step exists. }
  end;
end;
