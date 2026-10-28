; Windows installer of ALIEN, built by build-installer.bat. Opened directly in Inno Setup,
; the script packages the IDE build trees of build-windows-ninja.bat.

#if Ver < EncodeVer(6, 3, 0)
  #error Inno Setup 6.3 or newer is required.
#endif

#define RootDir ExtractFileDir(RemoveBackslashUnlessRoot(SourcePath))

#ifndef NvidiaDir
  #define NvidiaDir RootDir + "\build-ninja\Release"
#endif
#ifndef AmdDir
  #define AmdDir RootDir + "\build-ninja-hip\Release"
#endif
#define NvidiaDir RemoveBackslashUnlessRoot(NvidiaDir)
#define AmdDir RemoveBackslashUnlessRoot(AmdDir)

#if !FileExists(NvidiaDir + "\alien.exe")
  #error alien.exe was not found in the NVIDIA build directory. Build ALIEN first.
#endif
#if FileExists(AmdDir + "\alien.exe")
  #define WithAmd
#endif

#define AppVersion ""
#define FileHandle
#sub ReadVersionLine
  #define FileLine = FileRead(FileHandle)
  #if Pos("ProgramVersion = ", FileLine) > 0
    #define public AppVersion = Copy(FileLine, Pos('"', FileLine) + 1, RPos('"', FileLine) - Pos('"', FileLine) - 1)
  #endif
#endsub
#for {FileHandle = FileOpen(RootDir + "\source\Base\Resources.cpp"); FileHandle && !FileEof(FileHandle); ""} ReadVersionLine
#if FileHandle
  #expr FileClose(FileHandle)
#endif
#if AppVersion == ""
  #error The program version was not found in source\Base\Resources.cpp.
#endif
#define DashPos Pos("-", AppVersion)
#define NumericVersion DashPos > 0 ? Copy(AppVersion, 1, DashPos - 1) : AppVersion

[Setup]
AppId={{4B02A122-BEF0-4D7B-9D03-936F8A3902C4}
AppName=ALIEN
AppVersion={#AppVersion}
AppPublisher=Christian Heinemann
AppPublisherURL=https://alien-project.org
AppSupportURL=https://github.com/chrxh/alien/issues
VersionInfoVersion={#NumericVersion}
; ALIEN writes its log and autosaves into its own folder, so it is installed per user
; into a folder the user can write to.
PrivilegesRequired=lowest
DefaultDirName={autopf}\ALIEN
DefaultGroupName=ALIEN
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
SetupIconFile={#RootDir}\resources\images\alien.ico
UninstallDisplayIcon={app}\alien.exe
UninstallDisplayName=ALIEN
OutputDir=build
OutputBaseFilename=alien-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Tasks]
#ifdef WithAmd
Name: nvidia; Description: "NVIDIA, GeForce RTX 20 series or newer"; GroupDescription: "Graphics card:"; Flags: exclusive
Name: amd; Description: "AMD, Radeon RX 6000 series or newer (experimental)"; GroupDescription: "Graphics card:"; Flags: exclusive unchecked
#endif
Name: startmenuicon; Description: "Create a &Start menu entry"; GroupDescription: "Shortcuts:"
Name: desktopicon; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "Shortcuts:"; Flags: unchecked

; ALIEN cannot create savepoints before this folder exists.
[Dirs]
Name: "{app}\resources\autosave"

[Files]
Source: "{#NvidiaDir}\alien.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#NvidiaDir}\alien-cli.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#NvidiaDir}\*.dll"; Excludes: "gtest*.dll,gmock*.dll,amdhip64*.dll"; DestDir: "{app}"; Flags: ignoreversion
#ifdef WithAmd
Source: "{#AmdDir}\alien.exe"; DestDir: "{app}"; DestName: "alien-amd.exe"; Flags: ignoreversion
Source: "{#AmdDir}\amdhip64*.dll"; DestDir: "{app}"; Flags: ignoreversion
#endif
Source: "{#RootDir}\resources\*"; Excludes: "\autosave"; DestDir: "{app}\resources"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#RootDir}\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

; ALIEN looks for its resources in the working directory.
[Icons]
Name: "{group}\ALIEN"; Filename: "{app}\{code:GuiExe}"; WorkingDir: "{app}"; Tasks: startmenuicon
Name: "{group}\{cm:UninstallProgram,ALIEN}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{autodesktop}\ALIEN"; Filename: "{app}\{code:GuiExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{code:GuiExe}"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,ALIEN}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\log.txt"
Type: files; Name: "{app}\profile.txt"
Type: files; Name: "{app}\trace.txt"

[Code]
#ifdef WithAmd
const
  NvidiaPciVendorId = '10DE';
  AmdPciVendorId = '1002';

var
  PreselectAmd: Boolean;

function CountGpus(const Wmi: Variant; const PciVendorId: String): Integer;
begin
  Result := Wmi.ExecQuery('SELECT Name FROM Win32_VideoController WHERE PNPDeviceID LIKE "%VEN_' + PciVendorId + '%"').Count;
end;

function HasAmdButNoNvidiaGpu: Boolean;
var
  Locator, Wmi: Variant;
begin
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Wmi := Locator.ConnectServer('.', 'root\CIMV2');
    Result := (CountGpus(Wmi, NvidiaPciVendorId) = 0) and (CountGpus(Wmi, AmdPciVendorId) > 0);
  except
    Result := False;
  end;
end;

function GpuTaskGivenOnCommandLine: Boolean;
var
  Tasks: String;
begin
  Tasks := Lowercase(ExpandConstant('{param:TASKS}') + ',' + ExpandConstant('{param:MERGETASKS}'));
  Result := (Pos('nvidia', Tasks) > 0) or (Pos('amd', Tasks) > 0);
end;

procedure InitializeWizard;
begin
  PreselectAmd := (WizardForm.PrevAppDir = '') and not GpuTaskGivenOnCommandLine and HasAmdButNoNvidiaGpu;
end;

// The task list is filled only after the folder page, also in silent installs, which pass through the pages as well.
procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpSelectTasks) and PreselectAmd then
  begin
    PreselectAmd := False;
    WizardSelectTasks('!nvidia,amd');
  end;
end;
#endif

function GuiExe(Param: String): String;
begin
  Result := 'alien.exe';
#ifdef WithAmd
  if WizardIsTaskSelected('amd') then
    Result := 'alien-amd.exe';
#endif
end;

function ContainsNonAsciiCharacters(const Text: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to Length(Text) do
    if Ord(Text[I]) > 127 then
      Result := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and not WizardSilent and ContainsNonAsciiCharacters(WizardDirValue) then
    Result := MsgBox('The folder contains non-English characters, which can prevent ALIEN from starting. Install it there anyway?'
      + #13#10#13#10 + 'Choose No to select another folder, for example C:\ALIEN.', mbConfirmation, MB_YESNO) = IDYES;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AutosaveDir: String;
begin
  AutosaveDir := ExpandConstant('{app}\resources\autosave');
  // RemoveDir fails if the folder is not empty.
  if (CurUninstallStep = usUninstall) and not UninstallSilent and DirExists(AutosaveDir) and not RemoveDir(AutosaveDir) then
    if MsgBox('Also delete the autosaves in ' + AutosaveDir + '?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      DelTree(AutosaveDir, True, True, True);
end;
