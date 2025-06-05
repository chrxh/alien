#pragma once

#include "Base/Definitions.h"

class _MainWindow;
using MainWindow = std::shared_ptr<_MainWindow>;

//class AlienWindow;

class SimulationView;

class _Shader;
using Shader = std::shared_ptr<_Shader>;

class _SimulationScrollbar;
using SimulationScrollbar = std::shared_ptr<_SimulationScrollbar>;

class Viewport;

class StyleRepository;

class TemporalControlWindow;

class SpatialControlWindow;

class SimulationParametersMainWindow;

class StatisticsWindow;

class SimulationInteractionController;

class GpuSettingsDialog;

class NewSimulationDialog;

class MainLoopController;

class ExitDialog;

class AboutDialog;

class MassOperationsDialog;

class LogWindow;

class _GuiLogger;
using GuiLogger = std::shared_ptr<_GuiLogger>;

class UiController;

class GettingStartedWindow;

class DisplaySettingsDialog;

class EditorModel;

class EditorController;

class SelectionWindow;

class PatternEditorWindow;

class WindowController;

class ResizeWorldDialog;

class _InspectorWindow;
using InspectorWindow = std::shared_ptr<_InspectorWindow>;

class CreatorWindow;

class MultiplierWindow;

class PatternAnalysisDialog;

class FpsController;

class BrowserWindow;

class ShaderWindow;

class LoginDialog;

class UploadSimulationDialog;

class EditSimulationDialog;

class CreateUserDialog;

class ActivateUserDialog;

class DeleteUserDialog;

class NetworkSettingsDialog;

class ResetPasswordDialog;

class NewPasswordDialog;

class ImageToPatternDialog;

class GenomeEditorWindow;

class RadiationSourcesWindow;

class ChangeColorDialog;

class AutosaveWindow;

class CreatureEditorWindow;

class FileTransferController;

class _LocationWidgets;
using LocationWidgets = std::shared_ptr<_LocationWidgets>;

class _CreatureTabWidget;
using CreatureTabWidget = std::shared_ptr<_CreatureTabWidget>;

class _CreatureTabLayoutData;
using CreatureTabLayoutData = std::shared_ptr<_CreatureTabLayoutData>;

struct UserInfo;

struct GLFWvidmode;
struct GLFWwindow;
struct ImFont;
struct ImVec2;

struct TextureData
{
    unsigned int textureId;
    int width;
    int height;
};

