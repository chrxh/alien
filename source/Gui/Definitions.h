#pragma once

#include "Base/Definitions.h"

class _MainWindow;
using MainWindow = std::shared_ptr<_MainWindow>;

class AlienWindow;

class SimulationView;

class _Shader;
using Shader = std::shared_ptr<_Shader>;

class _SimulationScrollbar;
using SimulationScrollbar = std::shared_ptr<_SimulationScrollbar>;

class Viewport;

class StyleRepository;

class TemporalControlWindow;

class SpatialControlWindow;

class SimulationParametersWindow;

class StatisticsWindow;

class SimulationInteractionController;

class GpuSettingsDialog;

class _NewSimulationDialog;
using NewSimulationDialog = std::shared_ptr<_NewSimulationDialog>;

class StartupController;

class ExitDialog;

class AboutDialog;

class MassOperationsDialog;

class LogWindow;

class _GuiLogger;
using GuiLogger = std::shared_ptr<_GuiLogger>;

class UiController;

class AutosaveController;

class _GettingStartedWindow;
using GettingStartedWindow = std::shared_ptr<_GettingStartedWindow>;

class _DisplaySettingsDialog;
using DisplaySettingsDialog = std::shared_ptr<_DisplaySettingsDialog>;

class _EditorModel;
using EditorModel = std::shared_ptr<_EditorModel>;

class EditorController;

class _SelectionWindow;
using SelectionWindow = std::shared_ptr<_SelectionWindow>;

class _PatternEditorWindow;
using PatternEditorWindow = std::shared_ptr<_PatternEditorWindow>;

class WindowController;

class _ResizeWorldDialog;
using ResizeWorldDialog = std::shared_ptr<_ResizeWorldDialog>;

class _OpenSymbolsDialog;
using OpenSymbolsDialog = std::shared_ptr<_OpenSymbolsDialog>;

class _InspectorWindow;
using InspectorWindow = std::shared_ptr<_InspectorWindow>;

class _CreatorWindow;
using CreatorWindow = std::shared_ptr<_CreatorWindow>;

class _MultiplierWindow;
using MultiplierWindow = std::shared_ptr<_MultiplierWindow>;

class _PatternAnalysisDialog;
using PatternAnalysisDialog = std::shared_ptr<_PatternAnalysisDialog>;

class FpsController;

class BrowserWindow;

class _ShaderWindow;
using ShaderWindow = std::shared_ptr<_ShaderWindow>;

class LoginDialog;

class UploadSimulationDialog;

class EditSimulationDialog;

class CreateUserDialog;

class ActivateUserDialog;

class _DeleteUserDialog;
using DeleteUserDialog = std::shared_ptr<_DeleteUserDialog>;

class _NetworkSettingsDialog;
using NetworkSettingsDialog = std::shared_ptr<_NetworkSettingsDialog>;

class _ResetPasswordDialog;
using ResetPasswordDialog = std::shared_ptr<_ResetPasswordDialog>;

class _NewPasswordDialog;
using NewPasswordDialog = std::shared_ptr<_NewPasswordDialog>;

class _ImageToPatternDialog;
using ImageToPatternDialog = std::shared_ptr<_ImageToPatternDialog>;

class _GenomeEditorWindow;
using GenomeEditorWindow = std::shared_ptr<_GenomeEditorWindow>;

class RadiationSourcesWindow;

class _ChangeColorDialog;
using ChangeColorDialog = std::shared_ptr<_ChangeColorDialog>;

class _AutosaveWindow;
using AutosaveWindow = std::shared_ptr<_AutosaveWindow>;

class FileTransferController;

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

