#pragma once

#include "Base/Definitions.h"

class _MainWindow;
using MainWindow = std::shared_ptr<_MainWindow>;

class _AlienWindow;
using AlienWindow = std::shared_ptr<_AlienWindow>;

class _SimulationView;
using SimulationView= std::shared_ptr<_SimulationView>;

class _Shader;
using Shader = std::shared_ptr<_Shader>;

class _SimulationScrollbar;
using SimulationScrollbar = std::shared_ptr<_SimulationScrollbar>;

class Viewport;

class StyleRepository;

class _TemporalControlWindow;
using TemporalControlWindow = std::shared_ptr<_TemporalControlWindow>;

class _SpatialControlWindow;
using SpatialControlWindow = std::shared_ptr<_SpatialControlWindow>;

class _SimulationParametersWindow;
using SimulationParametersWindow = std::shared_ptr<_SimulationParametersWindow>;

class _StatisticsWindow;
using StatisticsWindow = std::shared_ptr<_StatisticsWindow>;

class _ModeController;
using ModeController = std::shared_ptr<_ModeController>;

class _GpuSettingsDialog;
using GpuSettingsDialog = std::shared_ptr<_GpuSettingsDialog>;

class _NewSimulationDialog;
using NewSimulationDialog = std::shared_ptr<_NewSimulationDialog>;

class _StartupController;
using StartupController = std::shared_ptr<_StartupController>;

class _ExitDialog;
using ExitDialog = std::shared_ptr<_ExitDialog>;

class _AboutDialog;
using AboutDialog = std::shared_ptr<_AboutDialog>;

class _MassOperationsDialog;
using MassOperationsDialog = std::shared_ptr<_MassOperationsDialog>;

class _LogWindow;
using LogWindow = std::shared_ptr<_LogWindow>;

class _GuiLogger;
using GuiLogger = std::shared_ptr<_GuiLogger>;

class _UiController;
using UiController = std::shared_ptr<_UiController>;

class _AutosaveController;
using AutosaveController = std::shared_ptr<_AutosaveController>;

class _GettingStartedWindow;
using GettingStartedWindow = std::shared_ptr<_GettingStartedWindow>;

class _DisplaySettingsDialog;
using DisplaySettingsDialog = std::shared_ptr<_DisplaySettingsDialog>;

class _EditorModel;
using EditorModel = std::shared_ptr<_EditorModel>;

class _EditorController;
using EditorController = std::shared_ptr<_EditorController>;
using EditorControllerWeakPtr = _EditorController*;

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

class _FpsController;
using FpsController = std::shared_ptr<_FpsController>;

class _BrowserWindow;
using BrowserWindow = std::shared_ptr<_BrowserWindow>;

class _ShaderWindow;
using ShaderWindow = std::shared_ptr<_ShaderWindow>;

class _LoginDialog;
using LoginDialog = std::shared_ptr<_LoginDialog>;
using LoginDialogWeakPtr = std::weak_ptr<_LoginDialog>;

class _UploadSimulationDialog;
using UploadSimulationDialog = std::shared_ptr<_UploadSimulationDialog>;
using UploadSimulationDialogWeakPtr = std::weak_ptr<_UploadSimulationDialog>;

class _EditSimulationDialog;
using EditSimulationDialog = std::shared_ptr<_EditSimulationDialog>;
using EditSimulationDialogWeakPtr = std::weak_ptr<_EditSimulationDialog>;

class _CreateUserDialog;
using CreateUserDialog = std::shared_ptr<_CreateUserDialog>;
using CreateUserDialogWeakPtr = std::weak_ptr<_CreateUserDialog>;

class _ActivateUserDialog;
using ActivateUserDialog = std::shared_ptr<_ActivateUserDialog>;

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
using GenomeEditorWindowWeakPtr = std::weak_ptr<_GenomeEditorWindow>;

class _RadiationSourcesWindow;
using RadiationSourcesWindow = std::shared_ptr<_RadiationSourcesWindow>;

class _ChangeColorDialog;
using ChangeColorDialog = std::shared_ptr<_ChangeColorDialog>;

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