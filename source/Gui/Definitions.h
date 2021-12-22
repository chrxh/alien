#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/optional.hpp>

class _MainWindow;
using MainWindow = boost::shared_ptr<_MainWindow>;

class _SimulationView;
using SimulationView= boost::shared_ptr<_SimulationView>;

class _Shader;
using Shader = boost::shared_ptr<_Shader>;

class _SimulationScrollbar;
using SimulationScrollbar = boost::shared_ptr<_SimulationScrollbar>;

class _Viewport;
using Viewport = boost::shared_ptr<_Viewport>;

class StyleRepository;

class _TemporalControlWindow;
using TemporalControlWindow = boost::shared_ptr<_TemporalControlWindow>;

class _SpatialControlWindow;
using SpatialControlWindow = boost::shared_ptr<_SpatialControlWindow>;

class _SimulationParametersWindow;
using SimulationParametersWindow = boost::shared_ptr<_SimulationParametersWindow>;

class _StatisticsWindow;
using StatisticsWindow = boost::shared_ptr<_StatisticsWindow>;

class _ModeWindow;
using ModeWindow = boost::shared_ptr<_ModeWindow>;

class _GpuSettingsDialog;
using GpuSettingsDialog = boost::shared_ptr<_GpuSettingsDialog>;

class _NewSimulationDialog;
using NewSimulationDialog = boost::shared_ptr<_NewSimulationDialog>;

class _StartupWindow;
using StartupWindow = boost::shared_ptr<_StartupWindow>;

class _FlowGeneratorWindow;
using FlowGeneratorWindow = boost::shared_ptr<_FlowGeneratorWindow>;

class _AboutDialog;
using AboutDialog = boost::shared_ptr<_AboutDialog>;

class _ColorizeDialog;
using ColorizeDialog = boost::shared_ptr<_ColorizeDialog>;

class _LogWindow;
using LogWindow = boost::shared_ptr<_LogWindow>;

class _SimpleLogger;
using SimpleLogger = boost::shared_ptr<_SimpleLogger>;

class _FileLogger;
using FileLogger = boost::shared_ptr<_FileLogger>;

class _UiController;
using UiController = boost::shared_ptr<_UiController>;

class _AutosaveController;
using AutosaveController = boost::shared_ptr<_AutosaveController>;

class _GettingStartedWindow;
using GettingStartedWindow = boost::shared_ptr<_GettingStartedWindow>;

class _OpenSimulationDialog;
using OpenSimulationDialog = boost::shared_ptr<_OpenSimulationDialog>;

class _SaveSimulationDialog;
using SaveSimulationDialog = boost::shared_ptr<_SaveSimulationDialog>;

class _DisplaySettingsDialog;
using DisplaySettingsDialog = boost::shared_ptr<_DisplaySettingsDialog>;

class _EditorModel;
using EditorModel = boost::shared_ptr<_EditorModel>;

class _EditorController;
using EditorController = boost::shared_ptr<_EditorController>;

class _SelectionWindow;
using SelectionWindow = boost::shared_ptr<_SelectionWindow>;

class _ManipulatorWindow;
using ManipulatorWindow = boost::shared_ptr<_ManipulatorWindow>;

class _WindowController;
using WindowController = boost::shared_ptr<_WindowController>;

class _SaveSelectionDialog;
using SaveSelectionDialog = boost::shared_ptr<_SaveSelectionDialog>;

class _OpenSelectionDialog;
using OpenSelectionDialog = boost::shared_ptr<_OpenSelectionDialog>;

struct GLFWvidmode;
struct GLFWwindow;
struct ImFont;

struct TextureData
{
    unsigned int textureId;
    int width;
    int height;
};
