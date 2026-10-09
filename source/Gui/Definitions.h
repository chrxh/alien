#pragma once

#include <imgui.h>

#include <Base/Definitions.h>

#include <Rendering/Definitions.h>

class _MainWindow;
using MainWindow = std::shared_ptr<_MainWindow>;

class SimulationView;

class _SimulationScrollbars;
using SimulationScrollbars = std::shared_ptr<_SimulationScrollbars>;

class Viewport;

class StyleService;

class TemporalControlWindow;

class SpatialControlWindow;

class SimulationParametersMainWindow;

class EvolutionDashboardWindow;

class SimulationInteractionController;


class NewSimulationDialog;

class MainLoopController;

class ExitDialog;

class AboutDialog;

class MassOperationsDialog;

class LogWindow;

class _GuiLogger;
using GuiLogger = std::shared_ptr<_GuiLogger>;

class UiController;

class DocumentationWindow;

class DisplaySettingsDialog;

class EditorModel;

class EditorController;

class WindowController;

class ResizeWorldDialog;

class SavePictureDialog;

class _InspectionWindow;
using InspectionWindow = std::shared_ptr<_InspectionWindow>;

class FpsController;

class ConsoleModeController;

class BrowserWindow;

class LoginDialog;

class UploadSimulationDialog;

class ReplaceSimulationDialog;

class EditSimulationDialog;

class CreateUserDialog;

class ActivateUserDialog;

class DeleteUserDialog;

class NetworkSettingsDialog;

class ResetPasswordDialog;

class NewPasswordDialog;

class ImageToPatternDialog;

class RadiationSourcesWindow;

class ChangeColorDialog;

class AutosaveController;

class AutosaveWindow;

class GenomeEditorWindow;

class PreviewSettingsDialog;

class MutationRatesDialog;

template <typename T>
class ColorMatrixDialog;

class FileTransferController;

class _LocationWidget;
using LocationWidget = std::shared_ptr<_LocationWidget>;

class _GenomeTabWidget;
using GenomeTabWidget = std::shared_ptr<_GenomeTabWidget>;

struct _GenomeTabLayoutData;
using GenomeTabLayoutData = std::shared_ptr<_GenomeTabLayoutData>;

struct _GenomeWindowEditData;
using GenomeWindowEditData = std::shared_ptr<_GenomeWindowEditData>;

struct _GenomeTabEditData;
using GenomeTabEditData = std::shared_ptr<_GenomeTabEditData>;

class _GenomeEditorWidget;
using GenomeEditorWidget = std::shared_ptr<_GenomeEditorWidget>;

class _GeneEditorWidget;
using GeneEditorWidget = std::shared_ptr<_GeneEditorWidget>;

class _NodeEditorWidget;
using NodeEditorWidget = std::shared_ptr<_NodeEditorWidget>;

class _NeuralNetEditorWidget;
using NeuralNetEditorWidget = std::shared_ptr<_NeuralNetEditorWidget>;

class _PreviewWidget;
using PreviewWidget = std::shared_ptr<_PreviewWidget>;

class _CreaturePreviewWidget;
using CreaturePreviewWidget = std::shared_ptr<_CreaturePreviewWidget>;

class _PreviewDescView;
using PreviewDescView = std::shared_ptr<_PreviewDescView>;

class _BrowserData;
using BrowserData = std::shared_ptr<_BrowserData>;

class _BrowserGalleryWidget;
using BrowserGalleryWidget = std::shared_ptr<_BrowserGalleryWidget>;

class _BrowserTableWidget;
using BrowserTableWidget = std::shared_ptr<_BrowserTableWidget>;

class _BrowserUserListWidget;
using BrowserUserListWidget = std::shared_ptr<_BrowserUserListWidget>;

class _BrowserLoginHintWidget;
using BrowserLoginHintWidget = std::shared_ptr<_BrowserLoginHintWidget>;

class _BrowserLoginBannerWidget;
using BrowserLoginBannerWidget = std::shared_ptr<_BrowserLoginBannerWidget>;

struct UserInfo;

struct GLFWvidmode;
struct GLFWwindow;
