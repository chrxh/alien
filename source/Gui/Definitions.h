#pragma once

#include <mutex>

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"

class QGraphicsItem;
class QGraphicsView;
class QGraphicsScene;
class QTabWidget;
class QTableWidgetItem;
class QSignalMapper;
class QAction;
class QBuffer;
class QLabel;
class QGraphicsSimpleTextItem;

class CellItem;
class ParticleItem;
class CellConnectionItem;
class ItemConfig;
class MonitorView;
class MetadataManager;
class UniverseView;
class OpenGLUniverseView;
class OpenGLUniverseScene;
class ItemUniverseView;
class ItemManager;
class DataRepository;
class GeneralInfoController;
class ViewportInterface;
class ViewportController;
class MarkerItem;
class DataEditController;
class DataEditContext;
class ToolbarController;
class ToolbarContext;
class ToolbarView;
class ActionModel;
class DataEditModel;
class DataEditView;
class DataRepository;
class ClusterEditTab;
class CellEditTab;
class MetadataEditTab;
class CellComputerEditTab;
class ParticleEditTab;
class SelectionEditTab;
class SymbolEditTab;
class HexEditWidget;
class TokenEditTabWidget;
class TokenEditTab;
class Notifier;
class MainView;
class MainModel;
class MainController;
class SnapshotController;
class SimulationViewWidget;
class ActionHolder;
class ActionController;
class MonitorController;
class PixelImageSectionItem;
class VectorImageSectionItem;
class VectorViewport;
class PixelViewport;
class ItemViewport;
class StartupController;
class SimulationViewController;

struct MonitorData;
using MonitorDataSP = boost::shared_ptr<MonitorData>;

enum class ActiveView {
    OpenGLScene,
    ItemScene
};
enum class Receiver { Simulation, VisualEditor, DataEditor, ActionController };
enum class UpdateDescription { All, AllExceptToken, AllExceptSymbols };
enum class NotifyScrollChanged { No, Yes };

class _SimulationConfig;
using SimulationConfig = boost::shared_ptr<_SimulationConfig>;

class WebSimulationSelectionView;
class WebSimulationSelectionController;
class WebSimulationTableModel;

class WebSimulationController;

enum class ModelComputationType
{
	Gpu = 1
};

class DataAnalyzer;
class Queue;
class GettingStartedWindow;

class LoggingView;
class LoggingController;
class GuiLogger;
class BugReportView;
class ZoomActionController;