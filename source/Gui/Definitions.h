#pragma once

#include "Base/Definitions.h"
#include "Model/Api/Definitions.h"

class QGraphicsItem;
class QGraphicsView;
class QGraphicsScene;
class QTabWidget;
class QTableWidgetItem;
class QSignalMapper;
class QAction;

class CellItem;
class ParticleItem;
class CellConnectionItem;
class ItemConfig;
class MonitorView;
class DocumentationWindow;
class StartScreenController;
class MetadataManager;
class PixelUniverseView;
class ItemUniverseView;
class ItemManager;
class DataRepository;
class InfoController;
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
class VersionController;
class VisualEditController;
class ActionHolder;
class ActionController;
class StartScreenWidget;
class StartScreenController;
class Manipulator;
class MonitorController;

struct _MonitorModel;
using MonitorModel = shared_ptr<_MonitorModel>;

enum class ActiveScene { PixelScene, ItemScene };
enum class Receiver { Simulation, VisualEditor, DataEditor, ActionController };
enum class UpdateDescription { All, AllExceptToken, AllExceptSymbols };
enum class NotifyScrollChanged { No, Yes };


struct SimulationConfig
{
	uint maxThreads;
	IntVector2D gridSize;
	IntVector2D universeSize;
};

struct NewSimulationConfig
{
	uint maxThreads;
	IntVector2D gridSize;
	IntVector2D universeSize;
	SymbolTable* symbolTable;
	SimulationParameters* parameters;

	double energy;
};

