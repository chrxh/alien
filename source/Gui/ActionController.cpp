#include <QFileDialog>
#include <QMessageBox>
#include <QAction>

#include "Model/Api/SimulationController.h"
#include "Model/Api/Serializer.h"
#include "Model/Api/SymbolTable.h"

#include "Gui/Toolbar/ToolbarController.h"
#include "Gui/Toolbar/ToolbarContext.h"
#include "Gui/VisualEditor/VisualEditController.h"

#include "ActionController.h"
#include "ActionHolder.h"
#include "SerializationHelper.h"
#include "InfoController.h"
#include "DataEditController.h"
#include "DataEditContext.h"
#include "NewSimulationDialog.h"
#include "Settings.h"
#include "MainController.h"
#include "MainModel.h"
#include "MainView.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"

ActionController::ActionController(QObject * parent)
	: QObject(parent)
{
	
}

void ActionController::init(MainController * mainController, MainModel* mainModel, MainView* mainView, VisualEditController* visualEditor
	, Serializer* serializer, InfoController* infoController, DataEditController* dataEditor, ToolbarController* toolbar)
{
	_mainController = mainController;
	_mainModel = mainModel;
	_mainView = mainView;
	_visualEditor = visualEditor;
	_serializer = serializer;
	_infoController = infoController;
	_dataEditor = dataEditor;
	_toolbar = toolbar;
	_actions = new ActionHolder(this);

	connect(_actions->actionNewSimulation, &QAction::triggered, this, &ActionController::onNewSimulation);
	connect(_actions->actionSaveSimulation, &QAction::triggered, this, &ActionController::onSaveSimulation);
	connect(_actions->actionLoadSimulation, &QAction::triggered, this, &ActionController::onLoadSimulation);
	connect(_actions->actionRunSimulation, &QAction::triggered, this, &ActionController::onRunClicked);
	connect(_actions->actionRunStepForward, &QAction::triggered, this, &ActionController::onStepForward);
	connect(_actions->actionRunStepBackward, &QAction::triggered, this, &ActionController::onStepBackward);
	connect(_actions->actionSnapshot, &QAction::triggered, this, &ActionController::onMakeSnapshot);
	connect(_actions->actionRestore, &QAction::triggered, this, &ActionController::onRestoreSnapshot);
	connect(_actions->actionExit, &QAction::triggered, _mainView, &MainView::close);
	connect(_actions->actionZoomIn, &QAction::triggered, this, &ActionController::onZoomInClicked);
	connect(_actions->actionZoomOut, &QAction::triggered, this, &ActionController::onZoomOutClicked);
	connect(_actions->actionEditor, &QAction::triggered, this, &ActionController::onSetEditorMode);
	connect(_actions->actionEditSimParameters, &QAction::triggered, this, &ActionController::onEditSimulationParameters);
	connect(_actions->actionLoadSimParameters, &QAction::triggered, this, &ActionController::onLoadSimulationParameters);
	connect(_actions->actionSaveSimParameters, &QAction::triggered, this, &ActionController::onSaveSimulationParameters);
	connect(_actions->actionEditSymbols, &QAction::triggered, this, &ActionController::onEditSymbolTable);
	connect(_actions->actionLoadSymbols, &QAction::triggered, this, &ActionController::onLoadSymbolTable);
	connect(_actions->actionSaveSymbols, &QAction::triggered, this, &ActionController::onSaveSymbolTable);
}

ActionHolder * ActionController::getActionHolder()
{
	return _actions;
}

void ActionController::onRunClicked(bool run)
{
	if (run) {
		_actions->actionRunSimulation->setIcon(QIcon("://Icons/pause.png"));
		_actions->actionRunStepForward->setEnabled(false);
	}
	else {
		_actions->actionRunSimulation->setIcon(QIcon("://Icons/play.png"));
		_actions->actionRunStepForward->setEnabled(true);
	}
	_actions->actionRunStepBackward->setEnabled(false);

	_mainController->onRunSimulation(run);
}

void ActionController::onStepForward()
{
	_mainController->onStepForward();
	_actions->actionRunStepBackward->setEnabled(true);
}

void ActionController::onStepBackward()
{
	bool emptyStack = false;
	_mainController->onStepBackward(emptyStack);
	if (emptyStack) {
		_actions->actionRunStepBackward->setEnabled(false);
	}
	_visualEditor->refresh();
}

void ActionController::onMakeSnapshot()
{
	_mainController->onMakeSnapshot();
	_actions->actionRestore->setEnabled(true);
}

void ActionController::onRestoreSnapshot()
{
	_mainController->onRestoreSnapshot();
	_visualEditor->refresh();
}

void ActionController::onZoomInClicked()
{
	_visualEditor->zoom(2.0);
	updateZoomFactor();
}

void ActionController::onZoomOutClicked()
{
	_visualEditor->zoom(0.5);
	updateZoomFactor();
}

void ActionController::onSetEditorMode()
{
	auto editMode = _mainModel->isEditMode();
	bool newEditMode = editMode ? !editMode.get() : false;
	_mainModel->setEditMode(newEditMode);

	_toolbar->getContext()->show(newEditMode);
	_dataEditor->getContext()->onShow(newEditMode);
	if (newEditMode) {
		_visualEditor->setActiveScene(ActiveScene::ItemScene);
		_actions->actionEditor->setIcon(QIcon("://Icons/PixelView.png"));
	}
	else {
		_visualEditor->setActiveScene(ActiveScene::PixelScene);
		_actions->actionEditor->setIcon(QIcon("://Icons/EditorView.png"));
	}
}

void ActionController::onNewSimulation()
{
	NewSimulationDialog dialog(_mainModel->getSimulationParameters(), _mainModel->getSymbolTable(), _serializer, _mainView);
	if (dialog.exec()) {
		NewSimulationConfig config{
			dialog.getMaxThreads(), dialog.getGridSize(), dialog.getUniverseSize(), dialog.getSymbolTable(), dialog.getSimulationParameters(), dialog.getEnergy()
		};
		_mainController->onNewSimulation(config);
		updateZoomFactor();
		_actions->actionRunSimulation->setChecked(false);
		_actions->actionRestore->setEnabled(false);
		_actions->actionRunStepBackward->setEnabled(false);
		onRunClicked(false);
	}
}

void ActionController::onSaveSimulation()
{
	QString filename = QFileDialog::getSaveFileName(_mainView, "Save Simulation", "", "Alien Simulation(*.sim)");
	if (!filename.isEmpty()) {
		_mainController->onSaveSimulation(filename.toStdString());
	}
}

void ActionController::onLoadSimulation()
{
	QString filename = QFileDialog::getOpenFileName(_mainView, "Load Simulation", "", "Alien Simulation (*.sim)");
	if (!filename.isEmpty()) {
		if (_mainController->onLoadSimulation(filename.toStdString())) {
			updateZoomFactor();
			_actions->actionRunSimulation->setChecked(false);
			_actions->actionRestore->setEnabled(false);
			_actions->actionRunStepBackward->setEnabled(false);
			onRunClicked(false);
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified simulation could not loaded.");
			msgBox.exec();
		}
	}
}

void ActionController::onEditSimulationParameters()
{
	SimulationParametersDialog dialog(_mainModel->getSimulationParameters()->clone(), _serializer, _mainView);
	if (dialog.exec()) {
		_mainModel->setSimulationParameters(dialog.getSimulationParameters());
		_mainController->onUpdateSimulationParametersForRunningSimulation();
	}
}

void ActionController::onLoadSimulationParameters()
{
	QString filename = QFileDialog::getOpenFileName(_mainView, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
	if (!filename.isEmpty()) {
		SimulationParameters* parameters;
		if (SerializationHelper::loadFromFile<SimulationParameters*>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeSimulationParameters(data); }, parameters)) {
			_mainModel->setSimulationParameters(parameters);
			_mainController->onUpdateSimulationParametersForRunningSimulation();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified simulation parameter file could not loaded.");
			msgBox.exec();
		}
	}
}

void ActionController::onSaveSimulationParameters()
{
	QString filename = QFileDialog::getSaveFileName(_mainView, "Save Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
	if (!filename.isEmpty()) {
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeSimulationParameters(_mainModel->getSimulationParameters()); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The simulation parameters could not saved.");
			msgBox.exec();
		}
	}

}

void ActionController::onEditSymbolTable()
{
	auto origSymbols = _mainModel->getSymbolTable();
	SymbolTableDialog dialog(origSymbols->clone(), _serializer, _mainView);
	if (dialog.exec()) {
		origSymbols->getSymbolsFrom(dialog.getSymbolTable());
		Q_EMIT _dataEditor->getContext()->onRefresh();
	}
}

void ActionController::onLoadSymbolTable()
{
	QString filename = QFileDialog::getOpenFileName(_mainView, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
	if (!filename.isEmpty()) {
		SymbolTable* symbolTable;
		if (SerializationHelper::loadFromFile<SymbolTable*>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeSymbolTable(data); }, symbolTable)) {
			_mainModel->getSymbolTable()->getSymbolsFrom(symbolTable);
			delete symbolTable;
			Q_EMIT _dataEditor->getContext()->onRefresh();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified symbol table could not loaded.");
			msgBox.exec();
		}
	}
}

void ActionController::onSaveSymbolTable()
{
	QString filename = QFileDialog::getSaveFileName(_mainView, "Save Symbol Table", "", "Alien Symbol Table (*.sym)");
	if (!filename.isEmpty()) {
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeSymbolTable(_mainModel->getSymbolTable()); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The symbol table could not saved.");
			msgBox.exec();
			return;
		}
	}
}

void ActionController::updateZoomFactor()
{
	_infoController->setZoomFactor(_visualEditor->getZoomFactor());
}
