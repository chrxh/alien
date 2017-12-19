#include <QFileDialog>
#include <QMessageBox>

#include "Model/Api/SimulationController.h"
#include "Model/Api/Serializer.h"
#include "Model/Api/SymbolTable.h"

#include "Gui/Toolbar/ToolbarController.h"
#include "Gui/Toolbar/ToolbarContext.h"

#include "ActionHolder.h"
#include "SerializationHelper.h"
#include "InfoController.h"
#include "DataEditController.h"
#include "DataEditContext.h"
#include "NewSimulationDialog.h"
#include "Settings.h"
#include "MainView.h"
#include "MainController.h"
#include "MainModel.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"

#include "ui_MainView.h"

MainView::MainView(QWidget * parent)
	: QMainWindow(parent)
	, ui(new Ui::MainView)
{
	ui->setupUi(this);
}

MainView::~MainView()
{
	delete ui;
}

void MainView::init(MainModel* model, MainController* mainController, Serializer* serializer)
{
	_model = model;
	_controller = mainController;
	_serializer = serializer;
	_visualEditor = ui->visualEditController;
	_toolbar = new ToolbarController(_visualEditor);
	_dataEditor = new DataEditController(_visualEditor);
	_infoController = new InfoController(this);
	_actions = new ActionHolder(this);
	_infoController->init(ui->infoLabel, mainController);

	connectWidget();
	setupMenu();
	setupTheme();
	setWindowState(windowState() | Qt::WindowFullScreen);
	show();
}

void MainView::refresh()
{
	_visualEditor->refresh();
}

void MainView::setupEditors(SimulationController * controller, DataController* manipulator, Notifier* notifier)
{
	_toolbar->init({ 10, 10 }, notifier, manipulator, controller->getContext());
	_dataEditor->init({ 10, 60 }, notifier, manipulator, controller->getContext());
	_visualEditor->init(notifier, controller, manipulator);

	_actions->actionEditor->setChecked(false);
	_model->setEditMode(boost::none);
	onSetEditorMode();
}

InfoController * MainView::getInfoController() const
{
	return _infoController;
}

void MainView::connectWidget()
{
	connect(_actions->actionNewSimulation, &QAction::triggered, this, &MainView::onNewSimulation);
	connect(_actions->actionSaveSimulation, &QAction::triggered, this, &MainView::onSaveSimulation);
	connect(_actions->actionLoadSimulation, &QAction::triggered, this, &MainView::onLoadSimulation);
	connect(_actions->actionRunSimulation, &QAction::triggered, this, &MainView::onRunClicked);
	connect(_actions->actionRunStepForward, &QAction::triggered, this, &MainView::onStepForward);
	connect(_actions->actionRunStepBackward, &QAction::triggered, this, &MainView::onStepBackward);
	connect(_actions->actionSnapshot, &QAction::triggered, this, &MainView::onMakeSnapshot);
	connect(_actions->actionRestore, &QAction::triggered, this, &MainView::onRestoreSnapshot);
	connect(_actions->actionExit, &QAction::triggered, this, &MainView::close);
	connect(_actions->actionZoomIn, &QAction::triggered, this, &MainView::onZoomInClicked);
	connect(_actions->actionZoomOut, &QAction::triggered, this, &MainView::onZoomOutClicked);
	connect(_actions->actionEditor, &QAction::triggered, this, &MainView::onSetEditorMode);
	connect(_actions->actionEditSimParameters, &QAction::triggered, this, &MainView::onEditSimulationParameters);
	connect(_actions->actionLoadSimParameters, &QAction::triggered, this, &MainView::onLoadSimulationParameters);
	connect(_actions->actionSaveSimParameters, &QAction::triggered, this, &MainView::onSaveSimulationParameters);
	connect(_actions->actionEditSymbols, &QAction::triggered, this, &MainView::onEditSymbolTable);
	connect(_actions->actionLoadSymbols, &QAction::triggered, this, &MainView::onLoadSymbolTable);
	connect(_actions->actionSaveSymbols, &QAction::triggered, this, &MainView::onSaveSymbolTable);
}

void MainView::setupMenu()
{
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(_actions->actionEditor);
	ui->toolBar->addAction(_actions->actionMonitor);
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(_actions->actionZoomIn);
	ui->toolBar->addAction(_actions->actionZoomOut);
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(_actions->actionSnapshot);
	ui->toolBar->addAction(_actions->actionRestore);
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(_actions->actionRunSimulation);
	ui->toolBar->addAction(_actions->actionRunStepBackward);
	ui->toolBar->addAction(_actions->actionRunStepForward);
	ui->toolBar->addSeparator();


	ui->menuSimulation->addAction(_actions->actionNewSimulation);
	ui->menuSimulation->addAction(_actions->actionLoadSimulation);
	ui->menuSimulation->addAction(_actions->actionSaveSimulation);
	ui->menuSimulation->addSeparator();
	ui->menuSimulation->addAction(_actions->actionRunSimulation);
	ui->menuSimulation->addAction(_actions->actionRunStepForward);
	ui->menuSimulation->addAction(_actions->actionRunStepBackward);
	ui->menuSimulation->addAction(_actions->actionSnapshot);
	ui->menuSimulation->addAction(_actions->actionRestore);
	ui->menuSimulation->addSeparator();
	ui->menuSimulation->addAction(_actions->actionExit);

	ui->menuSimulationParameters->addAction(_actions->actionEditSimParameters);
	ui->menuSimulationParameters->addAction(_actions->actionLoadSimParameters);
	ui->menuSimulationParameters->addAction(_actions->actionSaveSimParameters);
	ui->menuSymbolTable->addAction(_actions->actionEditSymbols);
	ui->menuSymbolTable->addAction(_actions->actionLoadSymbols);
	ui->menuSymbolTable->addAction(_actions->actionSaveSymbols);
	ui->menuSymbolTable->addAction(_actions->actionMergeWithSymbols);

	ui->menuView->addAction(_actions->actionEditor);
	ui->menuView->addAction(_actions->actionMonitor);
	ui->menuView->addSeparator();
	ui->menuView->addAction(_actions->actionZoomIn);
	ui->menuView->addAction(_actions->actionZoomOut);
	ui->menuView->addAction(_actions->actionFullscreen);

	ui->menuEntity->addAction(_actions->actionNewCell);
	ui->menuEntity->addAction(_actions->actionNewParticle);
	ui->menuEntity->addSeparator();
	ui->menuEntity->addAction(_actions->actionCopyEntity);
	ui->menuEntity->addAction(_actions->actionPasteEntity);
	ui->menuEntity->addAction(_actions->actionDeleteEntity);
	ui->menuEntity->addSeparator();
	ui->menuEntity->addAction(_actions->actionNewToken);
	ui->menuEntity->addAction(_actions->actionCopyToken);
	ui->menuEntity->addAction(_actions->actionPasteToken);
	ui->menuEntity->addAction(_actions->actionDeleteToken);

	ui->menuNewEnsemble->addAction(_actions->actionNewRectangle);
	ui->menuNewEnsemble->addAction(_actions->actionNewHexagon);
	ui->menuNewEnsemble->addAction(_actions->actionNewParticles);
	ui->menuCollection->addAction(_actions->actionLoadCol);
	ui->menuCollection->addAction(_actions->actionSaveCol);
	ui->menuCollection->addAction(_actions->actionCopyCol);
	ui->menuCollection->addAction(_actions->actionPasteCol);
	ui->menuCollection->addAction(_actions->actionDeleteCol);
	ui->menuMultiplyCollection->addAction(_actions->actionMultiplyRandom);
	ui->menuMultiplyCollection->addAction(_actions->actionMultiplyArrangement);

	ui->menuHelp->addAction(_actions->actionAbout);
	ui->menuEntity->addSeparator();
	ui->menuHelp->addAction(_actions->actionDocumentation);
}

void MainView::setupTheme()
{
	setFont(GuiSettings::getGlobalFont());
	ui->menuSimulation->setFont(GuiSettings::getGlobalFont());
	ui->menuView->setFont(GuiSettings::getGlobalFont());
	ui->menuEntity->setFont(GuiSettings::getGlobalFont());
	ui->menuCollection->setFont(GuiSettings::getGlobalFont());
	ui->menuSettings->setFont(GuiSettings::getGlobalFont());
	ui->menuHelp->setFont(GuiSettings::getGlobalFont());
	ui->menuSimulationParameters->setFont(GuiSettings::getGlobalFont());
	ui->menuSymbolTable->setFont(GuiSettings::getGlobalFont());
	ui->menuNewEnsemble->setFont(GuiSettings::getGlobalFont());
	ui->menuMultiplyCollection->setFont(GuiSettings::getGlobalFont());

	ui->tpsForcingButton->setStyleSheet(GuiSettings::ButtonStyleSheet);
	ui->toolBar->setStyleSheet("background-color: #303030");
	QPalette p = ui->tpsForcingButton->palette();
	p.setColor(QPalette::ButtonText, GuiSettings::ButtonTextColor);
	ui->tpsForcingButton->setPalette(p);
}

void MainView::onRunClicked(bool run)
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

	_controller->onRunSimulation(run);
}

void MainView::onStepForward()
{
	_controller->onStepForward();
	_actions->actionRunStepBackward->setEnabled(true);
}

void MainView::onStepBackward()
{
	bool emptyStack = false;
	_controller->onStepBackward(emptyStack);
	if (emptyStack) {
		_actions->actionRunStepBackward->setEnabled(false);
	}
	refresh();
}

void MainView::onMakeSnapshot()
{
	_controller->onMakeSnapshot();
	_actions->actionRestore->setEnabled(true);
}

void MainView::onRestoreSnapshot()
{
	_controller->onRestoreSnapshot();
	refresh();
}

void MainView::onZoomInClicked()
{
	_visualEditor->zoom(2.0);
	updateZoomFactor();
}

void MainView::onZoomOutClicked()
{
	_visualEditor->zoom(0.5);
	updateZoomFactor();
}

void MainView::onSetEditorMode()
{
	auto editMode = _model->isEditMode();
	bool newEditMode = editMode ? !editMode.get() : false;
	_model->setEditMode(newEditMode);

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

void MainView::onNewSimulation()
{
	NewSimulationDialog dialog(_model->getSimulationParameters(), _model->getSymbolTable(), _serializer, this);
	if (dialog.exec()) {
		NewSimulationConfig config{ 
			dialog.getMaxThreads(), dialog.getGridSize(), dialog.getUniverseSize(), dialog.getSymbolTable(), dialog.getSimulationParameters(), dialog.getEnergy()
		};
		_controller->onNewSimulation(config);
		updateZoomFactor();
		_actions->actionRunSimulation->setChecked(false);
		_actions->actionRestore->setEnabled(false);
		_actions->actionRunStepBackward->setEnabled(false);
		onRunClicked(false);
	}
}

void MainView::onSaveSimulation()
{
	QString filename = QFileDialog::getSaveFileName(this, "Save Simulation", "", "Alien Simulation(*.sim)");
	if (!filename.isEmpty()) {
		_controller->onSaveSimulation(filename.toStdString());
	}
}

void MainView::onLoadSimulation()
{
	QString filename = QFileDialog::getOpenFileName(this, "Load Simulation", "", "Alien Simulation (*.sim)");
	if (!filename.isEmpty()) {
		if(_controller->onLoadSimulation(filename.toStdString())) {
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

void MainView::onEditSimulationParameters()
{
	SimulationParametersDialog dialog(_model->getSimulationParameters()->clone(), _serializer, this);
	if (dialog.exec()) {
		_model->setSimulationParameters(dialog.getSimulationParameters());
		_controller->onUpdateSimulationParametersForRunningSimulation();
	}
}

void MainView::onLoadSimulationParameters()
{
	QString filename = QFileDialog::getOpenFileName(this, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
	if (!filename.isEmpty()) {
		SimulationParameters* parameters;
		if (SerializationHelper::loadFromFile<SimulationParameters*>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeSimulationParameters(data); }, parameters)) {
			_model->setSimulationParameters(parameters);
			_controller->onUpdateSimulationParametersForRunningSimulation();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified simulation parameter file could not loaded.");
			msgBox.exec();
		}
	}
}

void MainView::onSaveSimulationParameters()
{
	QString filename = QFileDialog::getSaveFileName(this, "Save Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
	if (!filename.isEmpty()) {
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeSimulationParameters(_model->getSimulationParameters()); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The simulation parameters could not saved.");
			msgBox.exec();
		}
	}

}

void MainView::onEditSymbolTable()
{
	auto origSymbols = _model->getSymbolTable();
	SymbolTableDialog dialog(origSymbols->clone(), _serializer, this);
	if (dialog.exec()) {
		origSymbols->getSymbolsFrom(dialog.getSymbolTable());
		Q_EMIT _dataEditor->getContext()->onRefresh();
	}
}

void MainView::onLoadSymbolTable()
{
	QString filename = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
	if (!filename.isEmpty()) {
		SymbolTable* symbolTable;
		if (SerializationHelper::loadFromFile<SymbolTable*>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeSymbolTable(data); }, symbolTable)) {
			_model->getSymbolTable()->getSymbolsFrom(symbolTable);
			delete symbolTable;
			Q_EMIT _dataEditor->getContext()->onRefresh();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified symbol table could not loaded.");
			msgBox.exec();
		}
	}
}

void MainView::onSaveSymbolTable()
{
	QString filename = QFileDialog::getSaveFileName(this, "Save Symbol Table", "", "Alien Symbol Table (*.sym)");
	if (!filename.isEmpty()) {
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeSymbolTable(_model->getSymbolTable()); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The symbol table could not saved.");
			msgBox.exec();
			return;
		}
	}
}

void MainView::updateZoomFactor()
{
	_infoController->setZoomFactor(_visualEditor->getZoomFactor());
}

