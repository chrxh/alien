#include <QFileDialog>
#include <QMessageBox>

#include "Model/Api/SimulationController.h"

#include "Gui/Toolbar/ToolbarController.h"
#include "Gui/Toolbar/ToolbarContext.h"

#include "InfoController.h"
#include "DataEditController.h"
#include "DataEditContext.h"
#include "NewSimulationDialog.h"
#include "Settings.h"
#include "MainView.h"
#include "MainController.h"
#include "MainModel.h"
#include "SimulationParametersDialog.h"

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

void MainView::init(MainModel* model, MainController* mainController)
{
	_model = model;
	_mainController = mainController;
	_toolbar = new ToolbarController(ui->visualEditController);
	_dataEditor = new DataEditController(ui->visualEditController);
	_infoController = new InfoController(this);
	_infoController->init(ui->infoLabel, mainController);

	connectActions();
	setupTheme();
	setWindowState(windowState() | Qt::WindowFullScreen);
	show();
}

void MainView::refresh()
{
	ui->visualEditController->refresh();
}

void MainView::setupEditors(SimulationController * controller, DataController* manipulator, Notifier* notifier)
{
	_toolbar->init({ 10, 10 }, notifier, manipulator, controller->getContext());
	_dataEditor->init({ 10, 60 }, notifier, manipulator, controller->getContext());
	ui->visualEditController->init(notifier, controller, manipulator);

	ui->actionEditor->setChecked(false);
	_model->setEditMode(boost::none);
	onSetEditorMode();
}

InfoController * MainView::getInfoController() const
{
	return _infoController;
}

void MainView::connectActions()
{
	connect(ui->actionNewSimulation, &QAction::triggered, this, &MainView::onNewSimulation);
	connect(ui->actionSaveSimulation, &QAction::triggered, this, &MainView::onSaveSimulation);
	connect(ui->actionLoadSimulation, &QAction::triggered, this, &MainView::onLoadSimulation);
	connect(ui->actionExit, &QAction::triggered, this, &MainView::close);
	connect(ui->actionPlay, &QAction::triggered, this, &MainView::onRunClicked);
	connect(ui->actionZoomIn, &QAction::triggered, this, &MainView::onZoomInClicked);
	connect(ui->actionZoomOut, &QAction::triggered, this, &MainView::onZoomOutClicked);
	connect(ui->actionEditor, &QAction::triggered, this, &MainView::onSetEditorMode);
	connect(ui->actionEditSimulationParameters, &QAction::triggered, this, &MainView::onEditSimulationParameters);
	connect(ui->actionLoadSimulationParameters, &QAction::triggered, this, &MainView::onLoadSimulationParameters);

	ui->actionEditor->setEnabled(true);
	ui->actionZoomIn->setEnabled(true);
	ui->actionZoomOut->setEnabled(true);
}

void MainView::setupTheme()
{
	setFont(GuiSettings::getGlobalFont());
	ui->menuSimulation->setFont(GuiSettings::getGlobalFont());
	ui->menuView->setFont(GuiSettings::getGlobalFont());
	ui->menuEdit->setFont(GuiSettings::getGlobalFont());
	ui->menuSelection->setFont(GuiSettings::getGlobalFont());
	ui->menuSettings->setFont(GuiSettings::getGlobalFont());
	ui->menuHelp->setFont(GuiSettings::getGlobalFont());
	ui->menuSimulationParameters->setFont(GuiSettings::getGlobalFont());
	ui->menuSymbolTable->setFont(GuiSettings::getGlobalFont());
	ui->menuAddEnsemble->setFont(GuiSettings::getGlobalFont());
	ui->menuMultiplyExtension->setFont(GuiSettings::getGlobalFont());

	ui->tpsForcingButton->setStyleSheet(GuiSettings::ButtonStyleSheet);
	ui->toolBar->setStyleSheet("background-color: #303030");
	QPalette p = ui->tpsForcingButton->palette();
	p.setColor(QPalette::ButtonText, GuiSettings::ButtonTextColor);
	ui->tpsForcingButton->setPalette(p);
}

void MainView::onRunClicked(bool run)
{
	if (run) {
		ui->actionPlay->setIcon(QIcon("://Icons/pause.png"));
		ui->actionStep->setEnabled(false);
	}
	else {
		ui->actionPlay->setIcon(QIcon("://Icons/play.png"));
		ui->actionStep->setEnabled(true);
	}
	ui->actionSave_cell_extension->setEnabled(false);
	ui->actionCopy_cell_extension->setEnabled(false);
	ui->actionStepBack->setEnabled(false);
	ui->menuMultiplyExtension->setEnabled(false);
	ui->actionCopyCell->setEnabled(false);
	ui->actionDeleteCell->setEnabled(false);
	ui->actionDeleteExtension->setEnabled(false);

	_mainController->onRunSimulation(run);
}

void MainView::onZoomInClicked()
{
	ui->visualEditController->zoom(2.0);
	updateZoomFactor();
}

void MainView::onZoomOutClicked()
{
	ui->visualEditController->zoom(0.5);
	updateZoomFactor();
}

void MainView::onSetEditorMode()
{
	auto editMode = _model->isEditMode();
	bool newEditMode = editMode ? !editMode.get() : false;
	_model->setEditMode(newEditMode);

	_toolbar->getContext()->show(newEditMode);
	_dataEditor->getContext()->show(newEditMode);
	if (newEditMode) {
		ui->visualEditController->setActiveScene(ActiveScene::ItemScene);
		ui->actionEditor->setIcon(QIcon("://Icons/PixelView.png"));
	}
	else {
		ui->visualEditController->setActiveScene(ActiveScene::PixelScene);
		ui->actionEditor->setIcon(QIcon("://Icons/EditorView.png"));
		cellDefocused();
	}
}

void MainView::onNewSimulation()
{
	NewSimulationDialog dialog(_model->getSimulationParameters(), _model->getSymbolTable(), _mainController->getSerializer(), this);
	if (dialog.exec()) {
		NewSimulationConfig config{ 
			dialog.getMaxThreads(), dialog.getGridSize(), dialog.getUniverseSize(), dialog.getSymbolTable(), dialog.getSimulationParameters(), dialog.getEnergy()
		};
		_mainController->onNewSimulation(config);
		updateZoomFactor();
		ui->actionPlay->setChecked(false);
		onRunClicked(false);
	}
}

void MainView::onSaveSimulation()
{
	QString filename = QFileDialog::getSaveFileName(this, "Save Simulation", "", "Alien Simulation(*.sim)");
	if (!filename.isEmpty()) {
		_mainController->onSaveSimulation(filename.toStdString());
	}
}

void MainView::onLoadSimulation()
{
	QString filename = QFileDialog::getOpenFileName(this, "Load Simulation", "", "Alien Simulation (*.sim)");
	if (!filename.isEmpty()) {
		if(_mainController->onLoadSimulation(filename.toStdString())) {
			updateZoomFactor();
			ui->actionPlay->setChecked(false);
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
	SimulationParametersDialog dialog(_model->getSimulationParameters(), _mainController->getSerializer(), this);
	if (dialog.exec()) {
		_mainController->onUpdateSimulationParametersForRunningSimulation(dialog.getSimulationParameters());
	}
}

void MainView::onLoadSimulationParameters()
{
	QString filename = QFileDialog::getOpenFileName(this, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
	if (!filename.isEmpty()) {
		if (_mainController->onLoadSimulationParameters(filename.toStdString())) {
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified simulation parameter file could not loaded.");
			msgBox.exec();
		}
	}
}

void MainView::cellDefocused()
{
	ui->actionSave_cell_extension->setEnabled(false);
	ui->actionCopy_cell_extension->setEnabled(false);
	ui->menuMultiplyExtension->setEnabled(false);
	ui->actionCopyCell->setEnabled(false);
	ui->actionDeleteCell->setEnabled(false);
	ui->actionDeleteExtension->setEnabled(false);
}

void MainView::updateZoomFactor()
{
	_infoController->setZoomFactor(ui->visualEditController->getZoomFactor());
}

