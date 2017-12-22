#include "Model/Api/SimulationController.h"
#include "Model/Api/Serializer.h"
#include "Model/Api/SymbolTable.h"

#include "Gui/Toolbar/ToolbarController.h"
#include "Gui/Toolbar/ToolbarContext.h"

#include "Gui/Actions/ActionController.h"
#include "Gui/Actions/ActionHolder.h"
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

void MainView::init(MainModel* model, MainController* mainController, Serializer* serializer, DataRepository* repository, Notifier* notifier)
{
	_model = model;
	_controller = mainController;
	_repository = repository;
	_notifier = notifier;
	_visualEditor = ui->visualEditController;
	_toolbar = new ToolbarController(_visualEditor);
	_dataEditor = new DataEditController(_visualEditor);
	_infoController = new InfoController(this);
	_actions = new ActionController(this);
	_infoController->init(ui->infoLabel, mainController);
	_actions->init(_controller, _model, this, _visualEditor, serializer, _infoController, _dataEditor, _toolbar, repository, notifier);

	setupMenu();
	setupTheme();
	setWindowState(windowState() | Qt::WindowFullScreen);
	show();
}

void MainView::refresh()
{
	_visualEditor->refresh();
}

void MainView::setupEditors(SimulationController * controller)
{
	_toolbar->init({ 10, 10 }, _notifier, _repository, controller->getContext(), _actions->getActionHolder());
	_dataEditor->init({ 10, 60 }, _notifier, _repository, controller->getContext());
	_visualEditor->init(_notifier, controller, _repository);

	_visualEditor->setActiveScene(ActiveScene::PixelScene);
	_actions->getActionHolder()->actionEditor->setChecked(false);
}

InfoController * MainView::getInfoController() const
{
	return _infoController;
}

void MainView::setupMenu()
{
	auto actions = _actions->getActionHolder();
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(actions->actionEditor);
	ui->toolBar->addAction(actions->actionMonitor);
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(actions->actionZoomIn);
	ui->toolBar->addAction(actions->actionZoomOut);
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(actions->actionSnapshot);
	ui->toolBar->addAction(actions->actionRestore);
	ui->toolBar->addSeparator();
	ui->toolBar->addAction(actions->actionRunSimulation);
	ui->toolBar->addAction(actions->actionRunStepBackward);
	ui->toolBar->addAction(actions->actionRunStepForward);
	ui->toolBar->addSeparator();

	ui->menuSimulation->addAction(actions->actionNewSimulation);
	ui->menuSimulation->addAction(actions->actionLoadSimulation);
	ui->menuSimulation->addAction(actions->actionSaveSimulation);
	ui->menuSimulation->addAction(actions->actionConfig);
	ui->menuSimulation->addSeparator();
	ui->menuSimulation->addAction(actions->actionRunSimulation);
	ui->menuSimulation->addAction(actions->actionRunStepForward);
	ui->menuSimulation->addAction(actions->actionRunStepBackward);
	ui->menuSimulation->addAction(actions->actionSnapshot);
	ui->menuSimulation->addAction(actions->actionRestore);
	ui->menuSimulation->addSeparator();
	ui->menuSimulation->addAction(actions->actionExit);

	ui->menuSimulationParameters->addAction(actions->actionEditSimParameters);
	ui->menuSimulationParameters->addAction(actions->actionLoadSimParameters);
	ui->menuSimulationParameters->addAction(actions->actionSaveSimParameters);
	ui->menuSymbolTable->addAction(actions->actionEditSymbols);
	ui->menuSymbolTable->addAction(actions->actionLoadSymbols);
	ui->menuSymbolTable->addAction(actions->actionSaveSymbols);
	ui->menuSymbolTable->addAction(actions->actionMergeWithSymbols);

	ui->menuView->addAction(actions->actionEditor);
	ui->menuView->addAction(actions->actionMonitor);
	ui->menuView->addSeparator();
	ui->menuView->addAction(actions->actionZoomIn);
	ui->menuView->addAction(actions->actionZoomOut);
	ui->menuView->addAction(actions->actionFullscreen);
	ui->menuView->addSeparator();
	ui->menuView->addAction(actions->actionShowCellInfo);

	ui->menuEntity->addAction(actions->actionNewCell);
	ui->menuEntity->addAction(actions->actionNewParticle);
	ui->menuEntity->addSeparator();
	ui->menuEntity->addAction(actions->actionCopyEntity);
	ui->menuEntity->addAction(actions->actionPasteEntity);
	ui->menuEntity->addAction(actions->actionDeleteEntity);
	ui->menuEntity->addSeparator();
	ui->menuEntity->addAction(actions->actionNewToken);
	ui->menuEntity->addAction(actions->actionCopyToken);
	ui->menuEntity->addAction(actions->actionPasteToken);
	ui->menuEntity->addAction(actions->actionDeleteToken);

	ui->menuCollection->addAction(actions->actionNewRectangle);
	ui->menuCollection->addAction(actions->actionNewHexagon);
	ui->menuCollection->addAction(actions->actionNewParticles);
	ui->menuCollection->addSeparator();
	ui->menuCollection->addAction(actions->actionLoadCol);
	ui->menuCollection->addAction(actions->actionSaveCol);
	ui->menuCollection->addAction(actions->actionCopyCol);
	ui->menuCollection->addAction(actions->actionPasteCol);
	ui->menuCollection->addAction(actions->actionDeleteSel);
	ui->menuCollection->addAction(actions->actionDeleteCol);
	ui->menuCollection->addSeparator();
	ui->menuCollection->addAction(actions->actionMultiplyRandom);
	ui->menuCollection->addAction(actions->actionMultiplyArrangement);

	ui->menuHelp->addAction(actions->actionAbout);
	ui->menuEntity->addSeparator();
	ui->menuHelp->addAction(actions->actionDocumentation);
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

	ui->tpsForcingButton->setStyleSheet(GuiSettings::ButtonStyleSheet);
	ui->toolBar->setStyleSheet("background-color: #303030");
	QPalette p = ui->tpsForcingButton->palette();
	p.setColor(QPalette::ButtonText, GuiSettings::ButtonTextColor);
	ui->tpsForcingButton->setPalette(p);
}


