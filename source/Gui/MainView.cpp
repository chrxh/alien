#include <QDesktopServices>
#include <QUrl>

#include "ModelBasic/SimulationController.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SerializationHelper.h"

#include "Gui/ToolbarController.h"
#include "Gui/ToolbarContext.h"
#include "Gui/ActionController.h"
#include "Gui/ActionHolder.h"
#include "Gui/MonitorController.h"

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
	_visualEditor = ui->visualEditController;
	_toolbar = new ToolbarController(_visualEditor);
	_dataEditor = new DataEditController(_visualEditor);
	_infoController = new InfoController(this);
	_actions = new ActionController(this);
	_monitor = new MonitorController(this);
	connect(_monitor, &MonitorController::closed, this, &MainView::monitorClosed);
}

MainView::~MainView()
{
	delete ui;
}

void MainView::init(MainModel* model, MainController* mainController, Serializer* serializer, DataRepository* repository
	, SimulationMonitor* simMonitor, Notifier* notifier)
{
	_model = model;
	_controller = mainController;
	_repository = repository;
	_notifier = notifier;

	_infoController->init(ui->infoLabel, mainController);
	_monitor->init(mainController);
	_actions->init(_controller, _model, this, _visualEditor, serializer, _infoController, _dataEditor, _toolbar
		, _monitor, repository, notifier);

	setupMenu();
	setupFontsAndColors();
	setupWidgets();
	setupFullScreen();
	show();

	_initialied = true;
}

void MainView::refresh()
{
	_visualEditor->refresh();
}

void MainView::setupEditors(SimulationController * controller, SimulationAccess* access)
{
	_toolbar->init({ 10, 10 }, _notifier, _repository, controller->getContext(), _actions->getActionHolder());
	_dataEditor->init({ 10, 60 }, _notifier, _repository, controller->getContext());
	_visualEditor->init(_notifier, controller, access, _repository);

	_visualEditor->setActiveScene(ActiveScene::PixelScene);
	_actions->getActionHolder()->actionEditor->setChecked(false);
}

InfoController * MainView::getInfoController() const
{
	return _infoController;
}

void MainView::showDocumentation(bool show)
{
    QDesktopServices::openUrl(QUrl("https://alien-project.org/documentation.html"));
}

void MainView::resizeEvent(QResizeEvent *event)
{
	QMainWindow::resizeEvent(event);
	if (_initialied) {
		refresh();
	}
}

void MainView::closeEvent(QCloseEvent * event)
{
	_controller->autoSave();
	QMainWindow::closeEvent(event);
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
    ui->toolBar->addAction(actions->actionAcceleration);
    ui->toolBar->addSeparator();
    ui->toolBar->addAction(actions->actionDisplayLink);
    ui->toolBar->addSeparator();

	ui->menuSimulation->addAction(actions->actionNewSimulation);
	ui->menuSimulation->addAction(actions->actionLoadSimulation);
	ui->menuSimulation->addAction(actions->actionSaveSimulation);
	ui->menuSimulation->addSeparator();
	ui->menuSimulation->addAction(actions->actionRunSimulation);
	ui->menuSimulation->addAction(actions->actionRunStepForward);
	ui->menuSimulation->addAction(actions->actionRunStepBackward);
    ui->menuSimulation->addAction(actions->actionAcceleration);
    ui->menuSimulation->addAction(actions->actionSnapshot);
	ui->menuSimulation->addAction(actions->actionRestore);
    ui->menuSimulation->addSeparator();
    ui->menuSimulation->addAction(actions->actionSimulationChanger);
    ui->menuSimulation->addSeparator();
	ui->menuSimulation->addAction(actions->actionExit);

	ui->menuSettings->addAction(actions->actionComputationSettings);
    ui->menuSettings->addAction(actions->actionEditSimParameters);
    ui->menuSettings->addAction(actions->actionEditSymbols);

	ui->menuView->addAction(actions->actionEditor);
	ui->menuView->addAction(actions->actionMonitor);
	ui->menuView->addSeparator();
	ui->menuView->addAction(actions->actionZoomIn);
	ui->menuView->addAction(actions->actionZoomOut);
    ui->menuView->addAction(actions->actionDisplayLink);
    ui->menuView->addAction(actions->actionFullscreen);
    ui->menuView->addSeparator();
    ui->menuView->addAction(actions->actionGlowEffect);
    ui->menuView->addSeparator();
	ui->menuView->addAction(actions->actionShowCellInfo);
	ui->menuView->addAction(actions->actionCenterSelection);

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
    ui->menuEntity->addSeparator();
    ui->menuEntity->addAction(actions->actionCopyToClipboard);
    ui->menuEntity->addAction(actions->actionPasteFromClipboard);

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
	ui->menuCollection->addAction(actions->actionRandomMultiplier);
	ui->menuCollection->addAction(actions->actionGridMultiplier);

    ui->menuAnalysis->addAction(actions->actionMostFrequentCluster);

	ui->menuHelp->addAction(actions->actionAbout);
	ui->menuEntity->addSeparator();
	ui->menuHelp->addAction(actions->actionDocumentation);
}

void MainView::setupFontsAndColors()
{
	setFont(GuiSettings::getGlobalFont());
	ui->menuSimulation->setFont(GuiSettings::getGlobalFont());
	ui->menuView->setFont(GuiSettings::getGlobalFont());
	ui->menuEntity->setFont(GuiSettings::getGlobalFont());
	ui->menuCollection->setFont(GuiSettings::getGlobalFont());
	ui->menuSettings->setFont(GuiSettings::getGlobalFont());
	ui->menuHelp->setFont(GuiSettings::getGlobalFont());
    ui->menuAnalysis->setFont(GuiSettings::getGlobalFont());

	ui->tpsForcingButton->setStyleSheet(Const::ButtonStyleSheet);
	ui->toolBar->setStyleSheet("background-color: #151540");
	{
		QPalette p = ui->tpsForcingButton->palette();
		p.setColor(QPalette::ButtonText, Const::ButtonTextColor);
		ui->tpsForcingButton->setPalette(p);
	}

	{
		QPalette p = palette();
		p.setColor(QPalette::Background, QColor(0, 0, 0));
		setPalette(p);
	}

}

void MainView::setupWidgets()
{
	auto actions = _actions->getActionHolder();
	ui->tpsForcingButton->setDefaultAction(actions->actionRestrictTPS);

	ui->tpsSpinBox->setValue(_model->getTPS());
	connect(ui->tpsSpinBox, (void(QSpinBox::*)(int))(&QSpinBox::valueChanged), [this](int value) {
        value = std::max(1, value);
		_model->setTPS(value);
		_actions->getActionHolder()->actionRestrictTPS->setChecked(true);
		Q_EMIT _actions->getActionHolder()->actionRestrictTPS->triggered(true);
	});
}

void MainView::setupFullScreen()
{
	bool fullScreen = GuiSettings::getSettingsValue(Const::MainViewFullScreenKey, Const::MainViewFullScreenDefault);
	if (fullScreen) {
		setWindowState(windowState() | Qt::WindowFullScreen);
	}
}

void MainView::monitorClosed()
{
	_actions->getActionHolder()->actionMonitor->setChecked(false);
}


