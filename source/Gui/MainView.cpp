#include <QDesktopServices>
#include <QUrl>
#include <QTimer>
#include <QDockWidget>

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SymbolTable.h"
#include "EngineInterface/SerializationHelper.h"
#include "EngineInterface/SpaceProperties.h"
#include "EngineInterface/SimulationContext.h"

#include "Gui/ToolbarController.h"
#include "Gui/ToolbarContext.h"
#include "Gui/ActionController.h"
#include "Gui/ActionHolder.h"
#include "Gui/MonitorController.h"

#include "GeneralInfoController.h"
#include "DataEditController.h"
#include "DataEditContext.h"
#include "NewSimulationDialog.h"
#include "Settings.h"
#include "MainView.h"
#include "MainController.h"
#include "MainModel.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"
#include "GettingStartedWindow.h"
#include "LoggingController.h"

#include "ui_MainView.h"

MainView::MainView(QWidget * parent)
	: QMainWindow(parent)
	, ui(new Ui::MainView)
{
	ui->setupUi(this);
	_simulationViewWidget = ui->simulationViewWidget;
	_toolbar = new ToolbarController(_simulationViewWidget);
	_dataEditor = new DataEditController(_simulationViewWidget);
	_infoController = new GeneralInfoController(this);
	_actions = new ActionController(this);
	_monitor = new MonitorController(this);
    _logging = new LoggingController(this);
    _gettingStartedWindow = new GettingStartedWindow(this);

    {
        auto gridLayout = new QGridLayout(ui->monitorGroupBox);
        gridLayout->setSpacing(6);
        gridLayout->setVerticalSpacing(0);
        gridLayout->setContentsMargins(9, 9, 9, 9);
        gridLayout->addWidget(_monitor->getWidget(), 0, 0, 1, 1);
    }
    {
        auto gridLayout = new QGridLayout(ui->loggingGroupBox);
        gridLayout->setSpacing(6);
        gridLayout->setVerticalSpacing(0);
        gridLayout->setContentsMargins(9, 9, 9, 9);
        gridLayout->addWidget(_logging->getWidget(), 0, 0, 1, 1);
    }

    connect(_gettingStartedWindow, &GettingStartedWindow::closed, this, &MainView::gettingStartedWindowClosed);
    connect(ui->infobar, &QDockWidget::visibilityChanged, this, &MainView::infobarChanged);
    connect(_simulationViewWidget, &SimulationViewWidget::zoomFactorChanged, _infoController, &GeneralInfoController::setZoomFactor);
}

MainView::~MainView()
{
	delete ui;
}

void MainView::init(
    MainModel* model, 
    MainController* mainController, 
    Serializer* serializer, 
    DataRepository* repository, 
    Notifier* notifier, 
    WebSimulationController* webSimController)
{
	_model = model;
	_controller = mainController;
	_repository = repository;
	_notifier = notifier;

	_infoController->init(ui->infoLabel, mainController);
	_monitor->init(_controller);
	_actions->init(_controller, 
        _model, 
        this, 
        _simulationViewWidget, 
        serializer, 
        _infoController, 
        _dataEditor, 
        _toolbar, 
        _monitor, 
        repository, 
        notifier, 
        webSimController);

	setupMenuAndToolbar();
	setupFontsAndColors();
	setupWidgets();
	setupFullScreen();
	show();

	_initialied = true;
}

void MainView::initGettingStartedWindow()
{
    auto show = GuiSettings::getSettingsValue(Const::GettingStartedWindowKey, Const::GettingStartedWindowKeyDefault);
    _actions->getActionHolder()->actionGettingStarted->setChecked(show);
    toggleGettingStarted(show);
}

void MainView::refresh()
{
	_simulationViewWidget->refresh();
}

void MainView::setupEditors(SimulationController * controller, SimulationAccess* access)
{
	_toolbar->init({ 10, 10 }, _notifier, _repository, controller->getContext(), _actions->getActionHolder());
	_dataEditor->init({ 10, 60 }, _notifier, _repository, controller->getContext());
	_simulationViewWidget->init(_notifier, controller, access, _repository);

	_actions->getActionHolder()->actionEditor->setChecked(false);
}

GeneralInfoController * MainView::getInfoController() const
{
	return _infoController;
}

MonitorController* MainView::getMonitorController() const
{
    return _monitor;
}

void MainView::toggleGettingStarted(bool show)
{
    _gettingStartedWindow->setVisible(show);
}

void MainView::toggleInfobar(bool show)
{
    ui->infobar->setVisible(show);
    QTimer::singleShot(1, [&] { refresh(); });
}

void MainView::showDocumentation()
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
    _actions->close();
	_controller->autoSave();
	QMainWindow::closeEvent(event);
}

void MainView::setupMenuAndToolbar()
{
	auto actions = _actions->getActionHolder();

    ui->menuSimulation->addAction(actions->actionNewSimulation);
    ui->menuSimulation->addAction(actions->actionLoadSimulation);
    ui->menuSimulation->addAction(actions->actionSaveSimulation);
/*
    ui->menuSimulation->addSeparator();
    ui->menuSimulation->addAction(actions->actionWebSimulation);
*/
    ui->menuSimulation->addSeparator();
    ui->menuSimulation->addAction(actions->actionRunSimulation);
    ui->menuSimulation->addAction(actions->actionRunStepForward);
    ui->menuSimulation->addAction(actions->actionRunStepBackward);
    ui->menuSimulation->addAction(actions->actionAcceleration);
    ui->menuSimulation->addAction(actions->actionSnapshot);
    ui->menuSimulation->addAction(actions->actionRestore);
    ui->menuSimulation->addAction(actions->actionRestrictTPS);
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
    ui->menuCollection->addAction(actions->actionNewDisc);
    ui->menuCollection->addAction(actions->actionNewParticles);
    ui->menuCollection->addSeparator();
    ui->menuCollection->addAction(actions->actionColorizeSel);
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

    ui->menuTools->addAction(actions->actionMostFrequentCluster);
    ui->menuTools->addAction(actions->actionSimulationChanger);

    ui->menuHelp->addAction(actions->actionAbout);
    ui->menuEntity->addSeparator();
    ui->menuHelp->addAction(actions->actionGettingStarted);
    ui->menuHelp->addAction(actions->actionDocumentation);

    ui->toolBar->setIconSize(QSize(48, 48));
	ui->toolBar->addSeparator();
    ui->toolBar->addAction(actions->actionZoomIn);
	ui->toolBar->addAction(actions->actionZoomOut);
    ui->toolBar->addAction(actions->actionEditor);
    ui->toolBar->addAction(actions->actionMonitor);
    ui->toolBar->addSeparator();
    ui->toolBar->addAction(actions->actionRunSimulation);
    ui->toolBar->addAction(actions->actionAcceleration);
	ui->toolBar->addAction(actions->actionRunStepBackward);
	ui->toolBar->addAction(actions->actionRunStepForward);
    ui->toolBar->addSeparator();
	ui->toolBar->addAction(actions->actionSnapshot);
	ui->toolBar->addAction(actions->actionRestore);
    ui->toolBar->addSeparator();
    ui->toolBar->addAction(actions->actionDisplayLink);
    ui->toolBar->addSeparator();
}

void MainView::setupFontsAndColors()
{
    ui->generalInfoGroupBox->setFont(GuiSettings::getGlobalFont());
    ui->monitorGroupBox->setFont(GuiSettings::getGlobalFont());
    ui->loggingGroupBox->setFont(GuiSettings::getGlobalFont());

	ui->toolBar->setStyleSheet(Const::ToolbarStyleSheet);
    ui->infobar->setStyleSheet(Const::InfobarStyleSheet);
	{
        QPalette p = palette();
        p.setColor(QPalette::Window, QColor(7, 7, 21));
		setPalette(p);
	}

}

void MainView::setupWidgets()
{
	auto actions = _actions->getActionHolder();
}

void MainView::setupFullScreen()
{
	bool fullScreen = GuiSettings::getSettingsValue(Const::MainViewFullScreenKey, Const::MainViewFullScreenDefault);
	if (fullScreen) {
		setWindowState(windowState() | Qt::WindowFullScreen);
	}
}

void MainView::infobarChanged(bool show)
{
    if (!show) {
        _actions->getActionHolder()->actionMonitor->setChecked(false);
    }
}

void MainView::gettingStartedWindowClosed()
{
    _actions->getActionHolder()->actionGettingStarted->setChecked(false);
}
