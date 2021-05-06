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

#include "ToolbarController.h"
#include "ToolbarContext.h"
#include "ActionController.h"
#include "ActionHolder.h"
#include "MonitorController.h"

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
#include "StartupController.h"
#include "SimulationViewController.h"

#include "ui_MainView.h"

MainView::MainView(QWidget * parent)
	: QMainWindow(parent)
	, ui(new Ui::MainView)
{
	ui->setupUi(this);
    _simulationViewController = new SimulationViewController(this);
    auto simulationViewWidget = _simulationViewController->getWidget();

    _toolbar = new ToolbarController(simulationViewWidget);
    _dataEditor = new DataEditController(simulationViewWidget);
	_infoController = new GeneralInfoController(this);
	_actions = new ActionController(this);
	_monitor = new MonitorController(this);
    _logging = new LoggingController(this);
    _gettingStartedWindow = new GettingStartedWindow(this);

    simulationViewWidget->setParent(ui->centralWidget);
    simulationViewWidget->setObjectName(QString::fromUtf8("simulationViewWidget"));
    QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
    sizePolicy1.setHorizontalStretch(0);
    sizePolicy1.setVerticalStretch(0);
    sizePolicy1.setHeightForWidth(simulationViewWidget->sizePolicy().hasHeightForWidth());
    simulationViewWidget->setSizePolicy(sizePolicy1);
    simulationViewWidget->setMinimumSize(QSize(0, 0));
    simulationViewWidget->setMaximumSize(QSize(16777215, 16777215));
    simulationViewWidget->setLayoutDirection(Qt::LeftToRight);
    ui->gridLayout->addWidget(simulationViewWidget, 0, 0, 1, 1);

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
    connect(_simulationViewController, &SimulationViewController::zoomFactorChanged, _infoController, &GeneralInfoController::setZoomFactor);
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
    WebSimulationController* webSimController,
    StartupController* startupController)
{
	_model = model;
	_controller = mainController;
	_repository = repository;
	_notifier = notifier;
    _startupController = startupController;

	_infoController->init(ui->infoLabel, mainController);
	_monitor->init(_controller);
    _actions->init(
        _controller, 
        _model, 
        this, 
        _simulationViewController, 
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

    setupStartupWidget();
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
	_simulationViewController->refresh();
}

void MainView::initSimulation(SimulationController * controller, SimulationAccess* access)
{
	_toolbar->init({ 10, 10 }, _notifier, _repository, controller->getContext(), _actions->getActionHolder());
	_dataEditor->init({ 10, 60 }, _notifier, _repository, controller->getContext());
	_simulationViewController->init(_notifier, controller, access, _repository);

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
    _closing = true;
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

    ui->menuEdit->addAction(actions->actionEditor);
    ui->menuEdit->addAction(actions->actionActionMode);

    ui->menuView->addAction(actions->actionMonitor);
    ui->menuView->addSeparator();
    ui->menuView->addAction(actions->actionZoomIn);
    ui->menuView->addAction(actions->actionZoomOut);
    ui->menuView->addAction(actions->actionDisplayLink);
    ui->menuView->addAction(actions->actionFullscreen);
    ui->menuView->addSeparator();
    auto visualEffects = new QMenu("Visual effects", this);
    visualEffects->addAction(actions->actionGlowEffect);
    visualEffects->addAction(actions->actionMotionEffect);
    ui->menuView->addMenu(visualEffects);
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
    ui->menuCollection->addAction(actions->actionLoadCol);
    ui->menuCollection->addAction(actions->actionSaveCol);
    ui->menuCollection->addAction(actions->actionCopyCol);
    ui->menuCollection->addAction(actions->actionPasteCol);
    ui->menuCollection->addAction(actions->actionDeleteSel);
    ui->menuCollection->addAction(actions->actionDeleteCol);
    ui->menuCollection->addSeparator();
    ui->menuCollection->addAction(actions->actionColorizeSel);
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
    ui->toolBar->addAction(actions->actionActionMode);
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

void MainView::setupStartupWidget()
{
    auto startupWidget = _startupController->getWidget();
    auto simulationViewWidget = _simulationViewController->getWidget();
    startupWidget->setParent(simulationViewWidget);
    auto posX = simulationViewWidget->width() / 2 - startupWidget->width() / 2;
    auto posY = simulationViewWidget->height() / 2 - startupWidget->height() / 2;
    startupWidget->move(posX, posY);
    startupWidget->setVisible(true);
}

void MainView::infobarChanged(bool show)
{
    if (!show && !_closing) {
        _actions->getActionHolder()->actionMonitor->setChecked(false);
    }
}

void MainView::gettingStartedWindowClosed()
{
    _actions->getActionHolder()->actionGettingStarted->setChecked(false);
}
