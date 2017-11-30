#include "Model/Api/SimulationController.h"

#include "Gui/Toolbar/ToolbarController.h"
#include "Gui/Toolbar/ToolbarContext.h"
#include "DataEditController.h"
#include "DataEditContext.h"
#include "Settings.h"
#include "MainView.h"
#include "MainController.h"

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

void MainView::init(MainModel * model, MainController * controller)
{
	_model = model;
	_controller = controller;

	connectActions();
	setupTheme();
	setWindowState(windowState() | Qt::WindowFullScreen);
	show();
}

void MainView::refresh()
{
	ui->visualEditor->refresh();
}

void MainView::setupEditors(SimulationController * controller, DataManipulator* manipulator, Notifier* notifier)
{
	_toolbar = new ToolbarController(ui->visualEditor);
	connect(ui->actionEditor, &QAction::triggered, _toolbar->getContext(), &ToolbarContext::show);

	_dataEditor = new DataEditController(ui->visualEditor);
	connect(ui->actionEditor, &QAction::triggered, _dataEditor->getContext(), &DataEditContext::show);

	_toolbar->init({ 10, 10 }, notifier, manipulator, controller->getContext());
	_dataEditor->init({ 10, 60 }, notifier, manipulator, controller->getContext());
	ui->visualEditor->init(notifier, controller, manipulator);
}

void MainView::connectActions()
{
	connect(ui->actionExit, &QAction::triggered, this, &QMainWindow::close);
	connect(ui->actionPlay, &QAction::triggered, this, &MainView::runClicked);
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

	ui->fpsForcingButton->setStyleSheet(GuiSettings::ButtonStyleSheet);
	ui->toolBar->setStyleSheet("background-color: #303030");
	QPalette p = ui->fpsForcingButton->palette();
	p.setColor(QPalette::ButtonText, GuiSettings::ButtonTextColor);
	ui->fpsForcingButton->setPalette(p);
}

void MainView::runClicked(bool run)
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

	_controller->onRunSimulation(run);
}

