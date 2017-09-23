#include <QGraphicsScene>
#include <QTimer>
#include <QScrollBar>
#include <QSpinBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QFont>

#include "Base/ServiceLocator.h"
#include "Model/Settings.h"
#include "Model/SimulationController.h"
#include "Model/Context/SimulationContext.h"
#include "Model/SerializationFacade.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Metadata/SymbolTable.h"
#include "Gui/dialogs/AddEnergyDialog.h"
#include "Gui/dialogs/AddRectStructureDialog.h"
#include "Gui/dialogs/AddHexagonStructureDialog.h"
#include "Gui/dialogs/NewSimulationDialog.h"
#include "Gui/dialogs/SimulationParametersDialog.h"
#include "Gui/dialogs/SymbolTableDialog.h"
#include "Gui/dialogs/SelectionMultiplyRandomDialog.h"
#include "Gui/dialogs/SelectionMultiplyArrangementDialog.h"
#include "Gui/Monitoring/SimulationMonitor.h"
#include "Gui/Assistance/TutorialWindow.h"
#include "Gui/Misc/StartScreenController.h"
#include "Gui/TextEditor/TextEditor.h"
#include "Gui/VisualEditor/ViewportController.h"
#include "Gui/DataEditor/DataEditorController.h"
#include "Gui/DataEditor/DataEditorContext.h"
#include "Gui/Toolbar/ToolbarController.h"
#include "Gui/Toolbar/ToolbarContext.h"
#include "Gui/Settings.h"
#include "Gui/Settings.h"

#include "MainWindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(SimulationController* simController, SimulationAccess* access, QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
	, _simController(simController)
	, _textEditor(new TextEditor(this))
	, _oneSecondTimer(new QTimer(this))
    , _monitor(new SimulationMonitor(this))
    , _tutorialWindow(new TutorialWindow(this))
    , _startScreen(new StartScreenController(this))
{
    ui->setupUi(this);

	_toolbar = new ToolbarController({ 10, 10 }, ui->visualEditor);
	connect(ui->actionEditor, &QAction::triggered, _toolbar->getContext(), &ToolbarContext::show);

	_dataEditor = new DataEditorController({ 10, 60 }, ui->visualEditor);
	connect(ui->actionEditor, &QAction::triggered, _dataEditor->getContext(), &DataEditorContext::show);

	ui->visualEditor->init(simController, access, _dataEditor->getContext());
	connect(_simController, &SimulationController::updateTimestepsPerSecond, [this](int value) {
		_framedata.fps = value;
		updateFrameLabel();
	});
	connect(_simController, &SimulationController::nextTimestepCalculated, [this]() {
		_framedata.frame++;
		updateFrameLabel();
	});

	setupFont();



	TextEditor::TextEditorWidgets microWidgets{ ui->tabClusterWidget2, ui->tabComputerWidget2, ui->tabTokenWidget2, ui->tabSymbolsWidget
		, ui->cellEditor2, ui->clusterEditor2, ui->energyEditor2, ui->metadataEditor2, ui->cellComputerEdit
		, ui->symbolEdit2, ui->selectionEditor2, ui->requestCellButton2, ui->requestEnergyParticleButton2
		, ui->delEntityButton2, ui->delClusterButton2, ui->addTokenButton2, ui->delTokenButton2
		, ui->buttonShowInfo };
	_textEditor->init(microWidgets);

    //set color
    ui->fpsForcingButton->setStyleSheet(BUTTON_STYLESHEET);
    ui->toolBar->setStyleSheet("background-color: #303030");
    QPalette p = ui->fpsForcingButton->palette();
    p.setColor(QPalette::ButtonText, BUTTON_TEXT_COLOR);
    ui->fpsForcingButton->setPalette(p);

    //connect signals/slots for actions
    connect(ui->actionPlay, SIGNAL( triggered(bool) ), this, SLOT(runClicked(bool)));
    connect(ui->actionStep, SIGNAL( triggered(bool) ), this, SLOT(stepForwardClicked()));
    connect(ui->actionStepBack, SIGNAL( triggered(bool) ), this, SLOT(stepBackClicked()));
    connect(ui->actionEditor, SIGNAL(triggered(bool)), this, SLOT(setVisualMode(bool)));
    connect(ui->actionZoomIn, SIGNAL(triggered(bool)), ui->visualEditor, SLOT(zoomIn()));
    connect(ui->actionZoomIn, SIGNAL(triggered(bool)), this, SLOT(updateFrameLabel()));
    connect(ui->actionZoomOut, SIGNAL(triggered(bool)), ui->visualEditor, SLOT(zoomOut()));
    connect(ui->actionZoomOut, SIGNAL(triggered(bool)), this, SLOT(updateFrameLabel()));
    connect(ui->actionSnapshot, SIGNAL(triggered(bool)), this, SLOT(snapshotUniverse()));
    connect(ui->actionRestore, SIGNAL(triggered(bool)), this, SLOT(restoreUniverse()));
    connect(ui->actionAlienMonitor, SIGNAL(triggered(bool)), _monitor, SLOT(setVisible(bool)));
    connect(ui->actionAlienMonitor, SIGNAL(triggered(bool)), this, SLOT(alienMonitorTriggered(bool)));
    connect(ui->actionCopyCell, SIGNAL(triggered(bool)), this, SLOT(copyCell()));
    connect(ui->actionPasteCell, SIGNAL(triggered(bool)), this, SLOT(pasteCell()));
    connect(ui->actionCopy_cell_extension, SIGNAL(triggered(bool)), this, SLOT(copyExtendedSelection()));
    connect(ui->actionPaste_cell_extension, SIGNAL(triggered(bool)), this, SLOT(pasteExtendedSelection()));
    connect(ui->actionSave_cell_extension, SIGNAL(triggered(bool)), this, SLOT(saveExtendedSelection()));
    connect(ui->actionLoad_cell_extension, SIGNAL(triggered(bool)), this, SLOT(loadExtendedSelection()));
    connect(ui->actionNewCell, SIGNAL(triggered(bool)), this, SLOT(addCell()));
    connect(ui->actionAddBlockStructure, SIGNAL(triggered(bool)), this, SLOT(addBlockStructure()));
    connect(ui->actionAddHexagonStructure, SIGNAL(triggered(bool)), this, SLOT(addHexagonStructure()));
    connect(ui->actionAddEnergyParticle, SIGNAL(triggered(bool)), this, SLOT(addEnergyParticle()));
    connect(ui->actionAddRandomEnergy, SIGNAL(triggered(bool)), this, SLOT(addRandomEnergy()));
    connect(ui->actionNewSimulation, SIGNAL(triggered(bool)), this, SLOT(newSimulation()));
    connect(ui->actionSave_universe, SIGNAL(triggered(bool)), this, SLOT(saveSimulation()));
    connect(ui->actionLoad_universe, SIGNAL(triggered(bool)), this, SLOT(loadSimulation()));
    connect(ui->actionSave_symbols, SIGNAL(triggered(bool)), this, SLOT(saveSymbols()));
    connect(ui->actionLoad_symbols, SIGNAL(triggered(bool)), this, SLOT(loadSymbols()));
    connect(ui->actionMerge_with, SIGNAL(triggered(bool)), this, SLOT(loadSymbolsWithMerging()));
    connect(ui->actionNewToken, SIGNAL(triggered(bool)), _textEditor, SLOT(addTokenClicked()));
    connect(ui->actionDeleteToken, SIGNAL(triggered(bool)), _textEditor, SLOT(delTokenClicked()));
    connect(ui->actionCopyToken, SIGNAL(triggered(bool)), _textEditor, SLOT(copyTokenClicked()));
    connect(ui->actionPasteToken, SIGNAL(triggered(bool)), _textEditor, SLOT(pasteTokenClicked()));
    connect(ui->actionEditSimulationParameters, SIGNAL(triggered(bool)), this, SLOT(editSimulationParameters()));
    connect(ui->actionLoadSimulationParameters, SIGNAL(triggered(bool)), this, SLOT(loadSimulationParameters()));
    connect(ui->actionSaveSimulationParameters, SIGNAL(triggered(bool)), this, SLOT(saveSimulationParameters()));
    connect(ui->actionEditSymbols, SIGNAL(triggered(bool)), this, SLOT(editSymbolTable()));
    connect(ui->actionFullscreen, SIGNAL(triggered(bool)), this, SLOT(fullscreen(bool)));
    connect(ui->actionRandom, SIGNAL(triggered(bool)), this, SLOT(multiplyRandomExtendedSelection()));
    connect(ui->actionArrangement, SIGNAL(triggered(bool)), this, SLOT(multiplyArrangementExtendedSelection()));
    connect(ui->actionAboutAlien, SIGNAL(triggered(bool)), this, SLOT(aboutAlien()));
    connect(ui->actionDeleteCell, SIGNAL(triggered(bool)), _textEditor, SLOT(delSelectionClicked()));
    connect(ui->actionDeleteExtension, SIGNAL(triggered(bool)), _textEditor, SLOT(delExtendedSelectionClicked()));
    connect(ui->actionTutorial, SIGNAL(triggered(bool)), _tutorialWindow, SLOT(setVisible(bool)));
    connect(_tutorialWindow, SIGNAL(closed()), this, SLOT(tutorialClosed()));

    //connect simulation monitor
    connect(_monitor, SIGNAL(closed()), this, SLOT(alienMonitorClosed()));

    //connect fps widgets
    connect(ui->fpsForcingButton, SIGNAL(toggled(bool)), this, SLOT(fpsForcingButtonClicked(bool)));
    connect(ui->fpsForcingSpinBox, SIGNAL(valueChanged(int)), this, SLOT(fpsForcingSpinboxClicked()));

    //setup micro editor
    _textEditor->setVisible(false);

    //init widgets
    QFont f = ui->frameLabel->font();
    f.setBold(false);
    f.setItalic(true);
    ui->frameLabel->setFont(f);

    //init timer
    connect(_oneSecondTimer, SIGNAL(timeout()), this, SLOT(oneSecondTimeout()));
    _oneSecondTimer->start(1000);

    //connect and run start screen
    connect(_startScreen, SIGNAL(startScreenFinished()), SLOT(startScreenFinished()));
    _startScreen->runStartScreen(ui->visualEditor->getGraphicsView());
	ui->frameLabel->setVisible(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::newSimulation ()
{
/*
    NewSimulationDialog d(_simController->getSimulationContext());
    if( d.exec() ) {
		stopSimulation();

        //create new simulation
        _simController->newUniverse(d.getSize(), d.getNewSymbolTableRef(), d.getNewSimulationParameters());
		SimulationParameters* parameters = _simController->getSimulationContext()->getSimulationParameters();
        _simController->addRandomEnergy (d.getEnergy(), parameters->cellMinEnergy);

		updateControllerAndEditors();
    }
*/
}

void MainWindow::loadSimulation ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Simulation", "", "Alien Simulation (*.sim)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {
			stopSimulation();

            //read simulation data
            QDataStream in(&file);
/*
            _simController->loadUniverse(in);
*/
			file.close();

			updateControllerAndEditors();
		}
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The specified simulation could not loaded.");
            msgBox.exec();
        }
    }
}

void MainWindow::saveSimulation ()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Simulation", "", "Alien Simulation(*.sim)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {
            QDataStream out(&file);
/*
            _simController->saveUniverse(out);
*/
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The simulation could not saved.");
            msgBox.exec();
        }
    }
}

void MainWindow::runClicked (bool run)
{
    if( run ) {
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

    _undoUniverserses.clear();
    _textEditor->requestUpdate();
    _simController->setRun(run);
}

void MainWindow::stepForwardClicked ()
{
    //update widgets
    ui->actionSave_cell_extension->setEnabled(false);
    ui->actionCopy_cell_extension->setEnabled(false);
    ui->actionStepBack->setEnabled(true);
    ui->menuMultiplyExtension->setEnabled(false);
    ui->actionCopyCell->setEnabled(false);
    ui->actionDeleteCell->setEnabled(false);
    ui->actionDeleteExtension->setEnabled(false);
    _textEditor->requestUpdate();

    //save old universe
    QByteArray b;
    QDataStream out(&b, QIODevice::WriteOnly);
/*
    _simController->saveUniverse(out);
*/
    _undoUniverserses.push(b);

    //calc next time step
/*
    _simController->requestNextTimestep();
*/
}

void MainWindow::stepBackClicked ()
{
    ui->actionSave_cell_extension->setEnabled(false);
    ui->actionCopy_cell_extension->setEnabled(false);
    ui->menuMultiplyExtension->setEnabled(false);
    ui->actionCopyCell->setEnabled(false);
    ui->actionDeleteCell->setEnabled(false);
    ui->actionDeleteExtension->setEnabled(false);

    //load old universe
    QByteArray b = _undoUniverserses.pop();
    QDataStream in(&b, QIODevice::ReadOnly);

    //read simulation data
/*
    _simController->loadUniverse(in);

    //force simulator to update other coordinators
    _simController->updateUniverse();
	*/

    //no step back any more?
    if( _undoUniverserses.isEmpty() )
        ui->actionStepBack->setEnabled(false);
}

void MainWindow::snapshotUniverse ()
{
    ui->actionRestore->setEnabled(true);

    //save old universe
    _snapshot.clear();
    QDataStream out(&_snapshot, QIODevice::WriteOnly);
/*
    _simController->saveUniverse(out);
    ui->visualEditor->serializeViewMatrix(out);
*/}

void MainWindow::restoreUniverse ()
{

    //load old universe
    QDataStream in(&_snapshot, QIODevice::ReadOnly);

    //read simulation data
/*
    _simController->loadUniverse(in);

    //reset editors
    ui->visualEditor->reset();

    //load view
    ui->visualEditor->loadViewMatrix(in);

    //force simulator to update other coordinators
    _simController->updateUniverse();
*/

    //hide "step back" button
    _undoUniverserses.clear();
    ui->actionStepBack->setEnabled(false);
}

void MainWindow::editSimulationParameters ()
{
/*
    SimulationParametersDialog d(_simController->getSimulationContext()->getSimulationParameters());
    if( d.exec() ) {
		auto newParameters = d.getSimulationParameters();
		*_simController->getSimulationContext()->getSimulationParameters() = newParameters;
    }
*/
}

void MainWindow::loadSimulationParameters ()
{
/*
    QString fileName = QFileDialog::getOpenFileName(this, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {

			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
            QDataStream in(&file);
			SimulationParameters* parameters = facade->deserializeSimulationParameters(in);
            file.close();
			_simController->getSimulationContext()->getSimulationParameters()->setParameters(parameters);
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The specified simulation parameter file could not loaded.");
            msgBox.exec();
        }
    }
*/
}

void MainWindow::saveSimulationParameters ()
{
/*
    QString fileName = QFileDialog::getSaveFileName(this, "Save Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {

			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
			QDataStream out(&file);
			SimulationParameters* parameters = _simController->getSimulationContext()->getSimulationParameters();
			facade->serializeSimulationParameters(parameters, out);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The simulation parameters could not saved.");
            msgBox.exec();
        }
    }
*/
}

void MainWindow::fullscreen (bool triggered)
{
    QMainWindow::setWindowState(QMainWindow::windowState() ^ Qt::WindowFullScreen);
}

void MainWindow::setVisualMode (bool editMode)
{
    if( !editMode )
        _textEditor->requestUpdate();
    ui->actionPlay->setChecked(false);
    ui->actionStep->setEnabled(true);
    ui->actionPlay->setIcon(QIcon("://Icons/play.png"));

    //update macro editor and button
    if( editMode) {
        ui->visualEditor->setActiveScene(ActiveScene::ShapeScene);
        ui->actionEditor->setIcon(QIcon("://Icons/microscope_active.png"));
    }
    else {
        ui->visualEditor->setActiveScene(ActiveScene::PixelScene);
        ui->actionEditor->setIcon(QIcon("://Icons/microscope.png"));
        cellDefocused();
    }

    //update micro editor
    _textEditor->setVisible(editMode);

    //stop running
    _simController->setRun(false);
}

void MainWindow::alienMonitorTriggered (bool on)
{
    if( on ) {
        ui->actionAlienMonitor->setIcon(QIcon("://Icons/monitor_active.png"));
    }
    else {
        ui->actionAlienMonitor->setIcon(QIcon("://Icons/monitor.png"));
    }
}

void MainWindow::alienMonitorClosed()
{
    ui->actionAlienMonitor->setChecked(false);
    ui->actionAlienMonitor->setIcon(QIcon("://Icons/monitor.png"));
}

void MainWindow::addCell ()
{
/*    ui->visualEditor->newCellRequested();

    if( !_textEditor->isVisible() )
        _simController->updateUniverse();
*/
}

void MainWindow::addEnergyParticle ()
{
/*    ui->visualEditor->newEnergyParticleRequested();

    if( !_textEditor->isVisible() )
        _simController->updateUniverse();
*/
}

void MainWindow::addRandomEnergy ()
{
    AddEnergyDialog d;
/*
    if( d.exec() ) {
        _simController->addRandomEnergy(d.getTotalEnergy(), d.getMaxEnergyPerParticle());
        _simController->updateUniverse();
    }
*/
}

void MainWindow::copyCell ()
{
    //serialize cell
    Cell* focusCell = _textEditor->getFocusedCell();
    QDataStream out(&_serializedCellData, QIODevice::WriteOnly);
    quint64 clusterId;
    quint64 cellId;
/*
    _simController->saveCell(out, focusCell, clusterId, cellId);
*/

    //set actions
    ui->actionPasteCell->setEnabled(true);
}

void MainWindow::pasteCell ()
{
    QDataStream in(&_serializedCellData, QIODevice::ReadOnly);
/*
    _simController->loadCell(in, ui->visualEditor->getViewCenterPosWithInc());

    //force simulator to update other coordinators
    _simController->updateUniverse();
*/
}

void MainWindow::editSymbolTable ()
{
/*
	SymbolTable* symbolTable = _simController->getSimulationContext()->getSymbolTable();
	if (!symbolTable)
		return;
    SymbolTableDialog d(symbolTable);
    if (d.exec()) {

        //update symbol table
		symbolTable->setTable(d.getNewSymbolTable());

        //update editor
        _microEditor->update();
    }
*/
}

void MainWindow::loadSymbols ()
{
	QString fileName = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {

/*
            QDataStream in(&file);
			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
			SymbolTable* oldSymbolTable = _simController->getSimulationContext()->getSymbolTable();
			SymbolTable* newSymbolTable = facade->deserializeSymbolTable(in);
			oldSymbolTable->setTable(*newSymbolTable);
			delete newSymbolTable;
            file.close();
*/

            _textEditor->update();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
}

void MainWindow::saveSymbols ()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Symbol Table", "", "Alien Symbol Table (*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {

/*
            QDataStream out(&file);
			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
			SymbolTable* symbolTable = _simController->getSimulationContext()->getSymbolTable();
			facade->serializeSymbolTable(symbolTable, out);
            file.close();
*/
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The symbol table could not saved.");
            msgBox.exec();
        }
    }
}

void MainWindow::loadSymbolsWithMerging ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {

/*
			QDataStream in(&file);
			SerializationFacade* facade = ServiceLocator::getInstance().getService<SerializationFacade>();
			SymbolTable* oldSymbolTable = _simController->getSimulationContext()->getSymbolTable();
			SymbolTable* newSymbolTable = facade->deserializeSymbolTable(in);
			oldSymbolTable->mergeTable(*newSymbolTable);
			delete newSymbolTable;
			file.close();
*/

            _textEditor->update();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
}

void MainWindow::addBlockStructure ()
{
/*
    AddRectStructureDialog d(_simController->getSimulationContext()->getSimulationParameters());
    if( d.exec() ) {
        QVector2D center = ui->visualEditor->getViewCenterPosWithInc();
       _simController->addBlockStructure(center, d.getBlockSizeX(), d.getBlockSizeY(), QVector2D(d.getDistance(), d.getDistance(), 0.0)
		   , d.getInternalEnergy());
    }
*/
}

void MainWindow::addHexagonStructure ()
{
/*
    AddHexagonStructureDialog d(_simController->getSimulationContext()->getSimulationParameters());
    if( d.exec() ) {
        QVector2D center = ui->visualEditor->getViewCenterPosWithInc();
       _simController->addHexagonStructure(center, d.getLayers(), d.getDistance(), d.getInternalEnergy());
    }
*/
}

void MainWindow::loadExtendedSelection ()
{
/*
    QString fileName = QFileDialog::getOpenFileName(this, "Load Ensemble", "", "Alien Ensemble (*.ens)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {
            QDataStream in(&file);
            QMap< quint64, quint64 > oldNewCellIdMap;
            QMap< quint64, quint64 > oldNewClusterIdMap;
            QList< CellCluster* > newClusters;
            QList< EnergyParticle* > newEnergyParticles;
            _simController->loadExtendedSelection(in, ui->visualEditor->getViewCenterPosWithInc(), newClusters,  newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap);
            file.close();

            _simController->updateUniverse();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The specified ensemble could not loaded.");
            msgBox.exec();
        }
    }
*/
}

void MainWindow::saveExtendedSelection ()
{
/*
    QString fileName = QFileDialog::getSaveFileName(this, "Save Ensemble", "", "Alien Ensemble (*.ens)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {

            //get selected clusters and energy particles
            QList< CellCluster* > clusters;
            QList< EnergyParticle* > es;
            ui->visualEditor->getExtendedSelection(clusters, es);

            //serialize lists
            QDataStream out(&file);
            QList< quint64 > clusterIds;
            QList< quint64 > cellIds;
            _simController->saveExtendedSelection(out, clusters, es, clusterIds, cellIds);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The selected ensemble could not saved.");
            msgBox.exec();
        }
    }
*/
}

void MainWindow::copyExtendedSelection ()
{
/*
    //get selected clusters and energy particles
    QList< CellCluster* > clusters;
    QList< EnergyParticle* > es;
    ui->visualEditor->getExtendedSelection(clusters, es);

    //serialize lists
    QDataStream out(&_serializedEnsembleData, QIODevice::WriteOnly);
    QList< quint64 > clusterIds;
    QList< quint64 > cellIds;
    _simController->saveExtendedSelection(out, clusters, es, clusterIds, cellIds);

    //set actions
    ui->actionPaste_cell_extension->setEnabled(true);
*/
}

void MainWindow::pasteExtendedSelection ()
{
/*
    QDataStream in(&_serializedEnsembleData, QIODevice::ReadOnly);
    QMap< quint64, quint64 > oldNewCellIdMap;
    QMap< quint64, quint64 > oldNewClusterIdMap;
    QList< CellCluster* > newClusters;
    QList< EnergyParticle* > newEnergyParticles;
    _simController->loadExtendedSelection(in, ui->visualEditor->getViewCenterPosWithInc(), newClusters, newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap);

    //force simulator to update other coordinators
    _simController->updateUniverse();
*/
}

void MainWindow::multiplyRandomExtendedSelection ()
{
/*
    SelectionMultiplyRandomDialog d;
    if( d.exec() ) {

        //get selected clusters and energy particles
        QList< CellCluster* > clusters;
        QList< EnergyParticle* > es;
        ui->visualEditor->getExtendedSelection(clusters, es);

        //serialize lists
        QByteArray serializedEnsembleData;
        QDataStream out(&serializedEnsembleData, QIODevice::WriteOnly);
        QList< quint64 > clusterIds;
        QList< quint64 > cellIds;
        _simController->saveExtendedSelection(out, clusters, es, clusterIds, cellIds);

        //read list and rebuild structure n times
        for(int i = 0; i < d.getNumber(); ++i) {
            QDataStream in(&serializedEnsembleData, QIODevice::ReadOnly);
            QMap< quint64, quint64 > oldNewCellIdMap;
            QMap< quint64, quint64 > oldNewClusterIdMap;
            QList< CellCluster* > newClusters;
            QList< EnergyParticle* > newEnergyParticles;
			IntVector2D universeSize = _simController->getUniverseSize();
            QVector2D pos(NumberGenerator::getInstance().getInstance().random(0.0, universeSize.x), NumberGenerator::getInstance().getInstance().random(0.0, universeSize.y), 0.0);
            _simController->loadExtendedSelection(in, pos, newClusters, newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap, false);

            //randomize angles and velocities if desired
            if( d.randomizeAngle() )
                _simController->rotateExtendedSelection(NumberGenerator::getInstance().getInstance().random(d.randomizeAngleMin(), d.randomizeAngleMax()), newClusters, newEnergyParticles);
            if( d.randomizeVelX() )
                _simController->setVelocityXExtendedSelection(NumberGenerator::getInstance().getInstance().random(d.randomizeVelXMin(), d.randomizeVelXMax()), newClusters, newEnergyParticles);
            if( d.randomizeVelY() )
                _simController->setVelocityYExtendedSelection(NumberGenerator::getInstance().getInstance().random(d.randomizeVelYMin(), d.randomizeVelYMax()), newClusters, newEnergyParticles);
            if( d.randomizeAngVel() )
                _simController->setAngularVelocityExtendedSelection(NumberGenerator::getInstance().getInstance().random(d.randomizeAngVelMin(), d.randomizeAngVelMax()), newClusters);

            //draw selection
            _simController->drawToMapExtendedSelection(newClusters, newEnergyParticles);
        }

        _simController->updateUniverse();
    }
*/
}

void MainWindow::multiplyArrangementExtendedSelection ()
{
/*
    //get selected clusters and energy particles
    QList< CellCluster* > clusters;
    QList< EnergyParticle* > es;
    ui->visualEditor->getExtendedSelection(clusters, es);

    //celc center
    QVector2D centerPos = _simController->getCenterPosExtendedSelection(clusters, es);
    SelectionMultiplyArrangementDialog d(centerPos);
    if( d.exec() ) {

        //serialize lists
        QByteArray serializedEnsembleData;
        QDataStream out(&serializedEnsembleData, QIODevice::WriteOnly);
        QList< quint64 > clusterIds;
        QList< quint64 > cellIds;
        _simController->saveExtendedSelection(out, clusters, es, clusterIds, cellIds);

        //read list and rebuild structure n x m times
        for(int i = 0; i < d.getHorizontalNumber(); ++i) {
            for(int j = 0; j < d.getVerticalNumber(); ++j) {
                QDataStream in(&serializedEnsembleData, QIODevice::ReadOnly);
                QMap< quint64, quint64 > oldNewCellIdMap;
                QMap< quint64, quint64 > oldNewClusterIdMap;
                QList< CellCluster* > newClusters;
                QList< EnergyParticle* > newEnergyParticles;
                QVector2D pos(d.getInitialPosX() + (qreal)i*d.getHorizontalInterval(),
                              d.getInitialPosY() + (qreal)j*d.getVerticalInterval(), 0.0);
                _simController->loadExtendedSelection(in, pos, newClusters, newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap, false);

                //set angles and velocities
                if( d.changeAngle() ) {
                    qreal angle = d.getInitialAngle()+(qreal)i*d.getHorizontalAngleIncrement()+(qreal)j*d.getVerticalAngleIncrement();
                    _simController->rotateExtendedSelection(angle, newClusters, newEnergyParticles);
                }
                if( d.changeVelocityX() ) {
                    qreal velX = d.getInitialVelX()+(qreal)i*d.getHorizontalVelocityXIncrement()+(qreal)j*d.getVerticalVelocityXIncrement();
                    _simController->setVelocityXExtendedSelection(velX, newClusters, newEnergyParticles);
                }
                if( d.changeVelocityY() ) {
                    qreal velY = d.getInitialVelY()+(qreal)j*d.getHorizontalVelocityYIncrement()+(qreal)j*d.getVerticalVelocityYIncrement();
                    _simController->setVelocityYExtendedSelection(velY, newClusters, newEnergyParticles);
                }
                if( d.changeAngularVelocity() ) {
                    qreal angVel = d.getInitialAngVel()+(qreal)i*d.getHorizontalAngularVelocityIncrement()+(qreal)j*d.getVerticalAngularVelocityIncrement();
                    _simController->setAngularVelocityExtendedSelection(angVel, newClusters);
                }

                //draw selection
                _simController->drawToMapExtendedSelection(newClusters, newEnergyParticles);
            }
        }

        //delete original cluster
        _simController->delExtendedSelection(clusters, es);

        _simController->updateUniverse();
    }
*/
}

void MainWindow::aboutAlien ()
{
    QMessageBox msgBox(QMessageBox::Information,"about artificial life environment (alien)", "Developed by Christian Heinemann.");
    msgBox.exec();
}

void MainWindow::tutorialClosed()
{
    ui->actionTutorial->setChecked(false);
}

void MainWindow::oneSecondTimeout ()
{
/*
    _monitor->update(_simController->getMonitorData());
*/
	updateFrameLabel();
}

void MainWindow::fpsForcingButtonClicked (bool toggled)
{
/*
    if( toggled ) {
        _simController->forceFps(ui->fpsForcingSpinBox->value());
        QPalette p = ui->fpsForcingButton->palette();
        p.setColor(QPalette::ButtonText, BUTTON_TEXT_HIGHLIGHT_COLOR);
        ui->fpsForcingButton->setPalette(p);
    }
    else {
        _simController->forceFps(0);
        QPalette p = ui->fpsForcingButton->palette();
        p.setColor(QPalette::ButtonText, BUTTON_TEXT_COLOR);
        ui->fpsForcingButton->setPalette(p);
    }
*/
}

void MainWindow::fpsForcingSpinboxClicked ()
{
/*
    if( ui->fpsForcingButton->isChecked() )
        _simController->forceFps(ui->fpsForcingSpinBox->value());
*/
}

void MainWindow::numTokenChanged (int numToken, int maxToken, bool pasteTokenPossible)
{
    if( numToken < maxToken) {
        ui->actionNewToken->setEnabled(true);
        if( pasteTokenPossible )
            ui->actionPasteToken->setEnabled(true);
        else
            ui->actionPasteToken->setEnabled(false);
    }
    else {
        ui->actionNewToken->setEnabled(false);
        ui->actionPasteToken->setEnabled(false);
    }

    if( numToken > 0 ) {
        ui->actionCopyToken->setEnabled(true);
        ui->actionDeleteToken->setEnabled(true);
    }
    else {
        ui->actionCopyToken->setEnabled(false);
        ui->actionDeleteToken->setEnabled(false);
    }
}

void MainWindow::cellFocused (Cell* cell)
{
    if( _textEditor->isVisible() ) {
        ui->actionSave_cell_extension->setEnabled(true);
        ui->actionCopy_cell_extension->setEnabled(true);
        ui->menuMultiplyExtension->setEnabled(true);
        ui->actionCopyCell->setEnabled(true);
        ui->actionDeleteCell->setEnabled(true);
        ui->actionDeleteExtension->setEnabled(true);
    }
}

void MainWindow::cellDefocused ()
{
    ui->actionSave_cell_extension->setEnabled(false);
    ui->actionCopy_cell_extension->setEnabled(false);
    ui->menuMultiplyExtension->setEnabled(false);
    ui->actionCopyCell->setEnabled(false);
    ui->actionDeleteCell->setEnabled(false);
    ui->actionDeleteExtension->setEnabled(false);
}

void MainWindow::energyParticleFocused (Particle* e)
{
    if( _textEditor->isVisible() ) {
        ui->actionSave_cell_extension->setEnabled(true);
        ui->actionCopy_cell_extension->setEnabled(true);
        ui->menuMultiplyExtension->setEnabled(true);
        ui->actionCopyCell->setEnabled(false);
        ui->actionDeleteCell->setEnabled(true);
        ui->actionDeleteExtension->setEnabled(true);
    }
}

void MainWindow::entitiesSelected (int numCells, int numEnergyParticles)
{
    if( (numCells > 0) || (numEnergyParticles > 0) ) {
        ui->actionSave_cell_extension->setEnabled(true);
        ui->actionCopy_cell_extension->setEnabled(true);
        ui->menuMultiplyExtension->setEnabled(true);
        ui->actionDeleteCell->setEnabled(true);
        ui->actionDeleteExtension->setEnabled(true);
    }
    else {
        ui->actionSave_cell_extension->setEnabled(false);
        ui->actionCopy_cell_extension->setEnabled(false);
        ui->menuMultiplyExtension->setEnabled(false);
        ui->actionDeleteExtension->setEnabled(false);
    }
}

void MainWindow::updateFrameLabel ()
{
	ui->frameLabel->setText(QString("Frame: %1  FPS: %2  Magnification: %3x")
		.arg(_framedata.frame, 9, 10, QLatin1Char('0'))
		.arg(_framedata.fps, 5, 10, QLatin1Char('0'))
		.arg(ui->visualEditor->getZoomFactor()));
}

void MainWindow::startScreenFinished ()
{
    ui->actionEditor->setEnabled(true);
    ui->actionZoomIn->setEnabled(true);
    ui->actionZoomOut->setEnabled(true);
	ui->frameLabel->setVisible(true);
	updateFrameLabel();
}

void MainWindow::setupFont()
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
}

void MainWindow::changeEvent(QEvent *e)
{
    QMainWindow::changeEvent(e);
    switch (e->type()) {
    case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        break;
    }
}

void MainWindow::stopSimulation()
{
	ui->actionPlay->setChecked(false);
	ui->actionStepBack->setEnabled(false);
	_undoUniverserses.clear();
	runClicked(false);
}

void MainWindow::updateControllerAndEditors()
{
/*
	ui->visualEditor->reset();
	_textEditor->update();
	_simController->updateUniverse();
*/
}


