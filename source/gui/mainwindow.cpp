#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "microeditor.h"
#include "dialogs/addenergydialog.h"
#include "dialogs/addrectstructuredialog.h"
#include "dialogs/addhexagonstructuredialog.h"
#include "dialogs/newsimulationdialog.h"
#include "dialogs/simulationparametersdialog.h"
#include "dialogs/symboltabledialog.h"
#include "dialogs/selectionmultiplyrandomdialog.h"
#include "dialogs/selectionmultiplyarrangementdialog.h"
#include "monitoring/simulationmonitor.h"
#include "assistance/tutorialwindow.h"
#include "misc/startscreencontroller.h"
#include "global/guisettings.h"
#include "global/editorsettings.h"
#include "global/globalfunctions.h"
#include "global/simulationsettings.h"
#include "model/metadatamanager.h"
#include "model/aliensimulator.h"
#include "model/entities/aliencell.h"
#include "model/entities/aliencellcluster.h"
#include "model/processing/aliencellfunction.h"

#include <QGraphicsScene>
#include <QGLWidget>
#include <QTimer>
#include <QScrollBar>
#include <QSpinBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QFont>

MainWindow::MainWindow(AlienSimulator* simulator, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    _simulator(simulator),
    _microEditor(new MicroEditor(this)),
    _timer(0),
    _monitor(new SimulationMonitor(this)),
    _tutorialWindow(new TutorialWindow(this)),
    _startScreen(new StartScreenController(this)),
    _oldFrame(0),
    _frame(0),
    _frameSec(0)
{
    ui->setupUi(this);

    //init main objects
    _microEditor->init(ui->tabClusterWidget2,
                       ui->tabComputerWidget2,
                       ui->tabTokenWidget2,
                       ui->tabSymbolsWidget,
                       ui->cellEditor2,
                       ui->clusterEditor2,
                       ui->energyEditor2,
                       ui->metadataEditor2,
                       ui->cellComputerEdit,
                       ui->symbolEdit2,
                       ui->selectionEditor2,
                       ui->requestCellButton2,
                       ui->requestEnergyParticleButton2,
                       ui->delEntityButton2,
                       ui->delClusterButton2,
                       ui->addTokenButton2,
                       ui->delTokenButton2);

    //set font
    setFont(GlobalFunctions::getGlobalFont());
    ui->menuSimulation->setFont(GlobalFunctions::getGlobalFont());
    ui->menuView->setFont(GlobalFunctions::getGlobalFont());
    ui->menuEdit->setFont(GlobalFunctions::getGlobalFont());
    ui->menuSelection->setFont(GlobalFunctions::getGlobalFont());
    ui->menuMetadata->setFont(GlobalFunctions::getGlobalFont());
    ui->menuHelp->setFont(GlobalFunctions::getGlobalFont());
    ui->menuSimulationParameters->setFont(GlobalFunctions::getGlobalFont());
    ui->menuSymbolTable_2->setFont(GlobalFunctions::getGlobalFont());
    ui->menuAddEnsemble->setFont(GlobalFunctions::getGlobalFont());
    ui->menuMultiplyExtension->setFont(GlobalFunctions::getGlobalFont());

    //set color
    ui->fpsForcingButton->setStyleSheet(BUTTON_STYLESHEET);
    ui->toolBar->setStyleSheet("background-color: #303030");
    QPalette p = ui->fpsForcingButton->palette();
    p.setColor(QPalette::ButtonText, BUTTON_TEXT_COLOR);
    ui->fpsForcingButton->setPalette(p);


/*    menuBar()->setStyleSheet("QMenuBar::item {background-color: #606060; selection-color: #ffffff; selection-background-color: #606090; color: #DDDDDD; }"
                             "QMenuBar { background-color: #606060; selection-color: #ffffff; selection-background-color: #606090; color: #DDDDDD; }"
                             "QMenu { background-color: #606060; selection-color: #ffffff; selection-background-color: #606090; color: #DDDDDD;}"
                             );
*/
/*
    //layout
    QGridLayout* layout = new QGridLayout();
    layout->addWidget(ui->microEditor, 0, 0);
    delete ui->macroEditor->layout();
    ui->macroEditor->setLayout(layout);
*/
    //connect coordinators
    connect(_simulator, SIGNAL(cellCreated(AlienCell*)), ui->macroEditor, SLOT(cellCreated(AlienCell*)));
    connect(_simulator, SIGNAL(cellCreated(AlienCell*)), _microEditor, SLOT(cellFocused(AlienCell*)));
    connect(_simulator, SIGNAL(cellCreated(AlienCell*)), this, SLOT(cellFocused(AlienCell*)));
    connect(_simulator, SIGNAL(energyParticleCreated(AlienEnergy*)), ui->macroEditor, SLOT(energyParticleCreated(AlienEnergy*)));
    connect(_simulator, SIGNAL(energyParticleCreated(AlienEnergy*)), _microEditor, SLOT(energyParticleFocused(AlienEnergy*)));
    connect(_simulator, SIGNAL(energyParticleCreated(AlienEnergy*)), this, SLOT(energyParticleFocused(AlienEnergy*)));
    connect(_simulator, SIGNAL(universeUpdated(AlienGrid*, bool)), ui->macroEditor, SLOT(universeUpdated(AlienGrid*, bool)));
    connect(_simulator, SIGNAL(universeUpdated(AlienGrid*, bool)), _microEditor, SLOT(universeUpdated(AlienGrid*, bool)));
    connect(_simulator, SIGNAL(reclustered(QList<AlienCellCluster*>)), ui->macroEditor, SLOT(reclustered(QList<AlienCellCluster*>)));
    connect(_simulator, SIGNAL(reclustered(QList<AlienCellCluster*>)), _microEditor, SLOT(reclustered(QList<AlienCellCluster*>)));
    connect(_simulator, SIGNAL(computerCompilationReturn(bool,int)), _microEditor, SLOT(computerCompilationReturn(bool,int)));
    connect(ui->macroEditor, SIGNAL(requestNewCell(QVector3D)), _simulator, SLOT(newCell(QVector3D)));
    connect(ui->macroEditor, SIGNAL(requestNewEnergyParticle(QVector3D)), _simulator, SLOT(newEnergyParticle(QVector3D)));
    connect(ui->macroEditor, SIGNAL(defocus()), _microEditor, SLOT(defocused()));
    connect(ui->macroEditor, SIGNAL(defocus()), this, SLOT(cellDefocused()));
    connect(ui->macroEditor, SIGNAL(focusCell(AlienCell*)), _microEditor, SLOT(cellFocused(AlienCell*)));
    connect(ui->macroEditor, SIGNAL(focusCell(AlienCell*)), this, SLOT(cellFocused(AlienCell*)));
    connect(ui->macroEditor, SIGNAL(focusEnergyParticle(AlienEnergy*)), _microEditor, SLOT(energyParticleFocused(AlienEnergy*)));
    connect(ui->macroEditor, SIGNAL(updateCell(QList<AlienCell*>,QList<AlienCellTO>,bool)), _simulator, SLOT(updateCell(QList<AlienCell*>,QList<AlienCellTO>,bool)));
    connect(ui->macroEditor, SIGNAL(energyParticleUpdated(AlienEnergy*)), _microEditor, SLOT(energyParticleUpdated_Slot(AlienEnergy*)));
    connect(ui->macroEditor, SIGNAL(entitiesSelected(int,int)), _microEditor, SLOT(entitiesSelected(int,int)));
    connect(ui->macroEditor, SIGNAL(entitiesSelected(int,int)), this, SLOT(entitiesSelected(int,int)));
    connect(ui->macroEditor, SIGNAL(delSelection(QList<AlienCell*>,QList<AlienEnergy*>)), _simulator, SLOT(delSelection(QList<AlienCell*>,QList<AlienEnergy*>)));
    connect(ui->macroEditor, SIGNAL(delExtendedSelection(QList<AlienCellCluster*>,QList<AlienEnergy*>)), _simulator, SLOT(delExtendedSelection(QList<AlienCellCluster*>,QList<AlienEnergy*>)));
    connect(_microEditor, SIGNAL(requestNewCell()), ui->macroEditor, SLOT(newCellRequested()));
    connect(_microEditor, SIGNAL(requestNewEnergyParticle()), ui->macroEditor, SLOT(newEnergyParticleRequested()));
    connect(_microEditor, SIGNAL(updateCell(QList<AlienCell*>,QList<AlienCellTO>,bool)), _simulator, SLOT(updateCell(QList<AlienCell*>,QList<AlienCellTO>,bool)));
    connect(_microEditor, SIGNAL(energyParticleUpdated(AlienEnergy*)), ui->macroEditor, SLOT(energyParticleUpdated_Slot(AlienEnergy*)));
    connect(_microEditor, SIGNAL(delSelection()), ui->macroEditor, SLOT(delSelection_Slot()));
    connect(_microEditor, SIGNAL(delExtendedSelection()), ui->macroEditor, SLOT(delExtendedSelection_Slot()));
    connect(_microEditor, SIGNAL(defocus()), ui->macroEditor, SLOT(defocused()));
    connect(_microEditor, SIGNAL(defocus()), this, SLOT(cellDefocused()));
    connect(_microEditor, SIGNAL(metadataUpdated()), ui->macroEditor, SLOT(metadataUpdated()));
    connect(_microEditor, SIGNAL(numTokenUpdate(int,int,bool)), this, SLOT(numTokenChanged(int,int,bool)));

    //connect signals/slots for actions
    connect(ui->actionPlay, SIGNAL( triggered(bool) ), this, SLOT(runClicked(bool)));
    connect(ui->actionStep, SIGNAL( triggered(bool) ), this, SLOT(stepForwardClicked()));
    connect(ui->actionStepBack, SIGNAL( triggered(bool) ), this, SLOT(stepBackClicked()));
    connect(ui->actionEditor, SIGNAL(triggered(bool)), this, SLOT(setEditMode(bool)));
    connect(ui->actionZoomIn, SIGNAL(triggered(bool)), ui->macroEditor, SLOT(zoomIn()));
    connect(ui->actionZoomIn, SIGNAL(triggered(bool)), this, SLOT(updateFrameLabel()));
    connect(ui->actionZoomOut, SIGNAL(triggered(bool)), ui->macroEditor, SLOT(zoomOut()));
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
    connect(ui->actionNewToken, SIGNAL(triggered(bool)), _microEditor, SLOT(addTokenClicked()));
    connect(ui->actionDeleteToken, SIGNAL(triggered(bool)), _microEditor, SLOT(delTokenClicked()));
    connect(ui->actionCopyToken, SIGNAL(triggered(bool)), _microEditor, SLOT(copyTokenClicked()));
    connect(ui->actionPasteToken, SIGNAL(triggered(bool)), _microEditor, SLOT(pasteTokenClicked()));
    connect(ui->actionEditSimulationParameters, SIGNAL(triggered(bool)), this, SLOT(editSimulationParameters()));
    connect(ui->actionLoadSimulationParameters, SIGNAL(triggered(bool)), this, SLOT(loadSimulationParameters()));
    connect(ui->actionSaveSimulationParameters, SIGNAL(triggered(bool)), this, SLOT(saveSimulationParameters()));
    connect(ui->actionEditSymbols, SIGNAL(triggered(bool)), this, SLOT(editSymbolTable()));
    connect(ui->actionFullscreen, SIGNAL(triggered(bool)), this, SLOT(fullscreen(bool)));
    connect(ui->actionRandom, SIGNAL(triggered(bool)), this, SLOT(multiplyRandomExtendedSelection()));
    connect(ui->actionArrangement, SIGNAL(triggered(bool)), this, SLOT(multiplyArrangementExtendedSelection()));
    connect(ui->actionAboutAlien, SIGNAL(triggered(bool)), this, SLOT(aboutAlien()));
    connect(ui->actionDeleteCell, SIGNAL(triggered(bool)), _microEditor, SLOT(delSelectionClicked()));
    connect(ui->actionDeleteExtension, SIGNAL(triggered(bool)), _microEditor, SLOT(delExtendedSelectionClicked()));
    connect(ui->actionTutorial, SIGNAL(triggered(bool)), _tutorialWindow, SLOT(setVisible(bool)));
    connect(_tutorialWindow, SIGNAL(closed()), this, SLOT(tutorialClosed()));

    //connect simulation monitor
    connect(_monitor, SIGNAL(closed()), this, SLOT(alienMonitorClosed()));

    //connect fps widgets
    connect(_simulator, SIGNAL(calcNextTimestep()), this, SLOT( incFrame() ));
    connect(ui->fpsForcingButton, SIGNAL(toggled(bool)), this, SLOT(fpsForcingButtonClicked(bool)));
    connect(ui->fpsForcingSpinBox, SIGNAL(valueChanged(int)), this, SLOT(fpsForcingSpinboxClicked()));

    //setup simulator
    _simulator->updateUniverse();

    //setup micro editor
    _microEditor->setVisible(false);

    //init widgets
    QFont f = ui->frameLabel->font();
    f.setBold(false);
    f.setItalic(true);
    ui->frameLabel->setFont(f);

    //init frame counter
    incFrame();
    _timer = new QTimer(this);
    connect(_timer, SIGNAL(timeout()), this, SLOT(timeout()));
    _timer->start(1000);

    //connect and run start screen
//    connect(_startScreen, SIGNAL(startScreenFinished()), ui->actionEditor, SLOT(setEnabled(bool)));
    connect(_startScreen, SIGNAL(startScreenFinished()), SLOT(startScreenFinished()));
    _startScreen->runStartScreen(ui->macroEditor->getGraphicsView());
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::newSimulation ()
{
    NewSimulationDialog d;
    if( d.exec() ) {
        _frame = 0;

        //stop simulation
        ui->actionPlay->setChecked(false);
        runClicked(false);

        //create new simulation
        _simulator->newUniverse(d.getSizeX(), d.getSizeY());
        _simulator->addRandomEnergy (d.getEnergy(), simulationParameters.CRIT_CELL_TRANSFORM_ENERGY);

        //create energy

        //reset editors
        ui->macroEditor->reset();
        _microEditor->updateSymbolTable();

        //force simulator to update other coordinators
        _simulator->updateUniverse();

        //no step back option
        ui->actionStepBack->setEnabled(false);
        _undoUniverserses.clear();

        //update monitor
        _monitor->update(_simulator->getMonitorData());
    }
}

void MainWindow::loadSimulation ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Simulation", "", "Alien Simulation (*.sim)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {
            _frame = 0;

            //stop simulation
            ui->actionPlay->setChecked(false);
            runClicked(false);

            //read simulation data
            QDataStream in(&file);
            QMap< quint64, quint64 > oldNewCellIdMap;
            QMap< quint64, quint64 > oldNewClusterIdMap;
            _simulator->buildUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);
            simulationParameters.readData(in);
            MetadataManager::getGlobalInstance().readMetadataUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);
            MetadataManager::getGlobalInstance().readSymbolTable(in);
            readFrame(in);
            file.close();

            //reset editors
            ui->macroEditor->reset();
            _microEditor->updateSymbolTable();

            //force simulator to update other coordinators
            _simulator->updateUniverse();

            //no step back option
            ui->actionStepBack->setEnabled(false);
            _undoUniverserses.clear();

            //update monitor
            _monitor->update(_simulator->getMonitorData());
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified simulation could not loaded.");
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
            _simulator->serializeUniverse(out);
            simulationParameters.serializeData(out);
            MetadataManager::getGlobalInstance().serializeMetadataUniverse(out);
            MetadataManager::getGlobalInstance().serializeSymbolTable(out);
            out << _frame;
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The simulation could not saved.");
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
    _microEditor->requestUpdate();
    _simulator->setRun(run);
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
    _microEditor->requestUpdate();

    //save old universe
    QByteArray b;
    QDataStream out(&b, QIODevice::WriteOnly);
    _simulator->serializeUniverse(out);
    MetadataManager::getGlobalInstance().serializeMetadataUniverse(out);
    _undoUniverserses.push(b);

    //calc next time step
    _simulator->requestNextTimestep();
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
    QMap< quint64, quint64 > oldNewCellIdMap;
    QMap< quint64, quint64 > oldNewClusterIdMap;
    _simulator->buildUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);
    MetadataManager::getGlobalInstance().readMetadataUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);

    //reset coordinators
//    ui->macroEditor->reset();

    //force simulator to update other coordinators
    _simulator->updateUniverse();

    //no step back any more?
    if( _undoUniverserses.isEmpty() )
        ui->actionStepBack->setEnabled(false);

    //update monitor
    _monitor->update(_simulator->getMonitorData());
    decFrame();
}

void MainWindow::snapshotUniverse ()
{
    ui->actionRestore->setEnabled(true);

    //save old universe
    _snapshot.clear();
    QDataStream out(&_snapshot, QIODevice::WriteOnly);
    _simulator->serializeUniverse(out);
    MetadataManager::getGlobalInstance().serializeMetadataUniverse(out);
    ui->macroEditor->serializeViewMatrix(out);
    out << _frame;
}

void MainWindow::restoreUniverse ()
{
//    ui->actionRestore->setEnabled(false);

    //load old universe
    QDataStream in(&_snapshot, QIODevice::ReadOnly);

    //read simulation data
    QMap< quint64, quint64 > oldNewCellIdMap;
    QMap< quint64, quint64 > oldNewClusterIdMap;
    _simulator->buildUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);
    MetadataManager::getGlobalInstance().readMetadataUniverse(in, oldNewClusterIdMap, oldNewCellIdMap);
//    _snapshot.clear();

    //reset editors
    ui->macroEditor->reset();

    //load view
    ui->macroEditor->loadViewMatrix(in);

    //force simulator to update other coordinators
    _simulator->updateUniverse();

    //update monitor
    _monitor->update(_simulator->getMonitorData());

    //hide "step back" button
    _undoUniverserses.clear();
    ui->actionStepBack->setEnabled(false);

    //save frame
    readFrame(in);
}

void MainWindow::editSimulationParameters ()
{
    SimulationParametersDialog d;
    if( d.exec() ) {
        d.updateSimulationParameters();
    }
}

void MainWindow::loadSimulationParameters ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {

            //read simulation data
            QDataStream in(&file);
            simulationParameters.readData(in);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified simulation parameter file could not loaded.");
            msgBox.exec();
        }
    }
}

void MainWindow::saveSimulationParameters ()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {

            //serialize symbol table
            QDataStream out(&file);
            simulationParameters.serializeData(out);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The simulation parameters could not saved.");
            msgBox.exec();
        }
    }
}

void MainWindow::fullscreen (bool triggered)
{
    QMainWindow::setWindowState(QMainWindow::windowState() ^ Qt::WindowFullScreen);
}

void MainWindow::setEditMode (bool editMode)
{
    if( !editMode )
        _microEditor->requestUpdate();
    ui->actionPlay->setChecked(false);
    ui->actionStep->setEnabled(true);
    ui->actionPlay->setIcon(QIcon("://Icons/play.png"));

    //update menubar
//    ui->actionLoad_cell_extension->setEnabled(editMode);

    //update macro editor and button
    if( editMode) {
        ui->macroEditor->setActiveScene(MacroEditor::SHAPE_SCENE);
        ui->actionEditor->setIcon(QIcon("://Icons/microscope_active.png"));
    }
    else {
        ui->macroEditor->setActiveScene(MacroEditor::PIXEL_SCENE);
        ui->actionEditor->setIcon(QIcon("://Icons/microscope.png"));
        cellDefocused();
    }

    //update micro editor
    _microEditor->setVisible(editMode);

    //stop running
    _simulator->setRun(false);
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
    ui->macroEditor->newCellRequested();
    if( !_microEditor->isVisible() )
        _simulator->updateUniverse();
}

void MainWindow::addEnergyParticle ()
{
    ui->macroEditor->newEnergyParticleRequested();
    if( !_microEditor->isVisible() )
        _simulator->updateUniverse();
}

void MainWindow::addRandomEnergy ()
{
    AddEnergyDialog d;
    if( d.exec() ) {
        _simulator->addRandomEnergy(d.getTotalEnergy(), d.getMaxEnergyPerParticle());
        _simulator->updateUniverse();
    }
}

void MainWindow::copyCell ()
{
    //serialize cell
    AlienCell* focusCell = _microEditor->getFocusedCell();
    QDataStream out(&_serializedCellData, QIODevice::WriteOnly);
    quint64 clusterId;
    quint64 cellId;
    _simulator->serializeCell(out, focusCell, clusterId, cellId);
    MetadataManager::getGlobalInstance().serializeMetadataCell(out, clusterId, cellId);

    //set actions
    ui->actionPasteCell->setEnabled(true);
}

void MainWindow::pasteCell ()
{
    QDataStream in(&_serializedCellData, QIODevice::ReadOnly);
    QMap< quint64, quint64 > oldNewCellIdMap;
    QMap< quint64, quint64 > oldNewClusterIdMap;
    AlienCellCluster* newCluster;
    _simulator->buildCell(in, ui->macroEditor->getViewCenterPosWithInc(), newCluster, oldNewClusterIdMap, oldNewCellIdMap);
    MetadataManager::getGlobalInstance().readMetadata(in, oldNewClusterIdMap, oldNewCellIdMap);

    //force simulator to update other coordinators
    _simulator->updateUniverse();

    //update monitor
    _monitor->update(_simulator->getMonitorData());
}

void MainWindow::editSymbolTable ()
{
    SymbolTableDialog d;
    if( d.exec() ) {

        //update symbol table
        d.updateSymbolTable(&MetadataManager::getGlobalInstance());

        //update editor
        _microEditor->updateSymbolTable();
        _microEditor->updateTokenTab();
    }
}

void MainWindow::loadSymbols ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Symbol Table", "", "Alien Symbol Table(*.sym)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {

            //read simulation data
            QDataStream in(&file);
            MetadataManager::getGlobalInstance().readSymbolTable(in);
            file.close();

            //update editor
            _microEditor->updateSymbolTable();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified symbol table could not loaded.");
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

            //serialize symbol table
            QDataStream out(&file);
            MetadataManager::getGlobalInstance().serializeSymbolTable(out);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The symbol table could not saved.");
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

            //read simulation data
            QDataStream in(&file);
            MetadataManager::getGlobalInstance().readSymbolTable(in, true);
            file.close();

            //update editor
            _microEditor->updateSymbolTable();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified symbol table could not loaded.");
            msgBox.exec();
        }
    }
}

void MainWindow::addBlockStructure ()
{
    AddRectStructureDialog d;
    if( d.exec() ) {
        QVector3D center = ui->macroEditor->getViewCenterPosWithInc();
       _simulator->addBlockStructure(center,
                                     d.getBlockSizeX(), d.getBlockSizeY(),
                                     QVector3D(d.getDistance(), d.getDistance(), 0.0),
                                     d.getInternalEnergy());
    }
}

void MainWindow::addHexagonStructure ()
{
    AddHexagonStructureDialog d;
    if( d.exec() ) {
        QVector3D center = ui->macroEditor->getViewCenterPosWithInc();
       _simulator->addHexagonStructure(center,
                                       d.getLayers(),
                                       d.getDistance(),
                                       d.getInternalEnergy());
    }
}

void MainWindow::loadExtendedSelection ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Ensemble", "", "Alien Ensemble (*.ens)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {
            QDataStream in(&file);
            QMap< quint64, quint64 > oldNewCellIdMap;
            QMap< quint64, quint64 > oldNewClusterIdMap;
            QList< AlienCellCluster* > newClusters;
            QList< AlienEnergy* > newEnergyParticles;
            _simulator->buildExtendedSelection(in, ui->macroEditor->getViewCenterPosWithInc(), newClusters,  newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap);
            MetadataManager::getGlobalInstance().readMetadata(in, oldNewClusterIdMap, oldNewCellIdMap);
            file.close();

            //force simulator to update other coordinators
            _simulator->updateUniverse();

            //update monitor
            _monitor->update(_simulator->getMonitorData());
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified ensemble could not loaded.");
            msgBox.exec();
        }
    }
}

void MainWindow::saveExtendedSelection ()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Ensemble", "", "Alien Ensemble (*.ens)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {

            //get selected clusters and energy particles
            QList< AlienCellCluster* > clusters;
            QList< AlienEnergy* > es;
            ui->macroEditor->getExtendedSelection(clusters, es);

            //serialize lists
            QDataStream out(&file);
            QList< quint64 > clusterIds;
            QList< quint64 > cellIds;
            _simulator->serializeExtendedSelection(out, clusters, es, clusterIds, cellIds);
            MetadataManager::getGlobalInstance().serializeMetadataEnsemble(out, clusterIds, cellIds);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The selected ensemble could not saved.");
            msgBox.exec();
        }
    }
}

void MainWindow::copyExtendedSelection ()
{
    //get selected clusters and energy particles
    QList< AlienCellCluster* > clusters;
    QList< AlienEnergy* > es;
    ui->macroEditor->getExtendedSelection(clusters, es);

    //serialize lists
    QDataStream out(&_serializedEnsembleData, QIODevice::WriteOnly);
    QList< quint64 > clusterIds;
    QList< quint64 > cellIds;
    _simulator->serializeExtendedSelection(out, clusters, es, clusterIds, cellIds);
    MetadataManager::getGlobalInstance().serializeMetadataEnsemble(out, clusterIds, cellIds);

    //set actions
    ui->actionPaste_cell_extension->setEnabled(true);
}

void MainWindow::pasteExtendedSelection ()
{
    QDataStream in(&_serializedEnsembleData, QIODevice::ReadOnly);
    QMap< quint64, quint64 > oldNewCellIdMap;
    QMap< quint64, quint64 > oldNewClusterIdMap;
    QList< AlienCellCluster* > newClusters;
    QList< AlienEnergy* > newEnergyParticles;
    _simulator->buildExtendedSelection(in, ui->macroEditor->getViewCenterPosWithInc(), newClusters, newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap);
    MetadataManager::getGlobalInstance().readMetadata(in, oldNewClusterIdMap, oldNewCellIdMap);

    //force simulator to update other coordinators
    _simulator->updateUniverse();

    //update monitor
    _monitor->update(_simulator->getMonitorData());
}

void MainWindow::multiplyRandomExtendedSelection ()
{
    SelectionMultiplyRandomDialog d;
    if( d.exec() ) {

        //get selected clusters and energy particles
        QList< AlienCellCluster* > clusters;
        QList< AlienEnergy* > es;
        ui->macroEditor->getExtendedSelection(clusters, es);

        //serialize lists
        QByteArray serializedEnsembleData;
        QDataStream out(&serializedEnsembleData, QIODevice::WriteOnly);
        QList< quint64 > clusterIds;
        QList< quint64 > cellIds;
        _simulator->serializeExtendedSelection(out, clusters, es, clusterIds, cellIds);
        MetadataManager::getGlobalInstance().serializeMetadataEnsemble(out, clusterIds, cellIds);

        //read list and rebuild structure n times
        for(int i = 0; i < d.getNumber(); ++i) {
            QDataStream in(&serializedEnsembleData, QIODevice::ReadOnly);
            QMap< quint64, quint64 > oldNewCellIdMap;
            QMap< quint64, quint64 > oldNewClusterIdMap;
            QList< AlienCellCluster* > newClusters;
            QList< AlienEnergy* > newEnergyParticles;
            QVector3D pos(GlobalFunctions::random(0.0, _simulator->getUniverseSizeX()), GlobalFunctions::random(0.0, _simulator->getUniverseSizeY()), 0.0);
            _simulator->buildExtendedSelection(in, pos, newClusters, newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap, false);
            MetadataManager::getGlobalInstance().readMetadata(in, oldNewClusterIdMap, oldNewCellIdMap);

            //randomize angles and velocities if desired
            if( d.randomizeAngle() )
                _simulator->rotateExtendedSelection(GlobalFunctions::random(d.randomizeAngleMin(), d.randomizeAngleMax()), newClusters, newEnergyParticles);
            if( d.randomizeVelX() )
                _simulator->setVelocityXExtendedSelection(GlobalFunctions::random(d.randomizeVelXMin(), d.randomizeVelXMax()), newClusters, newEnergyParticles);
            if( d.randomizeVelY() )
                _simulator->setVelocityYExtendedSelection(GlobalFunctions::random(d.randomizeVelYMin(), d.randomizeVelYMax()), newClusters, newEnergyParticles);
            if( d.randomizeAngVel() )
                _simulator->setAngularVelocityExtendedSelection(GlobalFunctions::random(d.randomizeAngVelMin(), d.randomizeAngVelMax()), newClusters);

            //draw selection
            _simulator->drawToMapExtendedSelection(newClusters, newEnergyParticles);
        }

        //force simulator to update other coordinators
        _simulator->updateUniverse();

        //update monitor
        _monitor->update(_simulator->getMonitorData());

    }
}

void MainWindow::multiplyArrangementExtendedSelection ()
{
    //get selected clusters and energy particles
    QList< AlienCellCluster* > clusters;
    QList< AlienEnergy* > es;
    ui->macroEditor->getExtendedSelection(clusters, es);

    //celc center
    QVector3D centerPos = _simulator->getCenterPosExtendedSelection(clusters, es);
    SelectionMultiplyArrangementDialog d(centerPos);
    if( d.exec() ) {

        //serialize lists
        QByteArray serializedEnsembleData;
        QDataStream out(&serializedEnsembleData, QIODevice::WriteOnly);
        QList< quint64 > clusterIds;
        QList< quint64 > cellIds;
        _simulator->serializeExtendedSelection(out, clusters, es, clusterIds, cellIds);
        MetadataManager::getGlobalInstance().serializeMetadataEnsemble(out, clusterIds, cellIds);

        //read list and rebuild structure n x m times
        for(int i = 0; i < d.getHorizontalNumber(); ++i) {
            for(int j = 0; j < d.getVerticalNumber(); ++j) {
                QDataStream in(&serializedEnsembleData, QIODevice::ReadOnly);
                QMap< quint64, quint64 > oldNewCellIdMap;
                QMap< quint64, quint64 > oldNewClusterIdMap;
                QList< AlienCellCluster* > newClusters;
                QList< AlienEnergy* > newEnergyParticles;
                QVector3D pos(d.getInitialPosX() + (qreal)i*d.getHorizontalInterval(),
                              d.getInitialPosY() + (qreal)j*d.getVerticalInterval(), 0.0);
                _simulator->buildExtendedSelection(in, pos, newClusters, newEnergyParticles, oldNewClusterIdMap, oldNewCellIdMap, false);
                MetadataManager::getGlobalInstance().readMetadata(in, oldNewClusterIdMap, oldNewCellIdMap);

                //set angles and velocities
                if( d.changeAngle() ) {
                    qreal angle = d.getInitialAngle()+(qreal)i*d.getHorizontalAngleIncrement()+(qreal)j*d.getVerticalAngleIncrement();
                    _simulator->rotateExtendedSelection(angle, newClusters, newEnergyParticles);
                }
                if( d.changeVelocityX() ) {
                    qreal velX = d.getInitialVelX()+(qreal)i*d.getHorizontalVelocityXIncrement()+(qreal)j*d.getVerticalVelocityXIncrement();
                    _simulator->setVelocityXExtendedSelection(velX, newClusters, newEnergyParticles);
                }
                if( d.changeVelocityY() ) {
                    qreal velY = d.getInitialVelY()+(qreal)j*d.getHorizontalVelocityYIncrement()+(qreal)j*d.getVerticalVelocityYIncrement();
                    _simulator->setVelocityYExtendedSelection(velY, newClusters, newEnergyParticles);
                }
                if( d.changeAngularVelocity() ) {
                    qreal angVel = d.getInitialAngVel()+(qreal)i*d.getHorizontalAngularVelocityIncrement()+(qreal)j*d.getVerticalAngularVelocityIncrement();
                    _simulator->setAngularVelocityExtendedSelection(angVel, newClusters);
                }

                //draw selection
                _simulator->drawToMapExtendedSelection(newClusters, newEnergyParticles);
            }
        }

        //delete original cluster
        _simulator->delExtendedSelection(clusters, es);

        //force simulator to update other coordinators
        _simulator->updateUniverse();

        //update monitor
        _monitor->update(_simulator->getMonitorData());
    }
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

void MainWindow::timeout ()
{
    //update fps
    _frameSec = _frame - _oldFrame;
    _oldFrame = _frame;
    updateFrameLabel();

    //update monitor
    _monitor->update(_simulator->getMonitorData());
}

void MainWindow::fpsForcingButtonClicked (bool toggled)
{
    if( toggled ) {
        _simulator->forceFps(ui->fpsForcingSpinBox->value());
        QPalette p = ui->fpsForcingButton->palette();
        p.setColor(QPalette::ButtonText, BUTTON_TEXT_HIGHLIGHT_COLOR);
        ui->fpsForcingButton->setPalette(p);
    }
    else {
        _simulator->forceFps(0);
        QPalette p = ui->fpsForcingButton->palette();
        p.setColor(QPalette::ButtonText, BUTTON_TEXT_COLOR);
        ui->fpsForcingButton->setPalette(p);
    }
}

void MainWindow::fpsForcingSpinboxClicked ()
{
    if( ui->fpsForcingButton->isChecked() )
        _simulator->forceFps(ui->fpsForcingSpinBox->value());
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

void MainWindow::incFrame ()
{
    ++_frame;
    updateFrameLabel();
}

void MainWindow::decFrame ()
{
    --_frame;
    _oldFrame = _frame;
    updateFrameLabel();
}

void MainWindow::readFrame (QDataStream& stream)
{
    stream >> _frame;
    _oldFrame = _frame;
    updateFrameLabel();
}

void MainWindow::cellFocused (AlienCell* cell)
{
    if( _microEditor->isVisible() ) {
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

void MainWindow::energyParticleFocused (AlienEnergy* e)
{
    if( _microEditor->isVisible() ) {
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
    ui->frameLabel->setText(QString("Frame: %1  FPS: %2  Magnification: %3x").arg(_frame, 9, 10, QLatin1Char('0')).arg(_frameSec, 5, 10, QLatin1Char('0')).arg(ui->macroEditor->getZoomFactor()));
}

void MainWindow::startScreenFinished ()
{
    ui->actionEditor->setEnabled(true);
    ui->actionZoomIn->setEnabled(true);
    ui->actionZoomOut->setEnabled(true);
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


