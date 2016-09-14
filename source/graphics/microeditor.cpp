#include "microeditor.h"
//#include "ui_microeditor.h"

#include "microeditor/tokentab.h"
#include "microeditor/celledit.h"
#include "microeditor/clusteredit.h"
#include "microeditor/energyedit.h"
#include "microeditor/hexedit.h"
#include "microeditor/metadataedit.h"
#include "microeditor/symboledit.h"
//#include "microeditor/computercodeedit.h"
#include "microeditor/cellcomputeredit.h"
#include "../globaldata/editorsettings.h"
#include "../globaldata/guisettings.h"
#include "../globaldata/simulationparameters.h"
#include "../simulation/entities/aliencell.h"
#include "../simulation/entities/aliencellcluster.h"
#include "../simulation/entities/aliengrid.h"
#include "../simulation/processing/aliencellfunction.h"
#include "../simulation/processing/aliencellfunctionfactory.h"

#include <QTabWidget>
#include <QToolButton>
#include <QLabel>
#include <QTextEdit>
#include <QEvent>

const int tabPosX1 = 410;
const int tabPosX2 = 810;

MicroEditor::MicroEditor(QObject *parent) :
    QObject(parent),
//    ui(new Ui::MicroEditor),
    _focusCell(0),
    _focusEnergyParticle(0),
    _currentClusterTab(0),
    _currentTokenTab(0),
    _pasteTokenPossible(false),
    _savedTokenEnergy(0)
{
//    ui->setupUi(this);

}

MicroEditor::~MicroEditor()
{
//    delete ui;
}

void MicroEditor::init (QTabWidget* tabClusterWidget,
                        QTabWidget* tabComputerWidget,
                        QTabWidget* tabTokenWidget,
                        QTabWidget* tabSymbolsWidget,
                        CellEdit* cellEditor,
                        ClusterEdit* clusterEditor,
                        EnergyEdit* energyEditor,
//                        HexEdit* computerMemoryEditor,
                        MetaDataEdit* metaDataEditor,
                        CellComputerEdit* cellComputerEdit,
                        SymbolEdit* symbolEdit,
//                        ComputerCodeEdit* computerCodeEditor,
                        QTextEdit* selectionEditor,
//                        QLabel* compilationStateLabel,
                        QToolButton* requestCellButton,
                        QToolButton* requestEnergyParticleButton,
                        QToolButton* delEntityButton,
                        QToolButton* delClusterButton,
                        QToolButton* addTokenButton,
                        QToolButton* delTokenButton,
//                        QToolButton* compileButton,
                        MetaDataManager* meta)
{
    _tabClusterWidget = tabClusterWidget;
    _tabComputerWidget = tabComputerWidget;
    _tabTokenWidget = tabTokenWidget;
    _tabSymbolsWidget = tabSymbolsWidget;
    _cellEditor = cellEditor;
    _clusterEditor = clusterEditor;
    _energyEditor = energyEditor;
//    _computerMemoryEditor = computerMemoryEditor;
    _metaDataEditor = metaDataEditor;
    _cellComputerEdit = cellComputerEdit;
    _symbolEdit = symbolEdit;
//    _computerCodeEditor = computerCodeEditor;
    _selectionEditor = selectionEditor;
//    _compilationStateLabel = compilationStateLabel;
    _requestCellButton = requestCellButton;
    _requestEnergyParticleButton = requestEnergyParticleButton;
    _delEntityButton = delEntityButton;
    _delClusterButton = delClusterButton;
    _addTokenButton = addTokenButton;
    _delTokenButton = delTokenButton;
//    _compileButton = compileButton;
    _meta = meta;

    //save tab widgets
    _tabCluster = _tabClusterWidget->widget(0);
    _tabCell = _tabClusterWidget->widget(1);
    _tabParticle = _tabClusterWidget->widget(2);
    _tabSelection = _tabClusterWidget->widget(3);
    _tabMeta = _tabClusterWidget->widget(4);
    _tabComputer = _tabComputerWidget->widget(0);
    _tabSymbolTable = _tabComputerWidget->widget(1);

    //hide widgets
    _tabClusterWidget->setVisible(false);
    _tabComputerWidget->setVisible(false);
    _tabTokenWidget->setVisible(false);

    //set colors
    _requestCellButton->setStyleSheet(BUTTON_STYLESHEET);
    _requestEnergyParticleButton->setStyleSheet(BUTTON_STYLESHEET);
    _delEntityButton->setStyleSheet(BUTTON_STYLESHEET);
    _delClusterButton->setStyleSheet(BUTTON_STYLESHEET);
    _addTokenButton->setStyleSheet(BUTTON_STYLESHEET);
    _delTokenButton->setStyleSheet(BUTTON_STYLESHEET);

    //install event filter for parent widget
    _tabSymbolsWidget->parent()->installEventFilter(this);

    //establish connections
    connect(_cellEditor, SIGNAL(cellDataChanged(AlienCellReduced)), this, SLOT(changesFromCellEditor(AlienCellReduced)));
    connect(_clusterEditor, SIGNAL(clusterDataChanged(AlienCellReduced)), this, SLOT(changesFromClusterEditor(AlienCellReduced)));
    connect(_energyEditor, SIGNAL(energyParticleDataChanged(QVector3D,QVector3D,qreal)), this, SLOT(changesFromEnergyParticleEditor(QVector3D,QVector3D,qreal)));
    connect(_requestCellButton, SIGNAL(clicked()), this, SIGNAL(requestNewCell()));
    connect(_requestEnergyParticleButton, SIGNAL(clicked()), this, SIGNAL(requestNewEnergyParticle()));
    connect(_delEntityButton, SIGNAL(clicked()), this, SLOT(delSelectionClicked()));
    connect(_delClusterButton, SIGNAL(clicked()), this, SLOT(delExtendedSelectionClicked()));
    connect(_cellComputerEdit, SIGNAL(changesFromComputerMemoryEditor(QVector< quint8 >)), this, SLOT(changesFromComputerMemoryEditor(QVector< quint8 >)));
    connect(_cellComputerEdit, SIGNAL(compileButtonClicked(QString)), this, SLOT(compileButtonClicked(QString)));
    connect(_addTokenButton, SIGNAL(clicked()), this, SLOT(addTokenClicked()));
    connect(_delTokenButton, SIGNAL(clicked()), this, SLOT(delTokenClicked()));
    connect(_tabTokenWidget, SIGNAL(currentChanged(int)), this, SLOT(tokenTabChanged(int)));
    connect(_metaDataEditor, SIGNAL(metaDataChanged(QString,QString,quint8,QString)), this, SLOT(changesFromMetaDataEditor(QString,QString,quint8,QString)));
    connect(_symbolEdit, SIGNAL(symbolTableChanged()), this, SLOT(changesFromSymbolTableEditor()));

    _symbolEdit->loadSymbols(meta);
}


void MicroEditor::updateSymbolTable ()
{
    _symbolEdit->loadSymbols(_meta);
}

void MicroEditor::updateTokenTab ()
{
    changesFromSymbolTableEditor();
}

void MicroEditor::setVisible (bool visible)
{
    _requestCellButton->setVisible(visible);
    _requestEnergyParticleButton->setVisible(visible);
    _delEntityButton->setVisible(visible);
    _delClusterButton->setVisible(visible);
    _addTokenButton->setVisible(visible);
    _delTokenButton->setVisible(visible);
}

bool MicroEditor::isVisible ()
{
    return _requestCellButton->isVisible();
}

bool MicroEditor::eventFilter (QObject * watched, QEvent * event)
{
/*    if( (watched == _tabSymbolsWidget->parent()) && (event->type() == QEvent::Resize) ) {
        setTabSymbolsWidgetVisibility();
    }*/
    return QObject::eventFilter(watched, event);
}

AlienCell* MicroEditor::getFocusedCell ()
{
    return _focusCell;
}

void MicroEditor::computerCompilationReturn (bool error, int line)
{
    _cellComputerEdit->setCompilationState(error, line);
}


void MicroEditor::defocused (bool requestDataUpdate)
{
    disconnect(_tabClusterWidget, SIGNAL(currentChanged(int)), 0, 0);

    if( requestDataUpdate )
        requestUpdate();

    //close tabs
    if( _focusCell )
        _currentClusterTab = _tabClusterWidget->currentIndex();
    if( _tabClusterWidget->count() > 0 ) {
        while( _tabClusterWidget->count() > 0 )
            _tabClusterWidget->removeTab(0);
        _tabClusterWidget->setVisible(false);
    }
    if( _tabComputerWidget->count() > 0 ) {
        while( _tabComputerWidget->count() > 0)
            _tabComputerWidget->removeTab(0);
        _tabComputerWidget->setVisible(false);
    }
    if( _tabTokenWidget->count() > 0 ) {
        disconnect(_tabTokenWidget, SIGNAL(currentChanged(int)), 0, 0);  //token widgets will be generated dynamically
        while( _tabTokenWidget->count() > 0 ) {
            delete _tabTokenWidget->widget(0);
            _tabTokenWidget->removeTab(0);
        }
        connect(_tabTokenWidget, SIGNAL(currentChanged(int)), this, SLOT(tokenTabChanged(int)));
        _tabTokenWidget->setVisible(false);
    }
    setTabSymbolsWidgetVisibility();
    _currentTokenTab = 0;
    emit numTokenUpdate(0, 0, _pasteTokenPossible);

    //deactivate buttons
    if( _addTokenButton->isEnabled() )
        _addTokenButton->setEnabled(false);
    if( _delTokenButton->isEnabled() )
        _delTokenButton->setEnabled(false);
    if( _delEntityButton->isEnabled() )
        _delEntityButton->setEnabled(false);
    if( _delClusterButton->isEnabled() )
        _delClusterButton->setEnabled(false);

    _focusCell = 0;
    _focusEnergyParticle = 0;
}

void MicroEditor::cellFocused (AlienCell* cell, bool requestDataUpdate)
{
    if( (!isVisible()) || (!_grid) || (!cell) )
        return;

    defocused(requestDataUpdate);

    _focusCell = cell;

    //update data for cluster editor
    _grid->lockData();
    _focusCellReduced = AlienCellReduced(cell);
    quint64 id = cell->getId();
    QList< quint64 > ids = cell->getCluster()->getCellIds();
    quint8 color = cell->getColor();
    _grid->unlockData();
    _clusterEditor->updateCluster(_focusCellReduced);
    _cellEditor->updateCell(_focusCellReduced);
    _metaDataEditor->updateMetaData(_meta->getAndUniteClusterName(ids), _meta->getCellName(id), color, _meta->getCellDescription(id));

    //update data for cell function: computer
    if( _focusCellReduced.cellFunctionName == "COMPUTER" ) {

        //activate tab for computer widgets
//        _tabTokenWidget->move(10, tabPosY2);
        _tabComputerWidget->setVisible(true);
        _tabComputerWidget->insertTab(0, _tabComputer, "cell computer");
        _tabComputerWidget->insertTab(1, _tabSymbolTable, "symbol table");

        //update computer widgets
        _cellComputerEdit->updateComputerMemory(_focusCellReduced.computerMemory);

        //load computer code from meta data if available
        QString code = _meta->getCellCode(id);
        if( !code.isEmpty() ) {
            _cellComputerEdit->updateComputerCode(code);
        }

        //otherwise use translated cell data
        else
            _cellComputerEdit->updateComputerCode(_focusCellReduced.computerCode);
    }
//    else
//        _tabTokenWidget->move(10, tabPosY1);

    //activate cell cluster tabs
    _tabClusterWidget->setVisible(true);
    _tabClusterWidget->insertTab(0, _tabCell, "cell");
    _tabClusterWidget->insertTab(1, _tabCluster, "cluster");
    _tabClusterWidget->insertTab(2, _tabMeta, "meta data");
    _tabClusterWidget->setCurrentIndex(_currentClusterTab);
    connect(_tabClusterWidget, SIGNAL(currentChanged(int)), this, SLOT(clusterTabChanged(int)));


    //generate tab for each token
    int numToken = _focusCellReduced.tokenEnergies.size();
    if( numToken > 0 ) {
        _tabTokenWidget->setVisible(true);
        disconnect(_tabTokenWidget, SIGNAL(currentChanged(int)), 0, 0);
        for(int i = 0; i < numToken; ++i) {
            TokenTab* tokenTab = new TokenTab(_tabTokenWidget);
            connect(tokenTab, SIGNAL(tokenMemoryChanged(QVector< quint8 >)), this, SLOT(changesFromTokenMemoryEditor(QVector< quint8 >)));
            connect(tokenTab, SIGNAL(tokenPropChanged(qreal)), this, SLOT(changesFromTokenEditor(qreal)));
            _tabTokenWidget->addTab(tokenTab, QString("token %1").arg(i+1));
            tokenTab->update(_focusCellReduced.tokenEnergies[i], _focusCellReduced.tokenData[i], _meta);
        }
        connect(_tabTokenWidget, SIGNAL(currentChanged(int)), this, SLOT(tokenTabChanged(int)));

    }
    emit numTokenUpdate(numToken, simulationParameters.CELL_TOKENSTACKSIZE, _pasteTokenPossible);

    //update Symbols Widget
    setTabSymbolsWidgetVisibility();

    //update buttons
    _delEntityButton->setEnabled(true);
    _delClusterButton->setEnabled(true);
    if( numToken < simulationParameters.CELL_TOKENSTACKSIZE)
        _addTokenButton->setEnabled(true);
    else
        _addTokenButton->setEnabled(false);
    if( numToken > 0 )
        _delTokenButton->setEnabled(true);
    else
        _delTokenButton->setEnabled(false);
}

void MicroEditor::energyParticleFocused (AlienEnergy* e)
{
    if( (!isVisible()) || (!_grid) || (!e) )
        return;

    defocused();

    //activate tab
    _tabClusterWidget->setVisible(true);
    _tabClusterWidget->insertTab(0, _tabParticle, "energy particle");

    //activate widgets
    _delEntityButton->setEnabled(true);
    _delClusterButton->setEnabled(true);

    _focusEnergyParticle = e;
    energyParticleUpdated_Slot(e);
}

void MicroEditor::energyParticleUpdated_Slot (AlienEnergy* e)
{
    if( !e )
        return;

    //update data for editor if particle is focused (we also use cluster editor)
    if( _focusEnergyParticle == e ) {
        _grid->lockData();
        QVector3D pos = e->pos;
        QVector3D vel = e->vel;
        qreal energyValue = e->amount;
        _grid->unlockData();
        _energyEditor->updateEnergyParticle(pos, vel, energyValue);
    }
}


void MicroEditor::reclustered (QList< AlienCellCluster* > clusters)
{
    if( !_grid )
        return;
    if( _focusCell ) {

        //_focusCell contained in clusters?
        _grid->lockData();
        bool contained = false;
        foreach(AlienCellCluster* cluster, clusters)
            if( cluster->getCells().contains(_focusCell) )
                contained = true;
        _grid->unlockData();

        //proceed only if _focusCell is contained in clusters
        if( contained ) {

            //update data for cluster editor
            _grid->lockData();
            _focusCellReduced = AlienCellReduced(_focusCell);
            quint64 id = _focusCell->getId();
            QList< quint64 > ids = _focusCell->getCluster()->getCellIds();
            quint8 color = _focusCell->getColor();
            _grid->unlockData();
            _clusterEditor->updateCluster(_focusCellReduced);
            _cellEditor->updateCell(_focusCellReduced);
            _metaDataEditor->updateMetaData(_meta->getAndUniteClusterName(ids), _meta->getCellName(id), color, _meta->getCellDescription(id));

            //update computer code editor
            if( _focusCellReduced.cellFunctionName == "COMPUTER" ) {
                //_computerCodeEditor->update(_focusCellReduced.computerCode);
            }
        }
    }
}

void MicroEditor::universeUpdated (AlienGrid* grid, bool force)
{
    _grid = grid;
    defocused(false);
}

void MicroEditor::requestUpdate ()
{
    //save cell data
    if( _focusCell ) {

         //save edited code from code editor
        if( _focusCellReduced.cellFunctionName == "COMPUTER" ) {
            _grid->lockData();
            quint64 id = _focusCell->getId();
            _grid->unlockData();
            QString code = _cellComputerEdit->getComputerCode();
            _meta->setCellCode(id, code);
        }

        //save edited code from cluster editor
        if( _currentClusterTab == 0 )
            _cellEditor->requestUpdate();
        if( _currentClusterTab == 1 )
            _clusterEditor->requestUpdate();
        if( _currentClusterTab == 2 )
            _metaDataEditor->requestUpdate();

        //save token data => see
        if( _tabTokenWidget->count() > 0 ) {
            TokenTab* tab = (TokenTab*)_tabTokenWidget->currentWidget();
            tab->requestUpdate();
        }
    }

    //save energy particle data
    if( _focusEnergyParticle ) {
        _energyEditor->requestUpdate();
    }

}

void MicroEditor::entitiesSelected (int numCells, int numEnergyParticles)
{
    if( (numCells > 0) || (numEnergyParticles > 0) ){
        _delEntityButton->setEnabled(true);
        _delClusterButton->setEnabled(true);
    }
    else {
        _delEntityButton->setEnabled(false);
        _delClusterButton->setEnabled(false);
    }

    //active tab if not active
    if( _tabClusterWidget->currentWidget() != _tabSelection) {
        _tabClusterWidget->setVisible(true);
        while( _tabClusterWidget->count() > 0 )
            _tabClusterWidget->removeTab(0);
        _tabClusterWidget->insertTab(0, _tabSelection, "selection");
    }

    //update selection tab
    //define auxilliary strings
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorTextStart = "<span style=\"color:"+CELL_EDIT_TEXT_COLOR1.name()+"\">";
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorEnd = "</span>";
    QString text;
    text = parStart+colorTextStart+ "cells: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd
           + colorDataStart + QString("%1").arg(numCells)+colorEnd+parEnd;
    text += parStart+colorTextStart+ "energy particles: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd
           + colorDataStart + QString("%1").arg(numEnergyParticles)+colorEnd+parEnd;
    _selectionEditor->setText(text);
}

void MicroEditor::addTokenClicked ()
{
    //create token (new token is the last token on the stack)
    int newTokenTab = _currentTokenTab+1;
    _focusCellReduced.tokenEnergies.insert(newTokenTab, simulationParameters.NEW_TOKEN_ENERGY);
    QVector< quint8 > data(simulationParameters.TOKEN_MEMSIZE, 0);
    data[0] = _focusCellReduced.cellTokenAccessNum; //set access number for new token
    _focusCellReduced.tokenData.insert(newTokenTab, data);

    //emit signal to notify other instances => update _focusCell from _focusCellReduced
    invokeUpdateCell(false);

    cellFocused(_focusCell);
    _tabTokenWidget->setCurrentIndex(newTokenTab);

    //activate Symbols Widget
    setTabSymbolsWidgetVisibility();

/*
    //create token (new token is the last token on the stack)
    _focusCellReduced.tokenEnergies << NEW_TOKEN_ENERGY;
    QVector< quint8 > data(TOKEN_MEMSIZE, 0);
    data[0] = _focusCellReduced.cellTokenAccessNum; //set access number for new token
    _focusCellReduced.tokenData << data;

    //add token tab
    int numToken = _focusCellReduced.tokenEnergies.size();
    TokenTab* tokenTab = new TokenTab(_tabTokenWidget);
    _tabTokenWidget->setVisible(true);
    connect(tokenTab, SIGNAL(tokenMemoryChanged(QString)), this, SLOT(changesFromTokenMemoryEditor(QString)));
    connect(tokenTab, SIGNAL(tokenPropChanged(qreal)), this, SLOT(changesFromTokenEditor(qreal)));
    disconnect(_tabTokenWidget, SIGNAL(currentChanged(int)), 0, 0);
    _tabTokenWidget->addTab(tokenTab, QString("token %1").arg(numToken));
    connect(_tabTokenWidget, SIGNAL(currentChanged(int)), this, SLOT(tokenTabChanged(int)));
    tokenTab->update(_focusCellReduced.tokenEnergies[numToken-1], _focusCellReduced.tokenData[numToken-1]);
    _tabTokenWidget->setCurrentIndex(numToken-1);

    //update widgets
    ui->delTokenButton->setEnabled(true);
    if( numToken == CELL_TOKENSTACKSIZE )
        ui->addTokenButton->setEnabled(false);
    emit numTokenUpdate(numToken, CELL_TOKENSTACKSIZE, _pasteTokenPossible);

    //emit signal to notify other instances
    invokeUpdateCell(false);*/
}

void MicroEditor::delTokenClicked ()
{
    //remove token
    _focusCellReduced.tokenEnergies.removeAt(_tabTokenWidget->currentIndex());
    _focusCellReduced.tokenData.removeAt(_tabTokenWidget->currentIndex());

    //emit signal to notify other instances => update _focusCell from _focusCellReduced
    invokeUpdateCell(false);

    int newTokenTab = _currentTokenTab;
    int numToken = _focusCellReduced.tokenEnergies.size();
    if( (newTokenTab > 0) && (newTokenTab == numToken) )
        newTokenTab--;
    cellFocused(_focusCell, false);
    _tabTokenWidget->setCurrentIndex(newTokenTab);

    //update sybols widget
    setTabSymbolsWidgetVisibility();


/*
    //remove token
    _focusCellReduced.tokenEnergies.removeAt(_tabTokenWidget->currentIndex());
    _focusCellReduced.tokenData.removeAt(_tabTokenWidget->currentIndex());
    int numToken = _focusCellReduced.tokenEnergies.size();

    //remove token widget
    disconnect(_tabTokenWidget, SIGNAL(currentChanged(int)), 0, 0);
    delete _tabTokenWidget->currentWidget();
    connect(_tabTokenWidget, SIGNAL(currentChanged(int)), this, SLOT(tokenTabChanged(int)));

    //decrement token number on next tabs
    for(int i = _tabTokenWidget->currentIndex(); i < numToken; ++i)
        _tabTokenWidget->setTabText(i, QString("token %1").arg(i+1));

    //update button and tab widget
    ui->addTokenButton->setEnabled(true);
    if( numToken == 0 ) {
        ui->delTokenButton->setEnabled(false);
        _tabTokenWidget->setVisible(false);
    }
    emit numTokenUpdate(numToken, CELL_TOKENSTACKSIZE, _pasteTokenPossible);

    //emit signal to notify other instances
    invokeUpdateCell(false);*/
}

void MicroEditor::copyTokenClicked ()
{
    requestUpdate();
    _savedTokenEnergy = _focusCellReduced.tokenEnergies[_currentTokenTab];
    _savedTokenData = _focusCellReduced.tokenData[_currentTokenTab];
    _pasteTokenPossible = true;
    int numToken = _focusCellReduced.tokenEnergies.size();
    emit numTokenUpdate(numToken, simulationParameters.CELL_TOKENSTACKSIZE, _pasteTokenPossible);
}

void MicroEditor::pasteTokenClicked ()
{
    //create token (new token is the next to current token on the stack)
    int newTokenTab = _currentTokenTab+1;
    _focusCellReduced.tokenEnergies.insert(newTokenTab, _savedTokenEnergy);
    _savedTokenData[0] = _focusCellReduced.cellTokenAccessNum; //set access number for new token
    _focusCellReduced.tokenData.insert(newTokenTab, _savedTokenData);

    //emit signal to notify other instances => update _focusCell from _focusCellReduced
    invokeUpdateCell(false);

    cellFocused(_focusCell);
    _tabTokenWidget->setCurrentIndex(newTokenTab);

    //update Symbols Widget
    setTabSymbolsWidgetVisibility();
}

void MicroEditor::delSelectionClicked ()
{
    if( !_grid )
        return;

    //defocus
    defocused(false);

    //request deletion
    emit delSelection();
}

void MicroEditor::delExtendedSelectionClicked ()
{
    if( !_grid )
        return;

    //defocus
    defocused(false);

    //request deletion
    emit delExtendedSelection();
}

/*
void MicroEditor::mousePressEvent(QMouseEvent * event)
{
    QGraphicsView* m = parentWidget()->findChild<QGraphicsView*>("simulationView");
    if( m ) {
        QObjectList l = m->children();
        foreach(QObject* o, l) {
            qDebug() << o->objectName();
        }
//        QCoreApplication::sendEvent(m, event);
        m->setFocus(Qt::MouseFocusReason);
        QMouseEvent e(QEvent::MouseButtonPress, event->pos(), event->button(), event->buttons(), event->modifiers());
        qApp->sendEvent(m, &e);
        qDebug() << m->objectName() << event->pos().y();
    }
    return;
}
*/
void MicroEditor::changesFromCellEditor (AlienCellReduced newCellProperties)
{
    //copy cell properties editable by cluster editor
    _focusCellReduced.copyCellProperties(newCellProperties);

    //close tabs
    if( _tabComputerWidget->count() > 0 ) {
        while( _tabComputerWidget->count() > 0)
            _tabComputerWidget->removeTab(0);
        _tabComputerWidget->setVisible(false);
    }

    //update data for cell function: computer
    if( _focusCellReduced.cellFunctionName == "COMPUTER" ) {

        //activate tab for computer widgets
//        _tabTokenWidget->move(10, tabPosY2);
        _tabComputerWidget->setVisible(true);
        _tabComputerWidget->insertTab(0, _tabComputer, "cell computer");
        _tabComputerWidget->insertTab(1, _tabSymbolTable, "symbol table");
        _cellComputerEdit->updateComputerMemory(_focusCellReduced.computerMemory);

        //load computer code from meta data if available
        _grid->lockData();
        quint64 id = _focusCell->getId();
        _grid->unlockData();
        QString code = _meta->getCellCode(id);
        if( !code.isEmpty() ) {
            _cellComputerEdit->updateComputerCode(code);
        }

        //otherwise use translated cell data
        else
            _cellComputerEdit->updateComputerCode(_focusCellReduced.computerCode);
    }
//    else
//        _tabTokenWidget->move(10, tabPosY1);

    //update Symbols Widget
    setTabSymbolsWidgetVisibility();

    //emit signal to notify other instances
    invokeUpdateCell(false);

}

void MicroEditor::changesFromClusterEditor (AlienCellReduced newClusterProperties)
{
    //copy cell properties editable by cluster editor
    _focusCellReduced.copyClusterProperties(newClusterProperties);

    //emit signal to notify other instances
    invokeUpdateCell(true);
}

void MicroEditor::changesFromEnergyParticleEditor (QVector3D pos, QVector3D vel, qreal energyValue)
{
    if( (!_grid) || (!_focusEnergyParticle) )
        return;

    //update energy particle (we do this without informing the simulator...)
    _grid->lockData();
    _grid->setEnergy(_focusEnergyParticle->pos, 0);
    _focusEnergyParticle->pos = pos;
    _focusEnergyParticle->vel = vel;
    _focusEnergyParticle->amount = energyValue;
    _grid->setEnergy(_focusEnergyParticle->pos, _focusEnergyParticle);
    _grid->unlockData();

    //emit signal to notify other instances
    emit energyParticleUpdated(_focusEnergyParticle);
}

void MicroEditor::changesFromTokenEditor (qreal energy)
{
    _focusCellReduced.tokenEnergies[_currentTokenTab] = energy;

    //emit signal to notify other instances
    invokeUpdateCell(false);
}

void MicroEditor::changesFromComputerMemoryEditor (QVector< quint8 > data)
{
    //copy cell memory
    _focusCellReduced.computerMemory = data;

    //emit signal to notify other instances
    invokeUpdateCell(false);
}

void MicroEditor::changesFromTokenMemoryEditor (QVector< quint8 > data)
{
    //copy token memory
    _focusCellReduced.tokenData[_currentTokenTab] = data;

    //emit signal to notify other instances
    invokeUpdateCell(false);
}

void MicroEditor::changesFromMetaDataEditor (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription)
{
    //get cell id
    _grid->lockData();
    quint64 id = _focusCell->getId();
    QList< quint64 > ids = _focusCell->getCluster()->getCellIds();
    _focusCell->setColor(cellColor);
    _grid->unlockData();

    //set meta data
    _meta->setAndUniteClusterName(ids, clusterName);
    _meta->setCellName(id, cellName);
//    _meta->setCellColorNumber(id, cellColor);
    _meta->setCellDescription(id, cellDescription);

    //emit signal to notify macro editor
    emit metaDataUpdated();
}

void MicroEditor::changesFromSymbolTableEditor ()
{
    QWidget* widget = _tabTokenWidget->currentWidget();
    TokenTab* tokenTab= qobject_cast<TokenTab*>(widget);
    if( tokenTab ) {
        tokenTab->update(_focusCellReduced.tokenEnergies[_currentTokenTab], _focusCellReduced.tokenData[_currentTokenTab], _meta);
    }
//    _focusCellReduced.tokenData[_currentTokenTab] = data;
}

void MicroEditor::clusterTabChanged (int index)
{
    requestUpdate();
    _currentClusterTab = index;
}

void MicroEditor::tokenTabChanged (int index)
{
    if( _currentTokenTab >= 0 ) {
        TokenTab* tab = (TokenTab*)_tabTokenWidget->widget(_currentTokenTab);
        if( tab )
            tab->requestUpdate();
    }
    _currentTokenTab = index;
}

void MicroEditor::compileButtonClicked (QString code)
{
    if( (!_grid) || (!_focusCell) )
        return;

    //transfer code to cell meta data
    _grid->lockData();
    quint64 id = _focusCell->getId();
    _grid->unlockData();
    _meta->setCellCode(id, code);

    //translate code via symbol table
//    _meta->applySymbolTableToCode(code);

    //update cell data
    _focusCellReduced.computerCode = code;

    //NOTE: widgets are updated via reclustered(...)

    //emit signal to notify other instances
    _cellComputerEdit->expectCellCompilerAnswer();
    invokeUpdateCell(false);
}

void MicroEditor::invokeUpdateCell (bool clusterDataChanged)
{
    QList< AlienCell* > cells;
    QList< AlienCellReduced > newCellsData;
    cells << _focusCell;
    newCellsData << _focusCellReduced;
    emit updateCell(cells, newCellsData, clusterDataChanged);
}

void MicroEditor::setTabSymbolsWidgetVisibility ()
{
    if( _tabTokenWidget->isVisible() ) {
        _tabSymbolsWidget->setGeometry(tabPosX2, _tabClusterWidget->y(), _tabSymbolsWidget->width(), _tabSymbolsWidget->height());

        //adjust position
/*        if( _tabSymbolsWidget->parentWidget()->height() < (_tabSymbolsWidget->y() + _tabSymbolsWidget->height()+20) ) {
            _tabSymbolsWidget->setGeometry(_tabTokenWidget->x(), tabPosY1, _tabSymbolsWidget->width(), _tabSymbolsWidget->height());
        }*/
        _tabSymbolsWidget->setVisible(true);
    }
    else if( _tabComputerWidget->isVisible() ) {
        _tabSymbolsWidget->setGeometry(tabPosX1, _tabClusterWidget->y(), _tabSymbolsWidget->width(), _tabSymbolsWidget->height());
        _tabSymbolsWidget->setVisible(true);
    }
    else
        _tabSymbolsWidget->setVisible(false);
}

