#include "macroeditor.h"
#include "ui_macroeditor.h"

#include "microeditor.h"
#include "macroeditor/pixeluniverse.h"
#include "macroeditor/shapeuniverse.h"
#include "gui/editorsettings.h"
#include "gui/guisettings.h"

#include "model/entities/grid.h"
#include "model/entities/cellcluster.h"
#include "model/entities/cell.h"

#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>

MacroEditor::MacroEditor(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MacroEditor),
    _grid(0),
    _activeScene(PIXEL_SCENE),
    _pixelUniverse(new PixelUniverse(this)),
    _shapeUniverse(new ShapeUniverse(this)),
    _pixelUniverseInit(false),
    _shapeUniverseInit(false),
    _posIncrement(0),
    _updateTimer(0),
    _screenUpdatePossible(true)
{
    ui->setupUi(this);

//    ui->simulationView->setStyleSheet("background-color: #000000; color: #B0B0B0; gridline-color: #303030;");
    ui->simulationView->horizontalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
    ui->simulationView->verticalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);

    //start with pixel scene by default
    ui->simulationView->setScene(_pixelUniverse);

    //connect signals
    connect(_shapeUniverse, SIGNAL(updateCell(QList<Cell*>,QList<CellTO>,bool)), this, SIGNAL(updateCell(QList<Cell*>,QList<CellTO>,bool)));
    connect(_shapeUniverse, SIGNAL(defocus()), this, SIGNAL(defocus()), Qt::QueuedConnection);
    connect(_shapeUniverse, SIGNAL(focusCell(Cell*)), this, SIGNAL(focusCell(Cell*)), Qt::QueuedConnection);
    connect(_shapeUniverse, SIGNAL(focusEnergyParticle(EnergyParticle*)), this, SIGNAL(focusEnergyParticle(EnergyParticle*)), Qt::QueuedConnection);
    connect(_shapeUniverse, SIGNAL(energyParticleUpdated(EnergyParticle*)), this, SIGNAL(energyParticleUpdated(EnergyParticle*)), Qt::QueuedConnection);
    connect(_shapeUniverse, SIGNAL(entitiesSelected(int,int)), this, SIGNAL(entitiesSelected(int,int)));

    //set up timer
    _updateTimer = new QTimer(this);
    connect(_updateTimer, SIGNAL(timeout()), this, SLOT(updateTimerTimeout()));
    _updateTimer->start(30);

}

MacroEditor::~MacroEditor()
{
    delete ui;
}

void MacroEditor::reset ()
{
    //reset data
    _pixelUniverseInit = false;
    _shapeUniverseInit = false;
    _pixelUniverseViewMatrix = QMatrix();
    _shapeUniverseViewMatrix = QMatrix();
    ui->simulationView->setTransform(QTransform());

    //reset subobjects
    _pixelUniverse->reset();
}


void MacroEditor::setActiveScene (ActiveScene activeScene)
{
    _activeScene = activeScene;
    if( _activeScene == PIXEL_SCENE ) {

        //save position of shape universe
        _shapeUniverseViewMatrix = ui->simulationView->matrix();
        _shapeUniversePosX = ui->simulationView->horizontalScrollBar()->value();
        _shapeUniversePosY = ui->simulationView->verticalScrollBar()->value();

        //switch scene to pixel universe
        ui->simulationView->setScene(_pixelUniverse);

        //load position of pixel universe
        ui->simulationView->setMatrix(_pixelUniverseViewMatrix);
        ui->simulationView->horizontalScrollBar()->setValue(_pixelUniversePosX);
        ui->simulationView->verticalScrollBar()->setValue(_pixelUniversePosY);
    }
    if( _activeScene == SHAPE_SCENE ) {

        //save position
        _pixelUniverseViewMatrix = ui->simulationView->matrix();
        _pixelUniversePosX = ui->simulationView->horizontalScrollBar()->value();
        _pixelUniversePosY = ui->simulationView->verticalScrollBar()->value();

        //switch scene to shapeuniverse
        ui->simulationView->setScene(_shapeUniverse);

        //load position of shape universe
        ui->simulationView->setMatrix(_shapeUniverseViewMatrix);
        ui->simulationView->horizontalScrollBar()->setValue(_shapeUniversePosX);
        ui->simulationView->verticalScrollBar()->setValue(_shapeUniversePosY);
    }

    //update scene
    _screenUpdatePossible = true;
    universeUpdated(_grid, true);
}

QVector3D MacroEditor::getViewCenterPosWithInc ()
{
    //calc center of view
    QPointF posView(ui->simulationView->mapToScene(ui->simulationView->width()/2, ui->simulationView->height()/2));

    //calc center of view in simulation coordinate
    QVector3D pos(posView.x(), posView.y(), 0.0);

    //add increment
    QVector3D posIncrement(_posIncrement, -_posIncrement, 0.0);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;
    return pos + posIncrement;
}

void MacroEditor::getExtendedSelection (QList< CellCluster* >& clusters, QList< EnergyParticle* >& es)
{
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->getExtendedSelection(clusters, es);
    }
}

void MacroEditor::serializeViewMatrix (QDataStream& stream)
{
    //save position of pixel universe
    if( _activeScene == PIXEL_SCENE ) {
        _pixelUniverseViewMatrix = ui->simulationView->matrix();
        _pixelUniversePosX = ui->simulationView->horizontalScrollBar()->value();
        _pixelUniversePosY = ui->simulationView->verticalScrollBar()->value();
    }

    //save position of shape universe
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverseViewMatrix = ui->simulationView->matrix();
        _shapeUniversePosX = ui->simulationView->horizontalScrollBar()->value();
        _shapeUniversePosY = ui->simulationView->verticalScrollBar()->value();
    }

    //serialize data
    stream << _pixelUniverseViewMatrix << _shapeUniverseViewMatrix;
    stream << _pixelUniversePosX << _pixelUniversePosY;
    stream << _shapeUniversePosX << _shapeUniversePosY;
    stream << _pixelUniverseInit << _shapeUniverseInit;
}

void MacroEditor::loadViewMatrix (QDataStream& stream)
{
    stream >> _pixelUniverseViewMatrix >> _shapeUniverseViewMatrix;
    stream >> _pixelUniversePosX >> _pixelUniversePosY;
    stream >> _shapeUniversePosX >> _shapeUniversePosY;
    stream >> _pixelUniverseInit >> _shapeUniverseInit;

    //load position of pixel universe
    if( _activeScene == PIXEL_SCENE ) {
        ui->simulationView->setMatrix(_pixelUniverseViewMatrix);
        ui->simulationView->horizontalScrollBar()->setValue(_pixelUniversePosX);
        ui->simulationView->verticalScrollBar()->setValue(_pixelUniversePosY);
    }

    //load position of shape universe
    if( _activeScene == SHAPE_SCENE ) {
        ui->simulationView->setMatrix(_shapeUniverseViewMatrix);
        ui->simulationView->horizontalScrollBar()->setValue(_shapeUniversePosX);
        ui->simulationView->verticalScrollBar()->setValue(_shapeUniversePosY);
    }
}

QGraphicsView* MacroEditor::getGraphicsView ()
{
    return ui->simulationView;
}

qreal MacroEditor::getZoomFactor ()
{
    return  ui->simulationView->matrix().m11();
}

void MacroEditor::zoomIn ()
{
    ui->simulationView->scale(2.0,2.0);
}

void MacroEditor::zoomOut ()
{
    ui->simulationView->scale(0.5,0.5);
}

void MacroEditor::newCellRequested ()
{
    //calc center of view
    QPointF posView(ui->simulationView->mapToScene(ui->simulationView->width()/2, ui->simulationView->height()/2));

    //calc center of view in simulation coordinate
    QVector3D pos(posView.x(), posView.y(), 0.0);

    //add increment
    QVector3D posIncrement(_posIncrement, -_posIncrement, 0.0);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;

    //request new cell at pos
    emit requestNewCell(pos+posIncrement);
}

void MacroEditor::newEnergyParticleRequested ()
{
    //request new energy particle at pos
    emit requestNewEnergyParticle(getViewCenterPosWithInc());
}

void MacroEditor::defocused ()
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->defocused();
    }
}

void MacroEditor::delSelection_Slot ()
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        QList< Cell* > cells;
        QList< EnergyParticle* > es;
        _shapeUniverse->delSelection(cells, es);
        emit delSelection(cells, es);
    }
}

void MacroEditor::delExtendedSelection_Slot ()
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        QList< CellCluster* > clusters;
        QList< EnergyParticle* > es;
        _shapeUniverse->delExtendedSelection(clusters, es);
        //*********
//        foreach(CellCluster* c, _grid->getClusters())
//            clusters << c;
//            foreach(Cell* cell, c->getCellsRef())
//                if( (qrand() % 2 == 0) )
//                cells << cell;
        //*********
        emit delExtendedSelection(clusters, es);
    }
}

void MacroEditor::cellCreated (Cell* cell)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->cellCreated(cell);
    }
}

void MacroEditor::energyParticleCreated(EnergyParticle* e)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->energyParticleCreated(e);
    }
}

void MacroEditor::energyParticleUpdated_Slot (EnergyParticle* e)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->energyParticleUpdated_Slot(e);
    }
}

void MacroEditor::reclustered (QList< CellCluster* > clusters)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
//        _shapeUniverse->universeUpdated(_grid);
/*        foreach(CellCluster* cluster, clusters) {
            foreach(Cell* cell, cluster->getCellsRef()) {
                cell->setRelPos(cell->getRelPos());
            }
        }*/
        _shapeUniverse->reclustered(clusters);
    }
    else
        _pixelUniverse->universeUpdated(_grid);
}

void MacroEditor::universeUpdated (Grid* grid, bool force)
{
    //valid grid pointer available?
    if( grid )
        _grid = grid;
    else
        grid = _grid;
    if( !grid )
        return;

    //update possible? (see updateTimerTimeout())
    if( _screenUpdatePossible || force ) {
        _screenUpdatePossible = false;

        //update active scene
        if( _activeScene == PIXEL_SCENE ) {
            _pixelUniverse->universeUpdated(grid);

            //first time? => center view
            if( !_pixelUniverseInit ) {
               _pixelUniverseInit = true;
               ui->simulationView->scale(2.0,2.0);
               centerView(grid);
            }
        }
        if( _activeScene == SHAPE_SCENE ) {
            _shapeUniverse->universeUpdated(grid);

            //first time? => center view
            if( !_shapeUniverseInit ) {
               _shapeUniverseInit = true;
               ui->simulationView->scale(20.0,20.0);
               centerView(grid);
            }
            QGraphicsItem* cellItem = _shapeUniverse->getFocusCenterCell();
            if( cellItem )
                ui->simulationView->centerOn(cellItem);
        }
    }
}

void MacroEditor::metadataUpdated ()
{
    if( _activeScene == SHAPE_SCENE )
        _shapeUniverse->metadataUpdated();
}
/*
void MacroEditor::mousePressEvent (QMouseEvent* event)
{
    qDebug("hier");
    QWidget::mousePressEvent(event);
}
*/
void MacroEditor::updateTimerTimeout ()
{
    _screenUpdatePossible = true;
}

void MacroEditor::centerView (Grid* grid)
{
    //load size of the universe
    grid->lockData();
    qreal sizeX = grid->getSizeX();
    qreal sizeY = grid->getSizeY();
    grid->unlockData();

    //set view position
    ui->simulationView->centerOn(sizeX/2.0, sizeY/2.0);
}


