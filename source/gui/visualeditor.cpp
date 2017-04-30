#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>

#include "model/simulationunitcontext.h"
#include "model/topology.h"
#include "model/entities/cellcluster.h"
#include "model/entities/cell.h"
#include "visualeditor/pixeluniverse.h"
#include "visualeditor/shapeuniverse.h"
#include "gui/guisettings.h"
#include "gui/guisettings.h"


#include "texteditor.h"
#include "visualeditor.h"
#include "ui_visualeditor.h"


VisualEditor::VisualEditor(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::VisualEditor)
	, _activeScene(PIXEL_SCENE)
	, _pixelUniverse(new PixelUniverse(this))
	, _shapeUniverse(new ShapeUniverse(this))
{
    ui->setupUi(this);

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

VisualEditor::~VisualEditor()
{
    delete ui;
}

void VisualEditor::reset ()
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


void VisualEditor::setActiveScene (ActiveScene activeScene)
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
    universeUpdated(_context, true);
}

QVector3D VisualEditor::getViewCenterPosWithInc ()
{
    //calc center of view
    QPointF posView(ui->simulationView->mapToScene(ui->simulationView->width()/2, ui->simulationView->height()/2));

    //calc center of view in simulation coordinate
    QVector3D pos(posView.x(), posView.y(), 0.0);

	if (_activeScene == SHAPE_SCENE) {
		pos = pos / GRAPHICS_ITEM_SIZE;
	}

    //add increment
    QVector3D posIncrement(_posIncrement, -_posIncrement, 0.0);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;
    return pos + posIncrement;
}

void VisualEditor::getExtendedSelection (QList< CellCluster* >& clusters, QList< EnergyParticle* >& es)
{
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->getExtendedSelection(clusters, es);
    }
}

void VisualEditor::serializeViewMatrix (QDataStream& stream)
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

void VisualEditor::loadViewMatrix (QDataStream& stream)
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

QGraphicsView* VisualEditor::getGraphicsView ()
{
    return ui->simulationView;
}

qreal VisualEditor::getZoomFactor ()
{
    return  ui->simulationView->matrix().m11();
}

void VisualEditor::zoomIn ()
{
    ui->simulationView->scale(2.0,2.0);
}

void VisualEditor::zoomOut ()
{
    ui->simulationView->scale(0.5,0.5);
}

void VisualEditor::newCellRequested ()
{
    //request new cell at pos
    emit requestNewCell(getViewCenterPosWithInc());
}

void VisualEditor::newEnergyParticleRequested ()
{
    //request new energy particle at pos
    emit requestNewEnergyParticle(getViewCenterPosWithInc());
}

void VisualEditor::defocused ()
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->defocused();
    }
}

void VisualEditor::delSelection_Slot ()
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        QList< Cell* > cells;
        QList< EnergyParticle* > es;
        _shapeUniverse->delSelection(cells, es);
        emit delSelection(cells, es);
    }
}

void VisualEditor::delExtendedSelection_Slot ()
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        QList< CellCluster* > clusters;
        QList< EnergyParticle* > es;
        _shapeUniverse->delExtendedSelection(clusters, es);
        emit delExtendedSelection(clusters, es);
    }
}

void VisualEditor::cellCreated (Cell* cell)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->cellCreated(cell);
    }
}

void VisualEditor::energyParticleCreated(EnergyParticle* e)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->energyParticleCreated(e);
    }
}

void VisualEditor::energyParticleUpdated_Slot (EnergyParticle* e)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->energyParticleUpdated_Slot(e);
    }
}

void VisualEditor::reclustered (QList< CellCluster* > clusters)
{
    //function only in shape scene
    if( _activeScene == SHAPE_SCENE ) {
        _shapeUniverse->reclustered(clusters);
    }
    else
        _pixelUniverse->universeUpdated(_context);
}

void VisualEditor::universeUpdated (SimulationUnitContext* context, bool force)
{
    if(context)
        _context = context;
    else
		context = _context;
    if( !context)
        return;

    //update possible? (see updateTimerTimeout())
    if( _screenUpdatePossible || force ) {
        _screenUpdatePossible = false;

        //update active scene
        if( _activeScene == PIXEL_SCENE ) {
            _pixelUniverse->universeUpdated(context);

            //first time? => center view
            if( !_pixelUniverseInit ) {
               _pixelUniverseInit = true;
               ui->simulationView->scale(2.0,2.0);
               centerView(context);
            }
        }
        if( _activeScene == SHAPE_SCENE ) {
            _shapeUniverse->universeUpdated(context);

            //first time? => center view
            if( !_shapeUniverseInit ) {
               _shapeUniverseInit = true;
               ui->simulationView->scale(20.0 / GRAPHICS_ITEM_SIZE, 20.0 / GRAPHICS_ITEM_SIZE);
               centerView(context);
            }
            QGraphicsItem* cellItem = _shapeUniverse->getFocusCenterCell();
            if( cellItem )
                ui->simulationView->centerOn(cellItem);
        }
    }
}

void VisualEditor::metadataUpdated ()
{
    if( _activeScene == SHAPE_SCENE )
        _shapeUniverse->metadataUpdated();
}

void VisualEditor::toggleInformation(bool on)
{
	if (_activeScene == SHAPE_SCENE) {
		_shapeUniverse->toggleInformation(on);
	}
}

void VisualEditor::updateTimerTimeout ()
{
    _screenUpdatePossible = true;
}

void VisualEditor::centerView (SimulationUnitContext* context)
{
    //load size of the universe
	context->lock();
	Topology* topo = context->getTopology();
    qreal sizeX = topo->getSize().x;
    qreal sizeY = topo->getSize().y;
	context->unlock();

    //set view position
    ui->simulationView->centerOn(sizeX/2.0*GRAPHICS_ITEM_SIZE, sizeY/2.0*GRAPHICS_ITEM_SIZE);
}


