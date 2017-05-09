#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>

#include "gui/GuiSettings.h"
#include "gui/GuiSettings.h"
#include "gui/texteditor/texteditor.h"
#include "model/AccessPorts/SimulationAccess.h"
#include "model/context/UnitContext.h"
#include "model/context/SpaceMetric.h"
#include "pixeluniverse.h"
#include "shapeuniverse.h"


#include "visualeditor.h"
#include "ui_visualeditor.h"


VisualEditor::VisualEditor(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::VisualEditor)
	, _activeScene(PixelScene)
	, _pixelUniverse(new PixelUniverse(this))
	, _shapeUniverse(new ShapeUniverse(this))
{
    ui->setupUi(this);

    ui->simulationView->horizontalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
    ui->simulationView->verticalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
}

VisualEditor::~VisualEditor()
{
    delete ui;
}

void VisualEditor::init(SimulationController* controller)
{
	_controller = controller;
	_pixelUniverse->init(controller);
	ui->simulationView->setScene(_pixelUniverse);
}

void VisualEditor::reset ()
{
    _pixelUniverseInit = false;
    _shapeUniverseInit = false;
    _pixelUniverseViewMatrix = QMatrix();
    _shapeUniverseViewMatrix = QMatrix();
    ui->simulationView->setTransform(QTransform());
    _pixelUniverse->reset();
}


void VisualEditor::setActiveScene (ActiveScene activeScene)
{
    _activeScene = activeScene;
    if( _activeScene == PixelScene ) {

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
    if( _activeScene == ShapeScene ) {

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
}

QVector2D VisualEditor::getViewCenterPosWithInc ()
{
    //calc center of view
    QPointF posView(ui->simulationView->mapToScene(ui->simulationView->width()/2, ui->simulationView->height()/2));

    //calc center of view in simulation coordinate
    QVector2D pos(posView.x(), posView.y());

	if (_activeScene == ShapeScene) {
		pos = pos / GRAPHICS_ITEM_SIZE;
	}

    //add increment
    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;
    return pos + posIncrement;
}

void VisualEditor::serializeViewMatrix (QDataStream& stream)
{
    //save position of pixel universe
    if( _activeScene == PixelScene ) {
        _pixelUniverseViewMatrix = ui->simulationView->matrix();
        _pixelUniversePosX = ui->simulationView->horizontalScrollBar()->value();
        _pixelUniversePosY = ui->simulationView->verticalScrollBar()->value();
    }

    //save position of shape universe
    if( _activeScene == ShapeScene ) {
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
    if( _activeScene == PixelScene ) {
        ui->simulationView->setMatrix(_pixelUniverseViewMatrix);
        ui->simulationView->horizontalScrollBar()->setValue(_pixelUniversePosX);
        ui->simulationView->verticalScrollBar()->setValue(_pixelUniversePosY);
    }

    //load position of shape universe
    if( _activeScene == ShapeScene ) {
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


/*void VisualEditor::universeUpdated (SimulationContext* context, bool force)
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
*/

void VisualEditor::metadataUpdated ()
{
    if( _activeScene == ShapeScene )
        _shapeUniverse->metadataUpdated();
}

void VisualEditor::toggleInformation(bool on)
{
	if (_activeScene == ShapeScene) {
		_shapeUniverse->toggleInformation(on);
	}
}



