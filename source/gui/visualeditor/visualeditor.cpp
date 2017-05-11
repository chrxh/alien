#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>

#include "gui/Settings.h"
#include "gui/texteditor/texteditor.h"
#include "model/AccessPorts/SimulationAccess.h"
#include "model/Context/UnitContext.h"
#include "model/Context/SpaceMetric.h"
#include "pixeluniverse.h"
#include "shapeuniverse.h"
#include "ViewportController.h"

#include "visualeditor.h"
#include "ui_visualeditor.h"


VisualEditor::VisualEditor(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::VisualEditor)
	, _pixelUniverse(new PixelUniverse(this))
	, _shapeUniverse(new ShapeUniverse(this))
	, _viewport(new ViewportController(this))
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
	_viewport->init(ui->simulationView, _pixelUniverse, _shapeUniverse, ActiveScene::PixelScene);
}

void VisualEditor::reset ()
{
    _pixelUniverseInit = false;
    _shapeUniverseInit = false;
	_viewport->initViewMatrices();
    _pixelUniverse->reset();
}


void VisualEditor::setActiveScene (ActiveScene activeScene)
{
	_viewport->setActiveScene(activeScene);
    _screenUpdatePossible = true;
}

QVector2D VisualEditor::getViewCenterWithIncrement ()
{
	QVector2D center = _viewport->getCenter();

    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;
    return center + posIncrement;
}

QGraphicsView* VisualEditor::getGraphicsView ()
{
    return ui->simulationView;
}

qreal VisualEditor::getZoomFactor ()
{
	return _viewport->getZoomFactor();
}

void VisualEditor::zoomIn ()
{
	_viewport->zoomIn();
}

void VisualEditor::zoomOut ()
{
	_viewport->zoomOut();
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

void VisualEditor::toggleInformation(bool on)
{
	if (_viewport->getActiveScene() == ActiveScene::ShapeScene) {
		_shapeUniverse->toggleInformation(on);
	}
}



