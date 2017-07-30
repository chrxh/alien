#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>

#include "gui/Settings.h"
#include "gui/texteditor/TextEditor.h"
#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/SpaceMetric.h"
#include "PixelUniverseT.h"
#include "ShapeUniverseT.h"
#include "ViewportController.h"

#include "VisualEditorT.h"
#include "ui_visualeditor.h"


VisualEditorT::VisualEditorT(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::VisualEditor)
	, _pixelUniverse(new PixelUniverseT(this))
	, _shapeUniverse(new ShapeUniverseT(this))
	, _viewport(new ViewportController(this))
{
    ui->setupUi(this);

    ui->simulationView->horizontalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
    ui->simulationView->verticalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
}

VisualEditorT::~VisualEditorT()
{
    delete ui;
}

void VisualEditorT::init(SimulationController* controller, SimulationAccess* access)
{
	_pixelUniverseInit = false;
	_shapeUniverseInit = false;
	_controller = controller;
	_pixelUniverse->init(controller, access, _viewport);
	_shapeUniverse->init(controller, access, _viewport);
	_viewport->init(ui->simulationView, _pixelUniverse, _shapeUniverse, ActiveScene::PixelScene);
	setActiveScene(_activeScene);
}


void VisualEditorT::setActiveScene (ActiveScene activeScene)
{
	_activeScene = activeScene;
	_viewport->setActiveScene(activeScene);
    _screenUpdatePossible = true;
	if (activeScene == ActiveScene::PixelScene) {
		_shapeUniverse->deactivate();
		_pixelUniverse->activate();
	}
	if (activeScene == ActiveScene::ShapeScene) {
		_pixelUniverse->deactivate();
		_shapeUniverse->activate();
	}
}

QVector2D VisualEditorT::getViewCenterWithIncrement ()
{
	QVector2D center = _viewport->getCenter();

    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;
    return center + posIncrement;
}

QGraphicsView* VisualEditorT::getGraphicsView ()
{
    return ui->simulationView;
}

qreal VisualEditorT::getZoomFactor ()
{
	return _viewport->getZoomFactor();
}

void VisualEditorT::zoomIn ()
{
	_viewport->zoomIn();
}

void VisualEditorT::zoomOut ()
{
	_viewport->zoomOut();
}



