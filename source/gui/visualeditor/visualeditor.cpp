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
	_pixelUniverseInit = false;
	_shapeUniverseInit = false;
	_controller = controller;
	_pixelUniverse->init(controller, _viewport);
	_shapeUniverse->init(controller, _viewport);
	_viewport->init(ui->simulationView, _pixelUniverse, _shapeUniverse, ActiveScene::PixelScene);
	setActiveScene(ActiveScene::PixelScene);
}


void VisualEditor::setActiveScene (ActiveScene activeScene)
{
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



