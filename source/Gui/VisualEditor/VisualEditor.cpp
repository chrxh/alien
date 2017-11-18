#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>

#include "Gui/Settings.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/SpaceMetricLocal.h"
#include "PixelUniverse.h"
#include "ShapeUniverse.h"
#include "ViewportController.h"

#include "VisualEditor.h"
#include "ui_VisualEditor.h"


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

void VisualEditor::init(Notifier* notifier, SimulationController* controller, DataManipulator* manipulator, SimulationAccess* access)
{
	_pixelUniverseInit = false;
	_shapeUniverseInit = false;
	_controller = controller;
	_pixelUniverse->init(controller, access, _viewport);
	_shapeUniverse->init(notifier, controller, manipulator, _viewport);
	_viewport->init(ui->simulationView, _pixelUniverse, _shapeUniverse, ActiveScene::PixelScene);
	setActiveScene(_activeScene);
}


void VisualEditor::setActiveScene (ActiveScene activeScene)
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



