#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>
#include <QGLWidget>

#include "Gui/Settings.h"
#include "ModelBasic/SimulationAccess.h"
#include "PixelUniverseView.h"
#include "ItemUniverseView.h"
#include "ViewportController.h"

#include "VisualEditController.h"
#include "ui_VisualEditController.h"


VisualEditController::VisualEditController(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::VisualEditController)
	, _pixelUniverse(new PixelUniverseView(this))
	, _itemUniverse(new ItemUniverseView(this))
	, _viewport(new ViewportController(this))
{
    ui->setupUi(this);

    ui->simulationView->horizontalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->simulationView->verticalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);
    auto emptyScene = new QGraphicsScene(this);
    emptyScene->setBackgroundBrush(QBrush(Const::UniverseColor));

    QPixmap startScreenPixmap("://Tutorial/logo.png");
    emptyScene->addPixmap(startScreenPixmap);
    ui->simulationView->setScene(emptyScene);
}

VisualEditController::~VisualEditController()
{
    delete ui;
}

void VisualEditController::init(Notifier* notifier, SimulationController* controller, SimulationAccess* access, DataRepository* repository)
{
	_pixelUniverseInit = false;
	_itemUniverseInit = false;
	_controller = controller;
	_activeScene = ActiveScene::PixelScene;
    ui->simulationView->setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers)));
    _viewport->init(ui->simulationView, _pixelUniverse, _itemUniverse, _activeScene);
    _pixelUniverse->init(notifier, controller, access, repository, _viewport);
	_itemUniverse->init(notifier, controller, repository, _viewport);
}

void VisualEditController::refresh()
{
	_pixelUniverse->refresh();
	_itemUniverse->refresh();
}


void VisualEditController::setActiveScene (ActiveScene activeScene)
{
	_viewport->saveScrollPos();
	if (activeScene == ActiveScene::PixelScene) {
		_itemUniverse->deactivate();
	}
	if (activeScene == ActiveScene::ItemScene) {
		_pixelUniverse->deactivate();
	}
	_activeScene = activeScene;
	_viewport->setActiveScene(activeScene);

	if (activeScene == ActiveScene::PixelScene) {
		_pixelUniverse->activate();
	}
	if (activeScene == ActiveScene::ItemScene) {
		_itemUniverse->activate();
	}
	_viewport->restoreScrollPos();
}

QVector2D VisualEditController::getViewCenterWithIncrement ()
{
	QVector2D center = _viewport->getCenter();

    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;
    return center + posIncrement;
}

QGraphicsView* VisualEditController::getGraphicsView ()
{
    return ui->simulationView;
}

double VisualEditController::getZoomFactor ()
{
	return _viewport->getZoomFactor();
}

void VisualEditController::zoom (double factor)
{
	_viewport->zoom(factor);
}

void VisualEditController::toggleCenterSelection(bool value)
{
	_itemUniverse->toggleCenterSelection(value);
}



