#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>
#include <QGraphicsBlurEffect>

#include "Gui/Settings.h"
#include "ModelBasic/SimulationAccess.h"
#include "PixelUniverseView.h"
#include "VectorUniverseView.h"
#include "ItemUniverseView.h"
#include "ViewportController.h"

#include "SimulationViewWidget.h"
#include "ui_SimulationViewWidget.h"


SimulationViewWidget::SimulationViewWidget(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::SimulationViewWidget)
	, _pixelUniverse(new PixelUniverseView(this))
    , _vectorUniverse(new VectorUniverseView(this))
    , _itemUniverse(new ItemUniverseView(this))
	, _viewport(new ViewportController(this))
{
    ui->setupUi(this);

    ui->simulationView->horizontalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->simulationView->verticalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);
    auto emptyScene = new QGraphicsScene(this);
    emptyScene->setBackgroundBrush(QBrush(Const::UniverseColor));

    QPixmap startScreenPixmap("://logo.png");
    emptyScene->addPixmap(startScreenPixmap);
    ui->simulationView->setScene(emptyScene);
}

SimulationViewWidget::~SimulationViewWidget()
{
    delete ui;
}

void SimulationViewWidget::init(Notifier* notifier, SimulationController* controller, SimulationAccess* access, DataRepository* repository)
{
    _pixelUniverseInit = false;
	_itemUniverseInit = false;
	_controller = controller;
	_activeScene = ActiveScene::PixelScene;
    _pixelUniverse->deactivate();
    _vectorUniverse->deactivate();
    _itemUniverse->deactivate();
    _viewport->init(ui->simulationView, _pixelUniverse, _vectorUniverse, _itemUniverse, _activeScene);
    _pixelUniverse->init(notifier, controller, access, repository, _viewport);
    _vectorUniverse->init(notifier, controller, access, repository, _viewport);
    _itemUniverse->init(notifier, controller, repository, _viewport);
}

void SimulationViewWidget::refresh()
{
	_pixelUniverse->refresh();
    _vectorUniverse->refresh();
    _itemUniverse->refresh();
}


void SimulationViewWidget::setActiveScene (ActiveScene activeScene)
{
	_viewport->saveScrollPos();
	if (_activeScene == ActiveScene::PixelScene) {
        _pixelUniverse->deactivate();
	}
    if (_activeScene == ActiveScene::VectorScene) {
        _vectorUniverse->deactivate();
    }
	if (_activeScene == ActiveScene::ItemScene) {
		_itemUniverse->deactivate();
	}
    _activeScene = activeScene;
	_viewport->setActiveScene(activeScene);

	if (activeScene == ActiveScene::PixelScene) {
		_pixelUniverse->activate();
	}
    if (activeScene == ActiveScene::VectorScene) {
        _vectorUniverse->activate();
    }
    if (activeScene == ActiveScene::ItemScene) {
		_itemUniverse->activate();
	}
	_viewport->restoreScrollPos();
}

QVector2D SimulationViewWidget::getViewCenterWithIncrement ()
{
	QVector2D center = _viewport->getCenter();

    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if( _posIncrement > 9.0)
        _posIncrement = 0.0;
    return center + posIncrement;
}

double SimulationViewWidget::getZoomFactor ()
{
	return _viewport->getZoomFactor();
}

void SimulationViewWidget::scrollToPos(QVector2D const & pos)
{
    _viewport->scrollToPos(pos, NotifyScrollChanged::No);
}

void SimulationViewWidget::zoom (double factor)
{
	_viewport->zoom(factor);
}

void SimulationViewWidget::toggleCenterSelection(bool value)
{
	_itemUniverse->toggleCenterSelection(value);
}



