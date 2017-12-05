#include <QGraphicsView>
#include <QScrollBar>

#include "gui/Settings.h"
#include "ViewportController.h"
#include "CoordinateSystem.h"

void ViewportController::init(QGraphicsView * view, QGraphicsScene* pixelScene, QGraphicsScene* itemScene, ActiveScene activeScene)
{
	_view = view;
	_pixelScene = pixelScene;
	_itemScene = itemScene;
	_activeScene = activeScene;
	_view->resetTransform();
	setSceneToView(boost::none, activeScene);

	connect(_view->horizontalScrollBar(), &QScrollBar::valueChanged, this, &ViewportController::scrolling);
	connect(_view->verticalScrollBar(), &QScrollBar::valueChanged, this, &ViewportController::scrolling);
}

void ViewportController::setModeToUpdate()
{
	_view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
}

void ViewportController::setModeToNoUpdate()
{
	_view->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
	_view->update();
}

void ViewportController::setActiveScene(ActiveScene activeScene)
{
/*
	saveScenePos();
*/
	setSceneToView(_activeScene, activeScene);
/*
	loadScenePos();
*/
	_activeScene = activeScene;
}

ActiveScene ViewportController::getActiveScene() const
{
	return *_activeScene;
}

QRectF ViewportController::getRect() const
{
	auto p1 = _view->mapToScene(0, 0);
	auto p2 = _view->mapToScene(_view->width(), _view->height());
	if (_activeScene == ActiveScene::ItemScene) {
		p1 = CoordinateSystem::sceneToModel(p1);
		p2 = CoordinateSystem::sceneToModel(p2);
	}
	return{ p1, p2 };
}

QVector2D ViewportController::getCenter() const
{
	QPointF centerPoint = _view->mapToScene(_view->width() / 2, _view->height() / 2);
	QVector2D result(centerPoint.x(), centerPoint.y());
	if (_activeScene == ActiveScene::ItemScene) {
		result = CoordinateSystem::sceneToModel(result);
	}
	return result;
}

void ViewportController::zoomIn()
{
	_view->scale(2.0, 2.0);
}

void ViewportController::zoomOut()
{
	_view->scale(0.5, 0.5);
}

qreal ViewportController::getZoomFactor() const
{
	return  _view->matrix().m11();
}

void ViewportController::setSceneToView(optional<ActiveScene> origActiveScene, ActiveScene activeScene)
{
	if (origActiveScene && *origActiveScene == activeScene) {
		return;
	}
	if (activeScene == ActiveScene::PixelScene) {
		_view->setScene(_pixelScene);
		if (origActiveScene) {
			_view->scale(CoordinateSystem::modelToScene(1.0), CoordinateSystem::modelToScene(1.0));
		}
	}
	if (activeScene == ActiveScene::ItemScene) {
		_view->setScene(_itemScene);
		if (origActiveScene) {
			_view->scale(CoordinateSystem::sceneToModel(1.0), CoordinateSystem::sceneToModel(1.0));
		}
	}
}

void ViewportController::saveScrollPos()
{
	_sceneScrollbarPos.x = _view->horizontalScrollBar()->value();
	_sceneScrollbarPos.y = _view->verticalScrollBar()->value();
}

void ViewportController::restoreScrollPos()
{
	_view->horizontalScrollBar()->setValue(_sceneScrollbarPos.x);
	_view->verticalScrollBar()->setValue(_sceneScrollbarPos.y);
}
