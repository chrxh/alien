#include <QGraphicsView>
#include <QScrollBar>

#include "Gui/Settings.h"
#include "ViewportController.h"
#include "CoordinateSystem.h"

void ViewportController::init(QGraphicsView * view, QGraphicsScene* pixelScene, QGraphicsScene* itemScene, ActiveScene activeScene)
{
	_view = view;
	_pixelScene = pixelScene;
	_itemScene = itemScene;
	_activeScene = activeScene;
    _view->resetTransform();
	zoom(2.0);
	setSceneToView(boost::none, activeScene);

	disconnectAll();
	connectAll();
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

void ViewportController::zoom(double factor)
{
	disconnectAll();
	_view->scale(factor, factor);
    connectAll();

	Q_EMIT scrolled();
}

qreal ViewportController::getZoomFactor() const
{
	if (_activeScene == ActiveScene::PixelScene) {
		return  _view->matrix().m11();
	}
	return CoordinateSystem::modelToScene(_view->matrix().m11());
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

void ViewportController::connectAll()
{
	_connections.push_back(QObject::connect(_view->horizontalScrollBar(), &QScrollBar::valueChanged, this, &ViewportController::scrolled));
	_connections.push_back(QObject::connect(_view->verticalScrollBar(), &QScrollBar::valueChanged, this, &ViewportController::scrolled));
}

void ViewportController::disconnectAll()
{
	for (auto const& connection : _connections) {
		QObject::disconnect(connection);
	}
}

void ViewportController::scrollToPos(QVector2D pos, NotifyScrollChanged notify)
{
	if (notify == NotifyScrollChanged::No) {
		disconnectAll();
	}
	if (_activeScene == ActiveScene::ItemScene) {
		pos = CoordinateSystem::modelToScene(pos);
	}
	_view->centerOn(pos.x(), pos.y());
	if (notify == NotifyScrollChanged::No) {
		connectAll();
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
