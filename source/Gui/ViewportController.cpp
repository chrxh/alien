#include <QGraphicsView>
#include <QScrollBar>

#include "Gui/Settings.h"
#include "ViewportController.h"
#include "CoordinateSystem.h"

void ViewportController::init(QGraphicsView * view, QGraphicsScene* pixelScene, QGraphicsScene* vectorScene,
    QGraphicsScene* itemScene, ActiveScene activeScene)
{
	disconnectAll();

	_view = view;
	_pixelScene = pixelScene;
    _vectorScene = vectorScene;
	_itemScene = itemScene;
	_activeScene = activeScene;
    _view->resetTransform();
    _zoom = 1.0;
	zoom(2.0, false);
	setSceneToView(boost::none, activeScene);

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
	setSceneToView(_activeScene, activeScene);
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
    if (_activeScene == ActiveScene::VectorScene) {
        p1 /= _zoom;
        p2 /= _zoom;
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

void ViewportController::zoom(double factor, bool notify)
{
    _zoom *= factor;
	disconnectAll();
    if (_activeScene != ActiveScene::VectorScene) {
        _view->scale(factor, factor);
    }
    connectAll();

    if (notify) {
        Q_EMIT zoomed();
    }
}

qreal ViewportController::getZoomFactor() const
{
    return _zoom;
}

void ViewportController::setSceneToView(optional<ActiveScene> origActiveScene, ActiveScene activeScene)
{
	if (origActiveScene && *origActiveScene == activeScene) {
		return;
	}
	if (activeScene == ActiveScene::PixelScene) {
		_view->setScene(_pixelScene);
		if (origActiveScene) {
            if (*origActiveScene == ActiveScene::VectorScene) {
                _view->scale(_zoom, _zoom);
            }
            if (*origActiveScene == ActiveScene::ItemScene) {
                _view->scale(CoordinateSystem::modelToScene(1.0), CoordinateSystem::modelToScene(1.0));
            }
        }
	}
    if (activeScene == ActiveScene::VectorScene) {
        _view->setScene(_vectorScene);
        if (origActiveScene) {
            if (*origActiveScene == ActiveScene::PixelScene) {
                _view->scale(1 / _zoom, 1 / _zoom);
            }
            if (*origActiveScene == ActiveScene::ItemScene) {
                _view->scale(CoordinateSystem::modelToScene(1.0/_zoom), CoordinateSystem::modelToScene(1.0 / _zoom));
            }
        }
    }
    if (activeScene == ActiveScene::ItemScene) {
		_view->setScene(_itemScene);
        if (origActiveScene) {
            if (*origActiveScene == ActiveScene::PixelScene) {
                _view->scale(CoordinateSystem::sceneToModel(1.0), CoordinateSystem::sceneToModel(1.0));
            }
            if (*origActiveScene == ActiveScene::VectorScene) {
                _view->scale(CoordinateSystem::sceneToModel(_zoom), CoordinateSystem::sceneToModel(_zoom));
            }
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
    _connections.clear();
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
