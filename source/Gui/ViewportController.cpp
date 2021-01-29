/*
#include <QGraphicsView>
#include <QScrollBar>

#include "Gui/Settings.h"
#include "ViewportController.h"
#include "CoordinateSystem.h"

void ViewportController::init(QGraphicsView * view, QGraphicsScene* pixelScene, QGraphicsScene* vectorScene,
    QGraphicsScene* itemScene, ActiveView activeScene)
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

void ViewportController::setActiveScene(ActiveView activeScene)
{
	setSceneToView(_activeScene, activeScene);
	_activeScene = activeScene;
}

ActiveView ViewportController::getActiveScene() const
{
	return *_activeScene;
}

QRectF ViewportController::getRect() const
{
	auto p1 = _view->mapToScene(0, 0);
	auto p2 = _view->mapToScene(_view->width(), _view->height());
	if (_activeScene == ActiveView::ItemScene) {
		p1 = CoordinateSystem::sceneToModel(p1);
		p2 = CoordinateSystem::sceneToModel(p2);
	}
    if (_activeScene == ActiveView::VectorScene) {
        p1 /= _zoom;
        p2 /= _zoom;
    }
    p1.setX(std::max(0.0, p1.x()));
    p1.setY(std::max(0.0, p1.y()));
    p2.setX(std::max(0.0, p2.x()));
    p2.setY(std::max(0.0, p2.y()));
    return{ p1, p2 };
}

QVector2D ViewportController::getCenter() const
{
	QPointF centerPoint = _view->mapToScene(static_cast<double>(_view->width()) / 2.0, static_cast<double>(_view->height()) / 2.0);
	QVector2D result(centerPoint.x(), centerPoint.y());
	if (_activeScene == ActiveView::ItemScene) {
		result = CoordinateSystem::sceneToModel(result);
	}
    if (_activeScene == ActiveView::VectorScene) {
        result /= _zoom;
    }
	return result;
}

void ViewportController::zoom(double factor, bool notify)
{
    _zoom *= factor;

    disconnectAll();
    if (_activeScene != ActiveView::VectorScene) {
        _view->scale(factor, factor);
    }
    else {
        auto center = getCenter();
        auto sceneRect = _vectorScene->sceneRect();
        sceneRect.setWidth(sceneRect.width()*factor);
        sceneRect.setHeight(sceneRect.height()*factor);
        _vectorScene->setSceneRect(sceneRect);
        scrollToPos(center * factor, NotifyScrollChanged::No);
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

void ViewportController::setSceneToView(optional<ActiveView> origActiveScene, ActiveView activeScene)
{
	if (origActiveScene && *origActiveScene == activeScene) {
		return;
	}
	if (activeScene == ActiveView::PixelScene) {
		_view->setScene(_pixelScene);
		if (origActiveScene) {
            if (*origActiveScene == ActiveView::VectorScene) {
                _view->scale(_zoom, _zoom);
            }
            if (*origActiveScene == ActiveView::ItemScene) {
                _view->scale(CoordinateSystem::modelToScene(1.0), CoordinateSystem::modelToScene(1.0));
            }
        }
	}
    if (activeScene == ActiveView::VectorScene) {
        _view->setScene(_vectorScene);
        if (origActiveScene) {
            if (*origActiveScene == ActiveView::PixelScene) {
                _view->scale(1 / _zoom, 1 / _zoom);
            }
            if (*origActiveScene == ActiveView::ItemScene) {
                _view->scale(CoordinateSystem::modelToScene(1.0/_zoom), CoordinateSystem::modelToScene(1.0 / _zoom));
            }
        }
    }
    if (activeScene == ActiveView::ItemScene) {
		_view->setScene(_itemScene);
        if (origActiveScene) {
            if (*origActiveScene == ActiveView::PixelScene) {
                _view->scale(CoordinateSystem::sceneToModel(1.0), CoordinateSystem::sceneToModel(1.0));
            }
            if (*origActiveScene == ActiveView::VectorScene) {
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
	if (_activeScene == ActiveView::ItemScene) {
		pos = CoordinateSystem::modelToScene(pos);
	}
    if (_activeScene == ActiveView::VectorScene) {
        pos *= _zoom;
    }

    printf("center: %f, %f\n", pos.x(), pos.y());
    _view->centerOn(pos.x(), pos.y());
	if (notify == NotifyScrollChanged::No) {
		connectAll();
	}
}

void ViewportController::saveScrollPos()
{
	_sceneScrollbarPosX = static_cast<double>(_view->horizontalScrollBar()->value()) / static_cast<double>(_view->horizontalScrollBar()->maximum());
    _sceneScrollbarPosY = static_cast<double>(_view->verticalScrollBar()->value()) / static_cast<double>(_view->verticalScrollBar()->maximum());
}

void ViewportController::restoreScrollPos()
{
/ *
	_view->horizontalScrollBar()->setValue(static_cast<int>(_sceneScrollbarPosX * _view->horizontalScrollBar()->maximum()));
    _view->verticalScrollBar()->setValue(static_cast<int>(_sceneScrollbarPosY * _view->verticalScrollBar()->maximum()));
* /
//     printf("MAX: %d, %d\n", _view->horizontalScrollBar()->maximum(), _view->verticalScrollBar()->maximum());
}
*/
