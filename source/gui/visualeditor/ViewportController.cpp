#include <QGraphicsView>
#include <QScrollBar>

#include "gui/GuiSettings.h"
#include "ViewportController.h"

void ViewportController::init(QGraphicsView * view, QGraphicsScene* pixelScene, QGraphicsScene* shapeScene, ActiveScene activeScene)
{
	_view = view;
	_activeScene = activeScene;
	_pixelScene = pixelScene;
	_shapeScene = shapeScene;
	setSceneToView(activeScene);
	initViewMatrices();
}

void ViewportController::initViewMatrices()
{
	_pixelSceneViewMatrix = QMatrix();
	_pixelSceneViewMatrix.scale(2.0, 2.0);
	_shapeSceneViewMatrix = QMatrix();
	_shapeSceneViewMatrix.scale(20.0 / GRAPHICS_ITEM_SIZE, 20.0 / GRAPHICS_ITEM_SIZE);
}

void ViewportController::setActiveScene(ActiveScene activeScene)
{
	saveScenePos(_activeScene);
	setSceneToView(activeScene);
	loadScenePos(activeScene);
	_activeScene = activeScene;
}

ActiveScene ViewportController::getActiveScene() const
{
	return _activeScene;
}

QRectF ViewportController::getRect() const
{
	return _view->sceneRect();
}

QVector2D ViewportController::getCenter() const
{
	QPointF centerPoint = _view->mapToScene(_view->width() / 2, _view->height() / 2);
	QVector2D result(centerPoint.x(), centerPoint.y());
	if (_activeScene == ActiveScene::ShapeScene) {
		result = result / GRAPHICS_ITEM_SIZE;
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

void ViewportController::setSceneToView(ActiveScene activeScene)
{
	if (activeScene == ActiveScene::PixelScene) {
		_view->setScene(_pixelScene);
	}
	if (activeScene == ActiveScene::ShapeScene) {
		_view->setScene(_shapeScene);
	}
}

void ViewportController::saveScenePos(ActiveScene activeScene)
{
	if (activeScene == ActiveScene::PixelScene) {
		_pixelSceneViewMatrix = _view->matrix();
		_pixelSceneScrollbarPos.x = _view->horizontalScrollBar()->value();
		_pixelSceneScrollbarPos.y = _view->verticalScrollBar()->value();
	}
	if (activeScene == ActiveScene::ShapeScene) {
		_shapeSceneViewMatrix = _view->matrix();
		_shapeSceneScrollbarPos.x = _view->horizontalScrollBar()->value();
		_shapeSceneScrollbarPos.y = _view->verticalScrollBar()->value();
	}
}

void ViewportController::loadScenePos(ActiveScene activeScene)
{
	if (_activeScene == ActiveScene::PixelScene) {
		_view->setMatrix(_pixelSceneViewMatrix);
		_view->horizontalScrollBar()->setValue(_pixelSceneScrollbarPos.x);
		_view->verticalScrollBar()->setValue(_pixelSceneScrollbarPos.x);
	}
	if (_activeScene == ActiveScene::ShapeScene) {
		_view->setMatrix(_shapeSceneViewMatrix);
		_view->horizontalScrollBar()->setValue(_shapeSceneScrollbarPos.x);
		_view->verticalScrollBar()->setValue(_shapeSceneScrollbarPos.y);
	}
}
