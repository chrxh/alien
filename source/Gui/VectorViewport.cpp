#include <QGraphicsView>

#include "VectorViewport.h"

VectorViewport::VectorViewport(QGraphicsView * view, QObject * parent)
    : ViewportInterface(parent), _view(view)
{
}

void VectorViewport::setZoomFactor(double zoomFactor)
{
    _zoomFactor = zoomFactor;
}

QRectF VectorViewport::getRect() const
{
    auto p1 = _view->mapToScene(0, 0);
    auto p2 = _view->mapToScene(_view->width(), _view->height());
    p1.setX(std::max(0.0, p1.x() / _zoomFactor));
    p1.setY(std::max(0.0, p1.y() / _zoomFactor));
    p2.setX(std::max(0.0, p2.x() / _zoomFactor));
    p2.setY(std::max(0.0, p2.y() / _zoomFactor));
    return{ p1, p2 };
}
