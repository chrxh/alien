#include <QGraphicsView>

#include "PixelViewport.h"

PixelViewport::PixelViewport(QGraphicsView * view, QObject * parent)
    : ViewportInterface(parent), _view(view)
{
}

QRectF PixelViewport::getRect() const
{
    auto p1 = _view->mapToScene(0, 0);
    auto p2 = _view->mapToScene(_view->width(), _view->height());
    p1.setX(std::max(0.0, p1.x()));
    p1.setY(std::max(0.0, p1.y()));
    p2.setX(std::max(0.0, p2.x()));
    p2.setY(std::max(0.0, p2.y()));
    return{ p1, p2 };
}
