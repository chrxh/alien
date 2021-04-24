#include <QPainter>

#include "VectorImageSectionItem.h"
#include "ViewportInterface.h"

VectorImageSectionItem::VectorImageSectionItem(ViewportInterface* viewport, IntVector2D const& displaySize, std::mutex& mutex)
    : QGraphicsItem()
    , _viewport(viewport)
    , _displaySize(displaySize)
    , _mutex(mutex)
{
}

VectorImageSectionItem::~VectorImageSectionItem()
{
}

QImagePtr VectorImageSectionItem::getImageOfVisibleRect()
{
    if (!_imageOfVisibleRect
        || (_imageOfVisibleRect->width() != _displaySize.x || _imageOfVisibleRect->height() != _displaySize.y)) {
        _imageOfVisibleRect = boost::make_shared<QImage>(_displaySize.x, _displaySize.y, QImage::Format_ARGB32);
    }
    return _imageOfVisibleRect;
}

QRectF VectorImageSectionItem::boundingRect() const
{
    return QRectF(0, 0, _displaySize.x, _displaySize.y);
}

void VectorImageSectionItem::paint(
    QPainter* painter,
    const QStyleOptionGraphicsItem* option,
    QWidget* widget /*= Q_NULLPTR*/)
{
    auto const viewportRect = _viewport->getRect();

    std::lock_guard<std::mutex> lock(_mutex);
    if (_imageOfVisibleRect) {
        painter->drawImage(0, 0, *_imageOfVisibleRect);
    }
}

void VectorImageSectionItem::setZoomFactor(double zoom)
{
    _zoom = zoom;
}

