#include <QPainter>

#include "VectorImageSectionItem.h"
#include "ViewportInterface.h"

VectorImageSectionItem::VectorImageSectionItem(ViewportInterface* viewport, IntVector2D const& universeSize, std::mutex& mutex)
    : QGraphicsItem(), _viewport(viewport), _universeSize(universeSize), _mutex(mutex)
{
}

VectorImageSectionItem::~VectorImageSectionItem()
{
}

QImagePtr VectorImageSectionItem::getImageOfVisibleRect()
{
    //resize image?
    auto const rect = _viewport->getRect();
    IntVector2D imageSize = {
        static_cast<int>(std::min(static_cast<double>(_universeSize.x) * _zoom, rect.width() * _zoom)),
        static_cast<int>(std::min(static_cast<double>(_universeSize.y) * _zoom, rect.height() * _zoom)),
    };

    if (!_imageOfVisibleRect || (_imageOfVisibleRect->width() != imageSize.x || _imageOfVisibleRect->height() != imageSize.y)) {
        _imageOfVisibleRect = boost::make_shared<QImage>(imageSize.x, imageSize.y, QImage::Format_ARGB32);
    }

    return _imageOfVisibleRect;
}

QRectF VectorImageSectionItem::boundingRect() const
{
    return QRectF(0, 0, static_cast<double>(_universeSize.x) * _zoom, static_cast<double>(_universeSize.y) * _zoom);
}

void VectorImageSectionItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget /*= Q_NULLPTR*/)
{
    auto const viewportRect = _viewport->getRect();

    std::lock_guard<std::mutex> lock(_mutex);
    if (_imageOfVisibleRect) {
        painter->drawImage(
            static_cast<float>(viewportRect.x() * _zoom),
            static_cast<float>(viewportRect.y() * _zoom),
            *_imageOfVisibleRect);
    }
}

void VectorImageSectionItem::setZoomFactor(double zoom)
{
    _zoom = zoom;
}

