#include <QPainter>

#include "VectorImageSectionItem.h"
#include "ViewportInterface.h"

VectorImageSectionItem::VectorImageSectionItem(ViewportInterface* viewport, QRectF const& boundingRect, std::mutex& mutex)
    : QGraphicsItem(), _viewport(viewport), _boundingRect(boundingRect), _mutex(mutex)
{
    auto const viewportRect = _viewport->getRect();
//    _imageOfVisibleRect = boost::make_shared<QImage>(viewportRect.width()*zoom, viewportRect.height()*zoom, QImage::Format_RGB32);
}

VectorImageSectionItem::~VectorImageSectionItem()
{
}

QImagePtr VectorImageSectionItem::getImageOfVisibleRect()
{
    //resize image?
    auto const rect = _viewport->getRect();
    IntVector2D imageSize = {
        static_cast<int>(std::min(_boundingRect.width(), rect.width()) * _zoom),
        static_cast<int>(std::min(_boundingRect.height(), rect.height()) * _zoom),
    };
    if (!_imageOfVisibleRect || (_imageOfVisibleRect->width() != imageSize.x || _imageOfVisibleRect->height() != imageSize.y)) {
        _imageOfVisibleRect = boost::make_shared<QImage>(imageSize.x, imageSize.y, QImage::Format_ARGB32);
    }

    return _imageOfVisibleRect;
}

QRectF VectorImageSectionItem::boundingRect() const
{
    return QRectF(_boundingRect.left()*_zoom, _boundingRect.top()*_zoom, _boundingRect.width()*_zoom, _boundingRect.height()*_zoom);
}

void VectorImageSectionItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget /*= Q_NULLPTR*/)
{
    auto const viewportRect = _viewport->getRect();

    std::lock_guard<std::mutex> lock(_mutex);

    painter->drawImage(
        static_cast<float>(viewportRect.x())*_zoom,
        static_cast<float>(viewportRect.y())*_zoom,
        *_imageOfVisibleRect);
}

void VectorImageSectionItem::setZoom(double zoom)
{
    _zoom = zoom;
}

