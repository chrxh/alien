#include <QPainter>

#include "VectorImageSectionItem.h"
#include "ViewportInterface.h"

VectorImageSectionItem::VectorImageSectionItem(ViewportInterface* viewport, QRectF const& boundingRect, int zoom, std::mutex& mutex)
    : QGraphicsItem(), _viewport(viewport), _boundingRect(boundingRect), _zoom(zoom), _mutex(mutex)
{
    auto const viewportRect = _viewport->getRect();
    _imageOfVisibleRect = boost::make_shared<QImage>(viewportRect.width()*zoom, viewportRect.height()*zoom, QImage::Format_RGB32);
    _imageOfVisibleRect->fill(QColor(0, 0, 0));
}

VectorImageSectionItem::~VectorImageSectionItem()
{
}

QImagePtr VectorImageSectionItem::getImageOfVisibleRect()
{
    //resize image?
    auto rect = _viewport->getRect();
    IntVector2D imageSize = {
        std::min(static_cast<int>(_boundingRect.width()), static_cast<int>(rect.width())) * _zoom,
        std::min(static_cast<int>(_boundingRect.height()), static_cast<int>(rect.height())) * _zoom
    };
    if (_imageOfVisibleRect->width() != imageSize.x || _imageOfVisibleRect->height() != imageSize.y) {
        _imageOfVisibleRect = boost::make_shared<QImage>(imageSize.x, imageSize.y, QImage::Format_ARGB32);
        _imageOfVisibleRect->fill(QColor(0, 0, 0));
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
        std::max(0.0f, static_cast<float>(viewportRect.x())*_zoom),
        std::max(0.0f, static_cast<float>(viewportRect.y())*_zoom),
        *_imageOfVisibleRect);
}

