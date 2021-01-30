#include <QPainter>

#include "PixelImageSectionItem.h"
#include "ViewportInterface.h"

PixelImageSectionItem::PixelImageSectionItem(ViewportInterface* viewport, IntVector2D const& universeSize, std::mutex& mutex)
    : QGraphicsItem(), _viewport(viewport), _universeSize(universeSize), _mutex(mutex)
{
}

PixelImageSectionItem::~PixelImageSectionItem()
{
}

QImagePtr PixelImageSectionItem::getImageOfVisibleRect()
{
    //resize image?
    auto rect = _viewport->getRect();
    IntVector2D viewportSize{ static_cast<int>(rect.width()), static_cast<int>(rect.height()) };
    viewportSize.x = std::min(_universeSize.x, viewportSize.x);
    viewportSize.y = std::min(_universeSize.y, viewportSize.y);
    if (!_imageOfVisibleRect || (_imageOfVisibleRect->width() != viewportSize.x || _imageOfVisibleRect->height() != viewportSize.y)) {
        _imageOfVisibleRect = boost::make_shared<QImage>(viewportSize.x, viewportSize.y, QImage::Format_ARGB32);
    }

    return _imageOfVisibleRect;
}

QRectF PixelImageSectionItem::boundingRect() const
{
    return QRectF(0, 0, _universeSize.x, _universeSize.y);
}

void PixelImageSectionItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget /*= Q_NULLPTR*/)
{
    auto const viewportRect = _viewport->getRect();

    std::lock_guard<std::mutex> lock(_mutex);
    if (_imageOfVisibleRect) {
        painter->drawImage(
            std::max(0.0f, static_cast<float>(viewportRect.x())),
            std::max(0.0f, static_cast<float>(viewportRect.y())),
            *_imageOfVisibleRect);
    }
}

