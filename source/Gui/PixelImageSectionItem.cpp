#include <QPainter>

#include "PixelImageSectionItem.h"
#include "ViewportInterface.h"

PixelImageSectionItem::PixelImageSectionItem(ViewportInterface* viewport, QRectF const& boundingRect, std::mutex& mutex)
    : QGraphicsItem(), _viewport(viewport), _boundingRect(boundingRect), _mutex(mutex)
{
    auto const viewportRect = _viewport->getRect();
    _imageOfVisibleRect = boost::make_shared<QImage>(viewportRect.width(), viewportRect.height(), QImage::Format_RGB32);
    _imageOfVisibleRect->fill(QColor(0, 0, 0));
}

PixelImageSectionItem::~PixelImageSectionItem()
{
}

QImagePtr PixelImageSectionItem::getImageOfVisibleRect()
{
    //resize image?
    auto rect = _viewport->getRect();
    IntVector2D viewportSize{ static_cast<int>(rect.width()), static_cast<int>(rect.height()) };
    viewportSize.x = std::min(static_cast<int>(_boundingRect.width()), viewportSize.x);
    viewportSize.y = std::min(static_cast<int>(_boundingRect.height()), viewportSize.y);
    if (_imageOfVisibleRect->width() != viewportSize.x || _imageOfVisibleRect->height() != viewportSize.y) {
        _imageOfVisibleRect = boost::make_shared<QImage>(viewportSize.x, viewportSize.y, QImage::Format_ARGB32);
        _imageOfVisibleRect->fill(QColor(0, 0, 0));
    }

    return _imageOfVisibleRect;
}

QRectF PixelImageSectionItem::boundingRect() const
{
    return _boundingRect;
}

void PixelImageSectionItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget /*= Q_NULLPTR*/)
{
    auto const viewportRect = _viewport->getRect();

    std::lock_guard<std::mutex> lock(_mutex);

    painter->drawImage(
        std::max(0.0f, static_cast<float>(viewportRect.x())),
        std::max(0.0f, static_cast<float>(viewportRect.y())),
        *_imageOfVisibleRect);
}

