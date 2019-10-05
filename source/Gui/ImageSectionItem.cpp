#include <QPainter>

#include "ImageSectionItem.h"
#include "ViewportInterface.h"

ImageSectionItem::ImageSectionItem(ViewportInterface* viewport, QRectF const& boundingRect)
    : QGraphicsItem(), _viewport(viewport), _boundingRect(boundingRect)
{
    auto const viewportRect = _viewport->getRect();
    _imageOfVisibleRect = boost::make_shared<QImage>(viewportRect.width(), viewportRect.height(), QImage::Format_RGB32);
    _imageOfVisibleRect->fill(QColor(0, 0, 0));
}

ImageSectionItem::~ImageSectionItem()
{
}

QImagePtr ImageSectionItem::getImageOfVisibleRect()
{
    //resize image?
    IntVector2D const viewportSize{ static_cast<int>(_viewport->getRect().width()), static_cast<int>(_viewport->getRect().height()) };
    if (_imageOfVisibleRect->width() != viewportSize.x || _imageOfVisibleRect->height() != viewportSize.y) {
        _imageOfVisibleRect = boost::make_shared<QImage>(viewportSize.x, viewportSize.y, QImage::Format_RGB32);
        _imageOfVisibleRect->fill(QColor(0, 0, 0));
    }

    return _imageOfVisibleRect;
}

QRectF ImageSectionItem::boundingRect() const
{
    return _boundingRect;
}

void ImageSectionItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget /*= Q_NULLPTR*/)
{
    auto const viewportRect = _viewport->getRect();
    painter->drawImage(viewportRect.x(), viewportRect.y(), *_imageOfVisibleRect);
}

