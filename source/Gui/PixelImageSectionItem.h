#pragma once

#include <QGraphicsItem>

#include "Definitions.h"

class PixelImageSectionItem
    : public QGraphicsItem
{
public:
    PixelImageSectionItem(ViewportInterface* viewport, QRectF const& boundingRect, std::mutex& mutex);
    ~PixelImageSectionItem();

    QImagePtr getImageOfVisibleRect();
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = Q_NULLPTR) override;

private:
    QImagePtr _imageOfVisibleRect = nullptr;
    ViewportInterface* _viewport = nullptr;
    QRectF _boundingRect;
    std::mutex& _mutex;
};
