#pragma once

#include <QGraphicsItem>

#include "Definitions.h"

class ImageSectionItem
    : public QGraphicsItem
{
public:
    ImageSectionItem(ViewportInterface* viewport, QRectF const& boundingRect, std::mutex& mutex);
    ~ImageSectionItem();

    QImagePtr getImageOfVisibleRect();
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = Q_NULLPTR) override;

private:
    QImagePtr _imageOfVisibleRect = nullptr;
    ViewportInterface* _viewport = nullptr;
    QRectF _boundingRect;
    std::mutex& _mutex;
};
