#pragma once

#include <QGraphicsItem>

#include "Definitions.h"

class VectorImageSectionItem
    : public QGraphicsItem
{
public:
    VectorImageSectionItem(ViewportInterface* viewport, QRectF const& boundingRect, int zoom, std::mutex& mutex);
    ~VectorImageSectionItem();

    QImagePtr getImageOfVisibleRect();
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = Q_NULLPTR) override;

private:
    QImagePtr _imageOfVisibleRect = nullptr;
    ViewportInterface* _viewport = nullptr;
    QRectF _boundingRect;
    int _zoom = 0;
    std::mutex& _mutex;
};
