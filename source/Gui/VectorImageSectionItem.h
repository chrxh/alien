#pragma once

#include <QGraphicsItem>

#include "Definitions.h"

class VectorImageSectionItem
    : public QGraphicsItem
{
public:
    VectorImageSectionItem(ViewportInterface* viewport, IntVector2D const& universeSize, std::mutex& mutex);
    ~VectorImageSectionItem();

    QImagePtr getImageOfVisibleRect();
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = Q_NULLPTR) override;
    void setZoomFactor(double zoom);

private:
    QImagePtr _imageOfVisibleRect = nullptr;
    ViewportInterface* _viewport = nullptr;
    IntVector2D _universeSize;
    double _zoom = 0;
    std::mutex& _mutex;
};
