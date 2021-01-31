#pragma once

#include "ViewportInterface.h"

class VectorViewport : public ViewportInterface
{
    Q_OBJECT
public:
    VectorViewport(QGraphicsView* view, QObject* parent = nullptr);
    virtual ~VectorViewport() = default;

    void setZoomFactor(double zoomFactor);
    QRectF getRect() const override;

private:
    QGraphicsView* _view;
    double _zoomFactor = 0.0;
};