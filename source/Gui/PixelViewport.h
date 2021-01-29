#pragma once

#include "ViewportInterface.h"

class PixelViewport : public ViewportInterface
{
    Q_OBJECT
public:
    PixelViewport(QGraphicsView* view, QObject* parent = nullptr);
    virtual ~PixelViewport() = default;

    QRectF getRect() const override;

private:
    QGraphicsView* _view;
};