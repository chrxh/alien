#pragma once

#include "ViewportInterface.h"

class ItemViewport : public ViewportInterface
{
    Q_OBJECT
public:
    ItemViewport(QGraphicsView* view, QObject* parent = nullptr);
    virtual ~ItemViewport() = default;

    QRectF getRect() const override;

private:
    QGraphicsView* _view;
};