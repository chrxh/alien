#pragma once

#include <QObject>

#include "Definitions.h"

class UniverseView : public QObject
{
    Q_OBJECT
public:
    UniverseView(QGraphicsView* graphicsView, QObject* parent = nullptr);
    virtual ~UniverseView() = default;

    virtual void connectView() = 0;
    virtual void disconnectView() = 0;
    virtual void refresh() = 0;

    virtual bool isActivated() const = 0;
    virtual void activate(double zoomFactor) = 0;

    virtual double getZoomFactor() const = 0;
    virtual void setZoomFactor(double zoomFactor) = 0;
    virtual void setZoomFactor(double zoomFactor, QVector2D const& fixedPos) = 0;

    Q_SIGNAL void startContinuousZoomIn(QVector2D const& worldPos);
    Q_SIGNAL void startContinuousZoomOut(QVector2D const& worldPos);
    Q_SIGNAL void endContinuousZoom();

    virtual QVector2D getCenterPositionOfScreen() const = 0;
    virtual void centerTo(QVector2D const& position) = 0;

protected:
    QGraphicsView* _graphicsView = nullptr;
};