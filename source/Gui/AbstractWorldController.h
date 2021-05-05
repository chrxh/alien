#pragma once

#include <QObject>

#include "Definitions.h"

class AbstractWorldController : public QObject
{
    Q_OBJECT
public:
    AbstractWorldController(SimulationViewWidget* simulationViewWidget, QObject* parent = nullptr);
    virtual ~AbstractWorldController() = default;

    virtual void connectView() = 0;
    virtual void disconnectView() = 0;
    virtual void refresh() = 0;

    virtual bool isActivated() const = 0;
    virtual void activate(double zoomFactor) = 0;

    virtual double getZoomFactor() const = 0;
    virtual void setZoomFactor(double zoomFactor) = 0;
    virtual void setZoomFactor(double zoomFactor, IntVector2D const& viewPos) = 0;

    Q_SIGNAL void startContinuousZoomIn(IntVector2D const& viewPos);
    Q_SIGNAL void startContinuousZoomOut(IntVector2D const& viewPos);
    Q_SIGNAL void endContinuousZoom();

    virtual QVector2D getCenterPositionOfScreen() const = 0;
    virtual void centerTo(QVector2D const& worldPosition) = 0;
    virtual void centerTo(QVector2D const& worldPosition, IntVector2D const& viewPos) = 0;

protected:
    SimulationViewWidget* _simulationViewWidget = nullptr;
};