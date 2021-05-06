#pragma once

#include <QWidget>

#include <QVector2D>

#include "EngineInterface/Definitions.h"

#include "SimulationViewSettings.h"
#include "Definitions.h"

class SimulationViewController : public QObject
{
    Q_OBJECT
public:
    SimulationViewController(QWidget* parent = nullptr);
    virtual ~SimulationViewController() = default;

    QWidget* getWidget() const;

    void
    init(Notifier* notifier, SimulationController* controller, SimulationAccess* access, DataRepository* manipulator);

    void setSettings(SimulationViewSettings const& settings);

    void connectView();
    void disconnectView();
    void refresh();

    ActiveView getActiveView() const;
    void setActiveScene(ActiveView activeScene);

    double getZoomFactor();
    void setZoomFactor(double factor);
    void setZoomFactor(double factor, IntVector2D const& viewPos);

    QVector2D getViewCenterWithIncrement();

    void toggleCenterSelection(bool value);

    Q_SIGNAL void continuousZoomIn(IntVector2D const& viewPos);
    Q_SIGNAL void continuousZoomOut(IntVector2D const& viewPos);
    Q_SIGNAL void endContinuousZoom();
    Q_SIGNAL void zoomFactorChanged(double factor);

private:
    AbstractWorldController* getActiveUniverseView() const;
    AbstractWorldController* getView(ActiveView activeView) const;

    SimulationController* _controller = nullptr;

    SimulationViewWidget* _simulationViewWidget = nullptr;

    OpenGLWorldController* _openGLWorld = nullptr;
    ItemWorldController* _itemWorld = nullptr;

    qreal _posIncrement = 0.0;
};
