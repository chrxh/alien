#pragma once

#include <QWidget>
#include <QVector2D>
#include <QMatrix>

#include "ModelBasic/Definitions.h"
#include "Definitions.h"

namespace Ui {
	class SimulationViewWidget;
}

class SimulationViewWidget : public QWidget
{
    Q_OBJECT
public:
    SimulationViewWidget(QWidget *parent = 0);
    virtual ~SimulationViewWidget();

	void init(Notifier* notifier, SimulationController* controller, SimulationAccess* access, DataRepository* manipulator);
    
    void connectView();
    void disconnectView();
    void refresh();

    ActiveView getActiveView() const;
    void setActiveScene(ActiveView activeScene);

    double getZoomFactor();
    void setZoomFactor(double factor);

	QVector2D getViewCenterWithIncrement ();

	void toggleCenterSelection(bool value);

    Q_SIGNAL void zoomFactorChanged(double factor);

private:
    void setStartupScene();
    UniverseView* getActiveUniverseView() const;
    UniverseView* getView(ActiveView activeView) const;

    Ui::SimulationViewWidget *ui;

	SimulationController* _controller = nullptr;

    PixelUniverseView* _pixelUniverse = nullptr;
    VectorUniverseView* _vectorUniverse = nullptr;
    ItemUniverseView* _itemUniverse = nullptr;

    qreal _posIncrement = 0.0;
};





