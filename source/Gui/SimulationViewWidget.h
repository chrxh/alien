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

	void init(Notifier* notifier, SimulationController* controller
		, SimulationAccess* access, DataRepository* manipulator);

	void refresh();

	void setActiveScene(ActiveScene activeScene);
	QVector2D getViewCenterWithIncrement ();
	double getZoomFactor ();
    void scrollToPos(QVector2D const& pos);

    void zoom (double factor);
	void toggleCenterSelection(bool value);

private:
    Ui::SimulationViewWidget *ui;

	SimulationController* _controller = nullptr;
    PixelUniverseView* _pixelUniverse = nullptr;
    VectorUniverseView* _vectorUniverse = nullptr;
    ItemUniverseView* _itemUniverse = nullptr;
	ViewportController* _viewport = nullptr;

	ActiveScene _activeScene = ActiveScene::PixelScene;

    bool _pixelUniverseInit = false;
    bool _itemUniverseInit = false;

    qreal _posIncrement = 0.0;
};





