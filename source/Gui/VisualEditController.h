#pragma once

#include <QWidget>
#include <QVector2D>
#include <QMatrix>

#include "ModelBasic/Definitions.h"
#include "Gui/Definitions.h"

namespace Ui {
	class VisualEditController;
}

class VisualEditController : public QWidget
{
    Q_OBJECT
public:
    VisualEditController(QWidget *parent = 0);
    virtual ~VisualEditController();

	virtual void init(Notifier* notifier, SimulationController* controller
		, SimulationAccess* access, DataRepository* manipulator);

	virtual void refresh();

	virtual void setActiveScene(ActiveScene activeScene);
	virtual QVector2D getViewCenterWithIncrement ();
	virtual QGraphicsView* getGraphicsView ();
	virtual double getZoomFactor ();

    virtual void zoom (double factor);
	virtual void toggleCenterSelection(bool value);

private:
    Ui::VisualEditController *ui;

	SimulationController* _controller = nullptr;
    PixelUniverseView* _pixelUniverse = nullptr;
    ItemUniverseView* _itemUniverse = nullptr;
	ViewportController* _viewport = nullptr;

	ActiveScene _activeScene = ActiveScene::PixelScene;

    bool _pixelUniverseInit = false;
    bool _shapeUniverseInit = false;

    qreal _posIncrement = 0.0;
};





