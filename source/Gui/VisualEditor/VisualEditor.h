#pragma once

#include <QWidget>
#include <QVector2D>
#include <QMatrix>

#include "Model/Api/Definitions.h"
#include "gui/Definitions.h"

namespace Ui {
	class VisualEditor;
}

class VisualEditor : public QWidget
{
    Q_OBJECT
public:
    VisualEditor(QWidget *parent = 0);
    virtual ~VisualEditor();

	virtual void init(Notifier* notifier, SimulationController* controller, DataManipulator* manipulator);

	virtual void refresh();

	virtual void setActiveScene(ActiveScene activeScene);
	virtual QVector2D getViewCenterWithIncrement ();
	virtual QGraphicsView* getGraphicsView ();
	virtual qreal getZoomFactor ();

public:
    Q_SLOT void zoomIn ();
	Q_SLOT void zoomOut ();

private:
    Ui::VisualEditor *ui;

	SimulationController* _controller = nullptr;
    PixelUniverseView* _pixelUniverse = nullptr;
    ItemUniverseView* _itemUniverse = nullptr;
	ViewportController* _viewport = nullptr;

	ActiveScene _activeScene = ActiveScene::PixelScene;

    bool _pixelUniverseInit = false;
    bool _shapeUniverseInit = false;

    qreal _posIncrement = 0.0;

    bool _screenUpdatePossible = true;
};





