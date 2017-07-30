#pragma once

#include <QWidget>
#include <QVector2D>
#include <QMatrix>

#include "Model/Definitions.h"
#include "gui/Definitions.h"

namespace Ui {
class VisualEditorT;
}

class VisualEditorT : public QWidget
{
    Q_OBJECT
public:
    VisualEditorT(QWidget *parent = 0);
    virtual ~VisualEditorT();

	void init(SimulationController* controller, SimulationAccess* access);
    void reset();

	void setActiveScene(ActiveScene activeScene);
    QVector2D getViewCenterWithIncrement ();
    QGraphicsView* getGraphicsView ();
    qreal getZoomFactor ();

public Q_SLOTS:
    void zoomIn ();
    void zoomOut ();

private:
    Ui::VisualEditor *ui;

	SimulationController* _controller = nullptr;
    PixelUniverseT* _pixelUniverse = nullptr;
    ShapeUniverseT* _shapeUniverse = nullptr;
	ViewportController* _viewport = nullptr;

	ActiveScene _activeScene = ActiveScene::PixelScene;

    bool _pixelUniverseInit = false;
    bool _shapeUniverseInit = false;

    qreal _posIncrement = 0.0;

    bool _screenUpdatePossible = true;
};





