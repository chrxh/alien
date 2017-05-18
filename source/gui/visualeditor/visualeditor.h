#ifndef MACROEDITOR_H
#define MACROEDITOR_H

#include <QWidget>
#include <QVector2D>
#include <QMatrix>

#include "model/Definitions.h"
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

	void init(SimulationController* controller);
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
    PixelUniverse* _pixelUniverse = nullptr;
    ShapeUniverse* _shapeUniverse = nullptr;
	ViewportController* _viewport = nullptr;

    bool _pixelUniverseInit = false;
    bool _shapeUniverseInit = false;

    qreal _posIncrement = 0.0;

    bool _screenUpdatePossible = true;
};

#endif // MACROEDITOR_H









