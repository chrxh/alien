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
    enum ActiveScene {
        PixelScene,
        ShapeScene
    };

    VisualEditor(QWidget *parent = 0);
    virtual ~VisualEditor();

	void init(SimulationController* controller);

    void reset ();

    void setActiveScene (ActiveScene activeScene);
    QVector2D getViewCenterPosWithInc ();

    void serializeViewMatrix (QDataStream& stream);
    void loadViewMatrix (QDataStream& stream);

    QGraphicsView* getGraphicsView ();

    qreal getZoomFactor ();

public Q_SLOTS:
    void zoomIn ();
    void zoomOut ();

    void metadataUpdated ();
	void toggleInformation(bool on);

private:
    Ui::VisualEditor *ui;

	SimulationController* _controller = nullptr;
    ActiveScene _activeScene;
    PixelUniverse* _pixelUniverse;
    ShapeUniverse* _shapeUniverse;

    bool _pixelUniverseInit = false;
    bool _shapeUniverseInit = false;
    QMatrix _pixelUniverseViewMatrix;
    QMatrix _shapeUniverseViewMatrix;
    int _pixelUniversePosX, _pixelUniversePosY;
    int _shapeUniversePosX, _shapeUniversePosY;

    qreal _posIncrement = 0.0;

    //timer for limiting screen updates
    QTimer* _updateTimer = nullptr;
    bool _screenUpdatePossible = true;
};

#endif // MACROEDITOR_H









