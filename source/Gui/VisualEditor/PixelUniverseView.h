#pragma once

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "gui/Definitions.h"
#include "Model/Api/Definitions.h"
#include "Model/Api/Descriptions.h"

class PixelUniverseView : public QGraphicsScene
{
    Q_OBJECT
public:
    PixelUniverseView(QObject* parent = nullptr);
    virtual ~PixelUniverseView();

	virtual void init(SimulationController* controller, DataManipulator* manipulator, ViewportInterface* viewport);
	virtual void activate();
	virtual void deactivate();

private:
    Q_SLOT void requestData();
	Q_SLOT void retrieveAndDisplayData();
	Q_SLOT void scrolling();

	DataManipulator* _manipulator = nullptr;
	SimulationController* _controller = nullptr;
	ViewportInterface* _viewport = nullptr;
    QGraphicsPixmapItem* _pixmap = nullptr;
    QImage* _image = nullptr;

};

