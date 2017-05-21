 #ifndef SHAPEUNIVERSE_H
#define SHAPEUNIVERSE_H

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector2D>

#include "model/Definitions.h"
#include "model/Entities/CellTO.h"
#include "gui/Definitions.h"

class ShapeUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    ShapeUniverse (QObject *parent = nullptr);
	virtual ~ShapeUniverse();

	virtual void init(SimulationController* controller, ViewportInfo* viewport);
	virtual void activate();
	virtual void deactivate();
	virtual void requestData();

private:
	void retrieveAndDisplayData();

	SimulationAccess* _simAccess = nullptr;
	SimulationController* _controller = nullptr;
	ViewportInfo* _viewport = nullptr;

	GraphicsItemManager* _items = nullptr;
};

#endif // SHAPEUNIVERSE_H
