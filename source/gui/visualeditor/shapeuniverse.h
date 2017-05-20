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
	virtual void setActive();

private:
	void retrieveAndDisplayData();

	SimulationAccess* _simAccess = nullptr;
	SimulationContextApi* _context = nullptr;
	ViewportInfo* _viewport = nullptr;

	GraphicsItems* _items = nullptr;
};

#endif // SHAPEUNIVERSE_H
