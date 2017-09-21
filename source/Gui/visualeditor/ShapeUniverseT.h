#ifndef SHAPEUNIVERSE_H
#define SHAPEUNIVERSE_H

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector2D>

#include "Model/Definitions.h"
#include "Model/Entities/CellTO.h"
#include "Model/Entities/Descriptions.h"
#include "gui/Definitions.h"

class ShapeUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    ShapeUniverse (QObject *parent = nullptr);
	virtual ~ShapeUniverse();

	virtual void init(SimulationController* controller, SimulationAccess* access, ViewportInterface* viewport
		, DataEditorContext* dataEditorContext);
	virtual void activate();
	virtual void deactivate();
	virtual void requestData();

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* e);
	void mouseMoveEvent(QGraphicsSceneMouseEvent* e);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* e);

private:
	void retrieveAndDisplayData();

	SimulationAccess* _simAccess = nullptr;
	SimulationController* _controller = nullptr;
	ViewportInterface* _viewport = nullptr;
	CellConnector* _connector = nullptr;

	ItemManager* _itemManager = nullptr;
	VisualDescription* _visualDesc = nullptr;

	DataDescription _savedDataBeforeMovement;
	DataEditorContext* _dataEditorContext;
};

#endif // SHAPEUNIVERSE_H
