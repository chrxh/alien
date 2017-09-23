#pragma once

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

	struct Selection
	{
		list<uint64_t> cellIds;
		list<uint64_t> particleIds;
	};
	Selection getSelectionFromItems(std::list<QGraphicsItem*> const &items) const;
	void delegateSelection(Selection const& selection);
	void startMarking(QPointF const& scenePos);

	SimulationAccess* _simAccess = nullptr;
	SimulationController* _controller = nullptr;
	ViewportInterface* _viewport = nullptr;
	CellConnector* _connector = nullptr;

	ItemManager* _itemManager = nullptr;
	VisualDescription* _visualDesc = nullptr;

	DataDescription _savedDataBeforeMovement;
	DataEditorContext* _dataEditorContext;
};
