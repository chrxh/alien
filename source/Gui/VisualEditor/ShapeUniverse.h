#pragma once

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector2D>

#include "Model/Api/Definitions.h"
#include "Model/Api/Descriptions.h"
#include "Gui/Definitions.h"
#include "Gui/DataManipulator.h"

class ShapeUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    ShapeUniverse (QObject *parent = nullptr);
	virtual ~ShapeUniverse();

	virtual void init(Notifier* notifier, SimulationController* controller, DataManipulator* manipulator, ViewportInterface* viewport);
	virtual void activate();
	virtual void deactivate();
	virtual void requestData();

protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* e);
	void mouseMoveEvent(QGraphicsSceneMouseEvent* e);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* e);

private:
	Q_SLOT void receivedNotifications(set<Receiver> const& targets);

	struct Selection
	{
		list<uint64_t> cellIds;
		list<uint64_t> particleIds;
	};
	Selection getSelectionFromItems(std::list<QGraphicsItem*> const &items) const;
	void delegateSelection(Selection const& selection);
	void startMarking(QPointF const& scenePos);

	SimulationController* _controller = nullptr;
	ViewportInterface* _viewport = nullptr;

	ItemManager* _itemManager = nullptr;
	DataManipulator* _manipulator = nullptr;
	Notifier* _notifier = nullptr;
};
