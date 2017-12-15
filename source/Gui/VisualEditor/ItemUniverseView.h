#pragma once

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector2D>

#include "Model/Api/Definitions.h"
#include "Model/Api/Descriptions.h"
#include "Gui/Definitions.h"
#include "Gui/DataController.h"

class ItemUniverseView : public QGraphicsScene
{
    Q_OBJECT
public:
    ItemUniverseView (QObject *parent = nullptr);
	virtual ~ItemUniverseView() = default;

	virtual void init(Notifier* notifier, SimulationController* controller, DataController* manipulator, ViewportInterface* viewport);
	virtual void activate();
	virtual void deactivate();

	virtual void refresh();


protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* e);
	void mouseMoveEvent(QGraphicsSceneMouseEvent* e);
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* e);

private:
	void requestData();

	Q_SLOT void receivedNotifications(set<Receiver> const& targets);
	Q_SLOT void cellInfoToggled(bool showInfo);
	Q_SLOT void scrolled();

	struct Selection
	{
		list<uint64_t> cellIds;
		list<uint64_t> particleIds;
	};
	Selection getSelectionFromItems(std::list<QGraphicsItem*> const &items) const;
	void delegateSelection(Selection const& selection);
	void startMarking(QPointF const& scenePos);

	list<QMetaObject::Connection> _connections;

	SimulationController* _controller = nullptr;
	ViewportInterface* _viewport = nullptr;

	ItemManager* _itemManager = nullptr;
	DataController* _manipulator = nullptr;
	Notifier* _notifier = nullptr;
};
