#pragma once

#include "EngineInterface/Definitions.h"
#include "Gui/Definitions.h"

#include "ItemConfig.h"

class ItemManager
	: public QObject
{
	Q_OBJECT
public:
	ItemManager(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~ItemManager() = default;

	virtual void init(QGraphicsScene* scene, ViewportInterface* viewport, SimulationParameters const& parameters);

	virtual void activate(IntVector2D size);
	virtual void update(DataRepository* visualDesc);

	virtual void setMarkerItem(QPointF const &upperLeft, QPointF const &lowerRight);
	virtual void setMarkerLowerRight(QPointF const &lowerRight);
	virtual void deleteMarker();
	virtual bool isMarkerActive() const;
    virtual QList<QGraphicsItem*> getItemsWithinMarker() const;

	virtual void toggleCellInfo(bool showInfo);

private:
	void updateCells(DataRepository* visualDesc);
	void updateConnections(DataRepository* visualDesc);
	void updateParticles(DataRepository* visualDesc);
		
	QGraphicsScene* _scene = nullptr;
	ViewportInterface* _viewport = nullptr;
	SimulationParameters _parameters;
	ItemConfig* _config = nullptr;

	map<uint64_t, CellItem*> _cellsByIds;
	map<uint64_t, ParticleItem*> _particlesByIds;
	map<set<uint64_t>, CellConnectionItem*> _connectionsByIds;
	MarkerItem* _marker = nullptr;
};

