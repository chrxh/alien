#pragma once

#include "Model/Definitions.h"
#include "Gui/Definitions.h"

#include "ItemConfig.h"
#include "SelectedItems.h"

class ItemManager
	: public QObject
{
	Q_OBJECT
public:
	ItemManager(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~ItemManager() = default;

	virtual void init(QGraphicsScene* scene, ViewportInterface* viewport, SimulationParameters* parameters);

	virtual void activate(IntVector2D size);
	virtual void update(VisualDescription* visualDesc);

	virtual void setMarkerItem(QPointF const &upperLeft, QPointF const &lowerRight);
	virtual void setMarkerLowerRight(QPointF const &lowerRight);
	virtual void deleteMarker();

private:
	void updateCells(VisualDescription* visualDesc);
	void updateConnections(VisualDescription* visualDesc);
	void updateParticles(VisualDescription* visualDesc);
		
	QGraphicsScene *_scene = nullptr;
	ViewportInterface *_viewport = nullptr;
	SimulationParameters *_parameters = nullptr;
	ItemConfig *_config = nullptr;

	map<uint64_t, CellItem*> _cellsByIds;
	map<uint64_t, ParticleItem*> _particlesByIds;
	map<set<uint64_t>, CellConnectionItem*> _connectionsByIds;
	MarkerItem* _marker = nullptr;
};

