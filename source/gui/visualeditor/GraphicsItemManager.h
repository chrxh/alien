#ifndef GRAPHICSITEMMANAGER_H
#define GRAPHICSITEMMANAGER_H

#include "gui/Definitions.h"
#include "Model/Definitions.h"

#include "GraphicsItemConfig.h"
#include "SelectedItems.h"

class GraphicsItemManager
	: public QObject
{
	Q_OBJECT
public:
	GraphicsItemManager(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GraphicsItemManager() = default;

	virtual void init(QGraphicsScene* scene, ViewportInterface* viewport, SimulationParameters* parameters);

	virtual void activate(IntVector2D size);
	virtual void update(DataDescription const &desc);

	virtual void setSelection(list<QGraphicsItem*> const &items);

private:
	template<typename IdType, typename ItemType, typename DescriptionType>
	void updateEntities(vector<TrackerElement<DescriptionType>> const &desc, map<IdType, ItemType*>& itemsByIds
		, map<IdType, ItemType*>& newItemsByIds);
	void updateConnections(vector<TrackerElement<CellDescription>> const &desc
		, map<uint64_t, CellDescription> const &cellsByIds
		, map<set<uint64_t>, CellConnectionGraphicsItem*>& connectionsByIds
		, map<set<uint64_t>, CellConnectionGraphicsItem*>& newConnectionsByIds);
		
	QGraphicsScene *_scene = nullptr;
	ViewportInterface *_viewport = nullptr;
	SimulationParameters *_parameters = nullptr;
	GraphicsItemConfig *_config = nullptr;

	map<uint64_t, CellGraphicsItem*> _cellsByIds;
	map<uint64_t, ParticleGraphicsItem*> _particlesByIds;
	map<set<uint64_t>, CellConnectionGraphicsItem*> _connectionsByIds;
	map<uint64_t, uint64_t> _clusterIdsByCellIds;
	SelectedItems _selectedItems;
};

#endif // GRAPHICSITEMMANAGER_H
