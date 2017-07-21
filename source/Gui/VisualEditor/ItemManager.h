#ifndef GRAPHICSITEMMANAGER_H
#define GRAPHICSITEMMANAGER_H

#include "gui/Definitions.h"
#include "Model/Definitions.h"

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
	virtual void update(DataDescription const &desc);

	virtual void setSelection(list<QGraphicsItem*> const &items);
	virtual void moveSelection(QVector2D const &delta);

private:
	template<typename IdType, typename ItemType, typename DescriptionType>
	void updateEntities(vector<TrackerElement<DescriptionType>> const &desc, map<IdType, ItemType*>& itemsByIds
		, map<IdType, ItemType*>& newItemsByIds);
	void updateConnections(vector<TrackerElement<CellDescription>> const &desc
		, map<uint64_t, CellDescription> const &cellsByIds
		, map<set<uint64_t>, CellConnectionItem*>& connectionsByIds
		, map<set<uint64_t>, CellConnectionItem*>& newConnectionsByIds);
		
	QGraphicsScene *_scene = nullptr;
	ViewportInterface *_viewport = nullptr;
	SimulationParameters *_parameters = nullptr;
	ItemConfig *_config = nullptr;
	SelectedItems* _selectedItems = nullptr;

	map<uint64_t, CellItem*> _cellsByIds;
	map<uint64_t, ParticleItem*> _particlesByIds;
	map<set<uint64_t>, CellConnectionItem*> _connectionsByIds;
	map<uint64_t, uint64_t> _clusterIdsByCellIds;
};

#endif // GRAPHICSITEMMANAGER_H
