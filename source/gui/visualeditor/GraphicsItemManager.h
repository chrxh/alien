#ifndef GRAPHICSITEMMANAGER_H
#define GRAPHICSITEMMANAGER_H

#include "gui/Definitions.h"
#include "Model/Definitions.h"

#include "GraphicsItemConfig.h"

class GraphicsItemManager
	: public QObject
{
	Q_OBJECT
public:
	GraphicsItemManager(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GraphicsItemManager() = default;

	virtual void init(QGraphicsScene* scene);

	virtual void activate(IntVector2D size);
	virtual void update(DataDescription const &desc);

private:
	template<typename ItemType, typename DescriptionType>
	void updateItems(vector<TrackerElement<DescriptionType>> const &desc, unordered_map<uint64_t, ItemType*>& itemsByIds
		, unordered_map<uint64_t, ItemType*>& newItemsByIds);
		
	QGraphicsScene* _scene = nullptr;
	GraphicsItemConfig _config;

	unordered_map<uint64_t, CellGraphicsItem*> _cellsByIds;
	unordered_map<uint64_t, ParticleGraphicsItem*> _particlesByIds;
};

#endif // GRAPHICSITEMMANAGER_H
