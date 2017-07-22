#include <QGraphicsScene>

#include "Model/Entities/Descriptions.h"
#include "Gui/settings.h"
#include "Gui/visualeditor/ViewportInterface.h"

#include "ItemManager.h"
#include "DescriptionManager.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "CellConnectionItem.h"

void ItemManager::init(QGraphicsScene * scene, ViewportInterface* viewport, SimulationParameters* parameters)
{
	auto config = new ItemConfig();
	auto selectedItems = new SelectedItems();
	auto descManager = new DescriptionManager();

	_scene = scene;
	_viewport = viewport;
	_parameters = parameters;
	SET_CHILD(_config, config);
	SET_CHILD(_selectedItems, selectedItems);
	SET_CHILD(_descManager, descManager);

	_config->init(parameters);
}

void ItemManager::activate(IntVector2D size)
{
	_scene->clear();
	_scene->setSceneRect(0, 0, size.x*GRAPHICS_ITEM_SIZE, size.y*GRAPHICS_ITEM_SIZE);
	_cellsByIds.clear();
	_particlesByIds.clear();
}

template<typename IdType, typename ItemType, typename DescriptionType>
void ItemManager::updateEntities(vector<TrackerElement<DescriptionType>> const &desc
	, map<IdType, ItemType*>& itemsByIds, map<IdType, ItemType*>& newItemsByIds)
{
	for (auto const &descElementT : desc) {
		auto const &descElement = descElementT.getValue();
		auto it = itemsByIds.find(descElement.id);
		if (it != itemsByIds.end()) {
			auto item = it->second;
			item->update(descElement);
			newItemsByIds[descElement.id] = item;
			itemsByIds.erase(it);
		}
		else {
			ItemType* newItem = new ItemType(_config, descElement);
			_scene->addItem(newItem);
			newItemsByIds[descElement.id] = newItem;
		}
	}
}

void ItemManager::updateConnections(DataDescription const &data
	, map<uint64_t, CellDescription> const &cellsByIds)
{
	map<set<uint64_t>, CellConnectionItem*> newConnectionsByIds;
	for (auto const &clusterT : data.clusters) {
		auto const &cluster = clusterT.getValue();
		for (auto const &cellT : cluster.cells) {
			auto const &cellD = cellT.getValue();
			if (!cellD.connectingCells.isInitialized()) {
				continue;
			}
			for (uint64_t connectingCellId : cellD.connectingCells.getValue()) {
				auto cellIt = cellsByIds.find(connectingCellId);
				if (cellIt == cellsByIds.end()) {
					continue;
				}
				set<uint64_t> id;
				id.insert(cellD.id);
				id.insert(connectingCellId);
				if (newConnectionsByIds.find(id) != newConnectionsByIds.end()) {
					continue;
				}
				auto connectionIt = _connectionsByIds.find(id);
				if (connectionIt != _connectionsByIds.end()) {
					CellConnectionItem* connection = connectionIt->second;
					connection->update(cellD, cellIt->second);
					newConnectionsByIds[id] = connection;
					_connectionsByIds.erase(connectionIt);
				}
				else {
					CellConnectionItem* newConnection = new CellConnectionItem(_config, cellD, cellIt->second);
					_scene->addItem(newConnection);
					newConnectionsByIds[id] = newConnection;
				}
			}
		}
	}
	for (auto const& connectionById : _connectionsByIds) {
		delete connectionById.second;
	}
	_connectionsByIds = newConnectionsByIds;
}

namespace
{
	void getCellsByIds(DataDescription const &desc, map<uint64_t, CellDescription> &result)
	{
		for (auto const &clusterT : desc.clusters) {
			auto const &clusterD = clusterT.getValue();
			for (auto const &cellT : clusterD.cells) {
				auto const &cellD = cellT.getValue();
				result[cellD.id] = cellD;
			}
		}
	}

	void getClusterIdsByCellIds(DataDescription const &desc, map<uint64_t, uint64_t> &result)
	{
		for (auto const &clusterT : desc.clusters) {
			auto const &cluster = clusterT.getValue();
			for (auto const &cellT : cluster.cells) {
				auto const &cell = cellT.getValue();
				result[cell.id] = cluster.id;
			}
		}
	}

}

void ItemManager::update(DataDescription const &data)
{
	_descManager->setData(data);
	_viewport->setModeToNoUpdate();

	map<uint64_t, CellDescription> cellDescByIds;
	getCellsByIds(data, cellDescByIds);
	getClusterIdsByCellIds(data, _clusterIdsByCellIds);

	map<uint64_t, CellItem*> newCellsByIds;
	for (auto const &clusterT : data.clusters) {
		auto const &cluster = clusterT.getValue();
		updateEntities(cluster.cells, _cellsByIds, newCellsByIds);
	}
	for (auto const& cellById : _cellsByIds) {
		delete cellById.second;
	}
	_cellsByIds = newCellsByIds;

	updateConnections(data, cellDescByIds);

	map<uint64_t, ParticleItem*> newParticlesByIds;
	updateEntities(data.particles, _particlesByIds, newParticlesByIds);
	for (auto const& particleById : _particlesByIds) {
		delete particleById.second;
	}
	_particlesByIds = newParticlesByIds;

	_viewport->setModeToUpdate();
}

void ItemManager::setSelection(list<QGraphicsItem*> const &items)
{
	_selectedItems->update(items, _clusterIdsByCellIds, _cellsByIds);
	_scene->update();
}

void ItemManager::moveSelection(QVector2D const &delta)
{
	_viewport->setModeToNoUpdate();
	//1. SelectedItems: change cell descriptions of selection
	//2. SelectedItems: move graphic items
	//3. reconnect cells
	//4. update connections
	_selectedItems->move(delta);
/*
	for (set<uint64_t> connectionId : _selectedItems->getConnectionIds()) {
		auto connectionIdIt = connectionId.begin();
		auto const &cellDesc1 = _cellsByIds.at(*(connectionIdIt++))->getDescription();
		auto const &cellDesc2 = _cellsByIds.at(*connectionIdIt)->getDescription();
		_connectionsByIds.at(connectionId)->update(cellDesc1, cellDesc2);
	}
*/
	map<uint64_t, CellDescription> cellDescByIds;
	getCellsByIds(_descManager->getDataRef(), cellDescByIds);	//<-- should be done by DescManager
	updateConnections(_descManager->getDataRef(), cellDescByIds);
	_viewport->setModeToUpdate();
};
