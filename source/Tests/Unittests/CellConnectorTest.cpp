#include <gtest/gtest.h>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/Settings.h"
#include "Model/SimulationController.h"
#include "Model/CellConnector.h"
#include "Model/Context/SimulationContext.h"
#include "Model/Context/SimulationParameters.h"

#include "tests/Predicates.h"

class CellConnectorTest : public ::testing::Test
{
public:
	CellConnectorTest();
	~CellConnectorTest();

protected:
	bool isCellAdded(CellClusterDescription const &cluster, uint64_t cellId) const;
	bool isCellModified(CellClusterDescription const &cluster, uint64_t cellId) const;
	bool isCellDeleted(CellClusterDescription const &cluster, uint64_t cellId) const;
	bool areAllClustersDeletedExcept(set<uint64_t> const &clusterIds) const;
	bool areAllCellsDeletedExcept(CellClusterDescription const &cluster, set<uint64_t> const &cellIds) const;

	SimulationController* _controller = nullptr;
	SimulationParameters* _parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;
	CellConnector* _connector = nullptr;

	IntVector2D _universeSize{ 600, 300 };

	DataDescription _data;
	DescriptionNavigationMaps _navi;
};

CellConnectorTest::CellConnectorTest()
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto symbols = facade->buildDefaultSymbolTable();
	_parameters = facade->buildDefaultSimulationParameters();
	_controller = facade->buildSimulationController(1, { 1,1 }, _universeSize, symbols, _parameters);
	auto context = static_cast<SimulationContext*>(_controller->getContext());
	_numberGen = context->getNumberGenerator();
	_connector = facade->buildCellConnector(context);
}

CellConnectorTest::~CellConnectorTest()
{
	delete _controller;
	delete _connector;
}

bool CellConnectorTest::isCellAdded(CellClusterDescription const & cluster, uint64_t cellId) const
{
	for (auto const& cellT : cluster.cells) {
		if (cellT->id == cellId && !cellT.isAdded()) {
			return false;
		}
	}
	return true;
}

bool CellConnectorTest::isCellModified(CellClusterDescription const & cluster, uint64_t cellId) const
{
	for (auto const& cellT : cluster.cells) {
		if (cellT->id == cellId && !cellT.isModified()) {
			return false;
		}
	}
	return true;
}

bool CellConnectorTest::isCellDeleted(CellClusterDescription const & cluster, uint64_t cellId) const
{
	for (auto const& cellT : cluster.cells) {
		if (cellT->id == cellId && !cellT.isDeleted()) {
			return false;
		}
	}
	return true;
}

bool CellConnectorTest::areAllClustersDeletedExcept(set<uint64_t> const &clusterIds) const
{
	for (auto const& clusterT : _data.clusters) {
		if (clusterIds.find(clusterT->id) == clusterIds.end()) {
			if (!clusterT.isDeleted()) {
				return false;
			}
		}
		else {
			if (clusterT.isDeleted()) {
				return false;
			}
		}
	}
	return true;
}

bool CellConnectorTest::areAllCellsDeletedExcept(CellClusterDescription const & cluster, set<uint64_t> const & cellIds) const
{
	for (auto const& cellT : cluster.cells) {
		if (cellIds.find(cellT->id) == cellIds.end()) {
			if (!cellT.isDeleted()) {
				return false;
			}
		}
		else {
			if (cellT.isDeleted()) {
				return false;
			}
		}
	}
	return true;
}

TEST_F(CellConnectorTest, testMoveOneCellAway)
{
	vector<uint64_t> cellIds;
	cellIds.push_back(_numberGen->getTag());
	cellIds.push_back(_numberGen->getTag());
	_data.retainCellCluster(CellClusterDescription().setId(_numberGen->getTag()).retainCells(
	{
		CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setConnectingCells({ cellIds[1] }).setMaxConnections(1),
		CellDescription().setPos({ 101, 100 }).setId(cellIds[1]).setConnectingCells({ cellIds[0] }).setMaxConnections(1)
	}));
	_data.clusters[0]->cells[1]->pos.setValue({ 103, 100 });

	_connector->reconnect(_data);

	_navi.update(_data);

	auto cluster0 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[1]));

	ASSERT_EQ(2, _data.clusters.size());
	ASSERT_TRUE(cluster0.isModified());
	ASSERT_TRUE(cluster1.isAdded());

	ASSERT_EQ(2, cluster0->cells.size());
	ASSERT_TRUE(isCellModified(cluster0.getValue(), cellIds[0]));
	ASSERT_TRUE(isCellDeleted(cluster0.getValue(), cellIds[1]));

	ASSERT_EQ(1, cluster1->cells.size());
	ASSERT_TRUE(isCellAdded(cluster0.getValue(), cellIds[1]));
}

TEST_F(CellConnectorTest, testMoveOneCellToAnOtherCluster)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 4; ++i) { 
		cellIds.push_back(_numberGen->getTag());
	}
	_data.retainCellClusters({
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setConnectingCells({ cellIds[1] }).setMaxConnections(1),
			CellDescription().setPos({ 101, 100 }).setId(cellIds[1]).setConnectingCells({ cellIds[0] }).setMaxConnections(1)
		}), 
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 200, 100 }).setId(cellIds[2]).setConnectingCells({ cellIds[3] }).setMaxConnections(2),
			CellDescription().setPos({ 201, 100 }).setId(cellIds[3]).setConnectingCells({ cellIds[2] }).setMaxConnections(1)
		})
	});
	_data.clusters[0]->cells[1]->pos.setValue({ 199, 100 });

	_connector->reconnect(_data);
	
	_navi.update(_data);

	auto cluster0 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[2]));

	ASSERT_EQ(2, _data.clusters.size());
	ASSERT_TRUE(cluster0.isModified());
	ASSERT_TRUE(cluster1.isModified());

	ASSERT_EQ(2, cluster0->cells.size());
	ASSERT_TRUE(isCellModified(cluster0.getValue(), cellIds[0]));
	ASSERT_TRUE(isCellDeleted(cluster0.getValue(), cellIds[1]));

	ASSERT_EQ(3, cluster1->cells.size());
	ASSERT_TRUE(isCellModified(cluster1.getValue(), cellIds[2]));
	ASSERT_TRUE(isCellModified(cluster1.getValue(), cellIds[3]));
	ASSERT_TRUE(isCellAdded(cluster1.getValue(), cellIds[1]));
}

TEST_F(CellConnectorTest, testMoveOneCellToUniteClusters)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 5; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.retainCellClusters({
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setMaxConnections(2),
		}),
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 200, 98 }).setId(cellIds[1]).setConnectingCells({ cellIds[2] }).setMaxConnections(1),
			CellDescription().setPos({ 200, 99 }).setId(cellIds[2]).setConnectingCells({ cellIds[1] }).setMaxConnections(2)
		}),
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 200, 101 }).setId(cellIds[3]).setConnectingCells({ cellIds[4] }).setMaxConnections(2),
			CellDescription().setPos({ 200, 102 }).setId(cellIds[4]).setConnectingCells({ cellIds[3] }).setMaxConnections(1)
		})
	});
	_data.clusters[0]->cells[0]->pos.setValue({ 200, 100 });

	_connector->reconnect(_data);

	_navi.update(_data);

	auto cluster0 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[0]));

	ASSERT_EQ(3, _data.clusters.size());
	ASSERT_TRUE(cluster0.isModified());
	ASSERT_TRUE(areAllClustersDeletedExcept({ cluster0->id }));

	ASSERT_EQ(5, cluster0->cells.size());
	for (int i = 0; i < 5; ++i) {
		ASSERT_TRUE(isCellModified(cluster0.getValue(), cellIds[i]) || isCellAdded(cluster0.getValue(), cellIds[i]));
	}
}

TEST_F(CellConnectorTest, testMoveOneCellToUniteAndDevideClusters)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 5; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.retainCellClusters({
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setMaxConnections(2),
		}),
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 200, 98 }).setId(cellIds[1]).setConnectingCells({ cellIds[2] }).setMaxConnections(1),
			CellDescription().setPos({ 200, 99 }).setId(cellIds[2]).setConnectingCells({ cellIds[1] }).setMaxConnections(2)
		}),
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 200, 101 }).setId(cellIds[3]).setConnectingCells({ cellIds[4] }).setMaxConnections(2),
			CellDescription().setPos({ 200, 102 }).setId(cellIds[4]).setConnectingCells({ cellIds[3] }).setMaxConnections(1)
		})
	});
	_data.clusters[0]->cells[0]->pos.setValue({ 200, 100 });
	_connector->reconnect(_data);
	_navi.update(_data);

	uint64_t clusterIndex = _navi.clusterIndicesByCellIds.at(cellIds[0]);
	uint64_t cellIndex = _navi.cellIndicesByCellIds.at(cellIds[0]);
	_data.clusters[clusterIndex]->cells[cellIndex]->pos.setValue({ 100, 100 });
	_connector->reconnect(_data);
	_navi.update(_data);

	auto cluster0 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[1]));
	auto cluster2 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[3]));

	ASSERT_EQ(5, _data.clusters.size());
	ASSERT_TRUE(areAllClustersDeletedExcept({ cluster0->id, cluster1->id, cluster2->id }));
	ASSERT_TRUE(areAllCellsDeletedExcept(cluster0.getValue(), { cellIds[0] }));
	ASSERT_TRUE(areAllCellsDeletedExcept(cluster1.getValue(), { cellIds[1], cellIds[2] }));
	ASSERT_TRUE(areAllCellsDeletedExcept(cluster2.getValue(), { cellIds[3], cellIds[4] }));
}

TEST_F(CellConnectorTest, testMoveOneCellSeveralTimesToUniteAndDevideClusters)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 5; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.retainCellClusters({
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setMaxConnections(2),
		}),
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 200, 98 }).setId(cellIds[1]).setConnectingCells({ cellIds[2] }).setMaxConnections(1),
			CellDescription().setPos({ 200, 99 }).setId(cellIds[2]).setConnectingCells({ cellIds[1] }).setMaxConnections(2)
		}),
		CellClusterDescription().setId(_numberGen->getTag()).retainCells(
		{
			CellDescription().setPos({ 200, 101 }).setId(cellIds[3]).setConnectingCells({ cellIds[4] }).setMaxConnections(2),
			CellDescription().setPos({ 200, 102 }).setId(cellIds[4]).setConnectingCells({ cellIds[3] }).setMaxConnections(1)
		})
	});
	_navi.update(_data);
	for (int i = 0; i < 10; ++i) {
		uint64_t clusterIndex = _navi.clusterIndicesByCellIds.at(cellIds[0]);
		uint64_t cellIndex = _navi.cellIndicesByCellIds.at(cellIds[0]);
		_data.clusters[clusterIndex]->cells[cellIndex]->pos.setValue({ 200, 100 });
		_connector->reconnect(_data);
		_navi.update(_data);

		clusterIndex = _navi.clusterIndicesByCellIds.at(cellIds[0]);
		cellIndex = _navi.cellIndicesByCellIds.at(cellIds[0]);
		_data.clusters[clusterIndex]->cells[cellIndex]->pos.setValue({ 100, 100 });
		_connector->reconnect(_data);
		_navi.update(_data);
	}

	auto cluster0 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[1]));
	auto cluster2 = _data.clusters.at(_navi.clusterIndicesByCellIds.at(cellIds[3]));

	ASSERT_EQ(5, _data.clusters.size());
	ASSERT_TRUE(areAllClustersDeletedExcept({ cluster0->id, cluster1->id, cluster2->id }));
	ASSERT_TRUE(areAllCellsDeletedExcept(cluster0.getValue(), { cellIds[0] }));
	ASSERT_TRUE(areAllCellsDeletedExcept(cluster1.getValue(), { cellIds[1], cellIds[2] }));
	ASSERT_TRUE(areAllCellsDeletedExcept(cluster2.getValue(), { cellIds[3], cellIds[4] }));
}
