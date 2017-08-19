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
	cellIds.push_back(_numberGen->getTag());
	cellIds.push_back(_numberGen->getTag());
	cellIds.push_back(_numberGen->getTag());
	cellIds.push_back(_numberGen->getTag());
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
	auto cell0 = cluster0->cells.at(_navi.cellIndicesByCellIds.at(cellIds[0]));

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


