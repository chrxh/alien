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
	SimulationController* _controller = nullptr;
	SimulationParameters* _parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;
	CellConnector* _connector = nullptr;

	IntVector2D _universeSize{ 600, 300 };
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

TEST_F(CellConnectorTest, testMoveOneCellAway)
{
	DataDescription data;
	auto clusterId = _numberGen->getTag();
	auto cellId1 = _numberGen->getTag();
	auto cellId2 = _numberGen->getTag();
	data.retainCellCluster(CellClusterDescription().setId(clusterId).retainCells(
	{
		CellDescription().setPos({ 100, 100 }).setId(cellId1).setConnectingCells({ cellId2 }).setMaxConnections(1),
		CellDescription().setPos({ 101, 100 }).setId(cellId2).setConnectingCells({ cellId1 }).setMaxConnections(1)
	}));
	data.clusters[0]->cells[1]->pos.setValue({ 103, 100 });

	_connector->reconnect(data);

	ASSERT_EQ(2, data.clusters.size());
	ASSERT_TRUE(data.clusters[0].isModified());
	ASSERT_TRUE(data.clusters[1].isAdded());

	ASSERT_EQ(2, data.clusters[0]->cells.size());
	ASSERT_TRUE(data.clusters[0]->cells[0].isModified());
	ASSERT_TRUE(data.clusters[0]->cells[1].isDeleted());
	ASSERT_EQ(cellId1, data.clusters[0]->cells[0]->id);
	ASSERT_EQ(cellId2, data.clusters[0]->cells[1]->id);

	ASSERT_EQ(1, data.clusters[1]->cells.size());
	ASSERT_TRUE(data.clusters[1]->cells[0].isAdded());
	ASSERT_EQ(cellId2, data.clusters[1]->cells[0]->id);
}

TEST_F(CellConnectorTest, testMoveOneCellToAnOtherCluster)
{
	DataDescription data;
	auto clusterId1 = _numberGen->getTag();
	auto clusterId2 = _numberGen->getTag();
	auto cellId1 = _numberGen->getTag();
	auto cellId2 = _numberGen->getTag();
	auto cellId3 = _numberGen->getTag();
	auto cellId4 = _numberGen->getTag();
	data.retainCellClusters({
		CellClusterDescription().setId(clusterId1).retainCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellId1).setConnectingCells({ cellId2 }).setMaxConnections(1),
			CellDescription().setPos({ 101, 100 }).setId(cellId2).setConnectingCells({ cellId1 }).setMaxConnections(1)
		}), 
		CellClusterDescription().setId(clusterId2).retainCells(
		{
			CellDescription().setPos({ 200, 100 }).setId(cellId3).setConnectingCells({ cellId4 }).setMaxConnections(2),
			CellDescription().setPos({ 201, 100 }).setId(cellId4).setConnectingCells({ cellId3 }).setMaxConnections(1)
		})
	});
	data.clusters[0]->cells[1]->pos.setValue({ 199, 100 });

	_connector->reconnect(data);

	ASSERT_EQ(2, data.clusters.size());
	ASSERT_TRUE(data.clusters[0].isModified());
	ASSERT_TRUE(data.clusters[1].isModified());

	ASSERT_EQ(clusterId1, data.clusters[0]->id);
	ASSERT_EQ(2, data.clusters[0]->cells.size());
	ASSERT_TRUE(data.clusters[0]->cells[0].isModified());
	ASSERT_TRUE(data.clusters[0]->cells[1].isDeleted());
	ASSERT_EQ(cellId1, data.clusters[0]->cells[0]->id);
	ASSERT_EQ(cellId2, data.clusters[0]->cells[1]->id);

	ASSERT_EQ(clusterId2, data.clusters[1]->id);
	ASSERT_EQ(3, data.clusters[1]->cells.size());
	ASSERT_TRUE(data.clusters[1]->cells[0].isModified());
	ASSERT_TRUE(data.clusters[1]->cells[1].isModified());
	ASSERT_TRUE(data.clusters[1]->cells[2].isAdded());

}


