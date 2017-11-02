#include <gtest/gtest.h>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/Settings.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/DescriptionHelper.h"
#include "Model/Local/SimulationContextLocal.h"
#include "Model/Api/SimulationParameters.h"

#include "tests/Predicates.h"

class CellConnectorTest : public ::testing::Test
{
public:
	CellConnectorTest();
	~CellConnectorTest();

protected:
	bool clusterConsistsOfFollowingCells(ClusterDescription const &cluster, set<uint64_t> const &cellIds);

	SimulationController* _controller = nullptr;
	SimulationParameters* _parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;
	DescriptionHelper* _connector = nullptr;

	IntVector2D _universeSize{ 600, 300 };

	DataDescription _data;
	DescriptionNavigator _navi;
};

CellConnectorTest::CellConnectorTest()
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto symbols = facade->buildDefaultSymbolTable();
	_parameters = facade->buildDefaultSimulationParameters();
	_controller = facade->buildSimulationController(1, { 1,1 }, _universeSize, symbols, _parameters);
	auto context = static_cast<SimulationContextLocal*>(_controller->getContext());
	_numberGen = context->getNumberGenerator();
	_connector = facade->buildDescriptionHelper(context);
}

CellConnectorTest::~CellConnectorTest()
{
	delete _controller;
	delete _connector;
}

bool CellConnectorTest::clusterConsistsOfFollowingCells(ClusterDescription const &cluster, set<uint64_t> const & cellIds)
{
	vector<uint64_t> clusterCellIds(cluster.cells->size());
	std::transform(cluster.cells->begin(), cluster.cells->end(), clusterCellIds.begin(), [](auto const &cell) {
		return cell.id;
	});
	set<uint64_t> clusterCellIdSet(clusterCellIds.begin(), clusterCellIds.end());
	return clusterCellIdSet == cellIds;
}


TEST_F(CellConnectorTest, testMoveOneCellAway)
{
	vector<uint64_t> cellIds;
	cellIds.push_back(_numberGen->getTag());
	cellIds.push_back(_numberGen->getTag());
	_data.addCluster(ClusterDescription().setId(_numberGen->getTag()).addCells(
	{
		CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setConnectingCells({ cellIds[1] }).setMaxConnections(1),
		CellDescription().setPos({ 101, 100 }).setId(cellIds[1]).setConnectingCells({ cellIds[0] }).setMaxConnections(1)
	}));
	_data.clusters->at(0).cells->at(1).pos = QVector2D({ 103, 100 });

	_connector->reconnect(_data, { _data.clusters->at(0).cells->at(1).id });

	_navi.update(_data);
	auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[1]));
	ASSERT_EQ(2, _data.clusters->size());
	ASSERT_EQ(1, cluster0.cells->size());
	ASSERT_EQ(1, cluster1.cells->size());
}

TEST_F(CellConnectorTest, testMoveOneCellWithinCluster)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 3; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.addClusters({
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 100 }).setId(cellIds[0]).setConnectingCells({ cellIds[1] }).setMaxConnections(1),
			CellDescription().setPos({ 200, 101 }).setId(cellIds[1]).setConnectingCells({ cellIds[0], cellIds[2] }).setMaxConnections(2),
			CellDescription().setPos({ 200, 102 }).setId(cellIds[2]).setConnectingCells({ cellIds[1] }).setMaxConnections(1),
		})
	});
	_data.clusters->at(0).cells->at(1).pos = QVector2D({ 200, 101.1f });

	_connector->reconnect(_data, { _data.clusters->at(0).cells->at(1).id });

	_navi.update(_data);
	auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[1]));
	ASSERT_EQ(1, _data.clusters->size());
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster0, { cellIds[0], cellIds[1], cellIds[2] }));
}

TEST_F(CellConnectorTest, testMoveOneCellToAnOtherCluster)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 4; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.addClusters({
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setConnectingCells({ cellIds[1] }).setMaxConnections(1),
			CellDescription().setPos({ 101, 100 }).setId(cellIds[1]).setConnectingCells({ cellIds[0] }).setMaxConnections(1)
		}),
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 100 }).setId(cellIds[2]).setConnectingCells({ cellIds[3] }).setMaxConnections(2),
			CellDescription().setPos({ 201, 100 }).setId(cellIds[3]).setConnectingCells({ cellIds[2] }).setMaxConnections(1)
		})
	});
	_data.clusters->at(0).cells->at(1).pos = QVector2D({ 199, 100 });

	_connector->reconnect(_data, { _data.clusters->at(0).cells->at(1).id });

	_navi.update(_data);
	auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[2]));
	ASSERT_EQ(2, _data.clusters->size());
	ASSERT_EQ(1, cluster0.cells->size());
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster1, { cellIds[2], cellIds[3], cellIds[1] }));
}

TEST_F(CellConnectorTest, testMoveOneCellToUniteClusters)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 5; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.addClusters({
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setMaxConnections(2),
		}),
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 98 }).setId(cellIds[1]).setConnectingCells({ cellIds[2] }).setMaxConnections(1),
			CellDescription().setPos({ 200, 99 }).setId(cellIds[2]).setConnectingCells({ cellIds[1] }).setMaxConnections(2)
		}),
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 101 }).setId(cellIds[3]).setConnectingCells({ cellIds[4] }).setMaxConnections(2),
			CellDescription().setPos({ 200, 102 }).setId(cellIds[4]).setConnectingCells({ cellIds[3] }).setMaxConnections(1)
		})
	});
	_data.clusters->at(0).cells->at(0).pos = QVector2D({ 200, 100 });

	_connector->reconnect(_data, { _data.clusters->at(0).cells->at(0).id });
	_navi.update(_data);
	auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	ASSERT_EQ(1, _data.clusters->size());
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster0, { cellIds[0], cellIds[1], cellIds[2], cellIds[3], cellIds[4] }));
}

TEST_F(CellConnectorTest, testMoveOneCellToUniteAndDevideClusters)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 5; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.addClusters({
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setMaxConnections(2),
		}),
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 98 }).setId(cellIds[1]).setConnectingCells({ cellIds[2] }).setMaxConnections(1),
			CellDescription().setPos({ 200, 99 }).setId(cellIds[2]).setConnectingCells({ cellIds[1] }).setMaxConnections(2)
		}),
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 101 }).setId(cellIds[3]).setConnectingCells({ cellIds[4] }).setMaxConnections(2),
			CellDescription().setPos({ 200, 102 }).setId(cellIds[4]).setConnectingCells({ cellIds[3] }).setMaxConnections(1)
		})
	});
	_data.clusters->at(0).cells->at(0).pos = QVector2D({ 200, 100 });

	_connector->reconnect(_data, { _data.clusters->at(0).cells->at(0).id });
	_navi.update(_data);
	uint64_t clusterIndex = _navi.clusterIndicesByCellIds.at(cellIds[0]);
	uint64_t cellIndex = _navi.cellIndicesByCellIds.at(cellIds[0]);
	_data.clusters->at(clusterIndex).cells->at(cellIndex).pos = QVector2D({ 100, 100 });
	_connector->reconnect(_data, { _data.clusters->at(clusterIndex).cells->at(cellIndex).id });

	_navi.update(_data);
	auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[1]));
	auto cluster2 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[3]));
	ASSERT_EQ(3, _data.clusters->size());
	ASSERT_EQ(1, cluster0.cells->size());
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster1, { cellIds[1], cellIds[2] }));
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster2, { cellIds[3], cellIds[4] }));
}

TEST_F(CellConnectorTest, testMoveOneCellSeveralTimesToUniteAndDevideClusters)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 5; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.addClusters({
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setMaxConnections(2),
		}),
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 98 }).setId(cellIds[1]).setConnectingCells({ cellIds[2] }).setMaxConnections(1),
			CellDescription().setPos({ 200, 99 }).setId(cellIds[2]).setConnectingCells({ cellIds[1] }).setMaxConnections(2)
		}),
		ClusterDescription().setId(_numberGen->getTag()).addCells(
		{
			CellDescription().setPos({ 200, 101 }).setId(cellIds[3]).setConnectingCells({ cellIds[4] }).setMaxConnections(2),
			CellDescription().setPos({ 200, 102 }).setId(cellIds[4]).setConnectingCells({ cellIds[3] }).setMaxConnections(1)
		})
	});

	_navi.update(_data);
	for (int i = 0; i < 10; ++i) {
		uint64_t clusterIndex = _navi.clusterIndicesByCellIds.at(cellIds[0]);
		uint64_t cellIndex = _navi.cellIndicesByCellIds.at(cellIds[0]);
		_data.clusters->at(clusterIndex).cells->at(cellIndex).pos = QVector2D({ 200, 100 });
		_connector->reconnect(_data, { _data.clusters->at(clusterIndex).cells->at(cellIndex).id });
		_navi.update(_data);

		auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
		ASSERT_EQ(1, _data.clusters->size());
		ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster0, { cellIds[0], cellIds[1], cellIds[2], cellIds[3], cellIds[4] }));

		clusterIndex = _navi.clusterIndicesByCellIds.at(cellIds[0]);
		cellIndex = _navi.cellIndicesByCellIds.at(cellIds[0]);
		_data.clusters->at(clusterIndex).cells->at(cellIndex).pos = QVector2D({ 100, 100 });
		_connector->reconnect(_data, { _data.clusters->at(clusterIndex).cells->at(cellIndex).id });
		_navi.update(_data);

		cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
		auto cluster1 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[1]));
		auto cluster2 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[3]));
		ASSERT_EQ(3, _data.clusters->size());
		ASSERT_EQ(1, cluster0.cells->size());
		ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster1, { cellIds[1], cellIds[2] }));
		ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster2, { cellIds[3], cellIds[4] }));
	}
}

TEST_F(CellConnectorTest, testMoveSeveralCells)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 5; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}
	_data.addCluster(ClusterDescription().setId(_numberGen->getTag()).addCells(
	{
		CellDescription().setPos({ 100, 100 }).setId(cellIds[0]).setConnectingCells({ cellIds[1] }).setMaxConnections(1),
		CellDescription().setPos({ 101, 100 }).setId(cellIds[1]).setConnectingCells({ cellIds[0], cellIds[2] }).setMaxConnections(3),
		CellDescription().setPos({ 102, 100 }).setId(cellIds[2]).setConnectingCells({ cellIds[1], cellIds[3] }).setMaxConnections(5),
		CellDescription().setPos({ 103, 100 }).setId(cellIds[3]).setConnectingCells({ cellIds[2], cellIds[4] }).setMaxConnections(3),
		CellDescription().setPos({ 104, 100 }).setId(cellIds[4]).setConnectingCells({ cellIds[3] }).setMaxConnections(4)
	}));
	for (int i = 0; i < 5; ++i) {
		_data.clusters->at(0).cells->at(i).pos = QVector2D({ 200 + static_cast<float>(i), 100 });
	}

	_connector->reconnect(_data,
	{
		_data.clusters->at(0).cells->at(0).id,
		_data.clusters->at(0).cells->at(1).id,
		_data.clusters->at(0).cells->at(2).id,
		_data.clusters->at(0).cells->at(3).id,
		_data.clusters->at(0).cells->at(4).id
	});

	_navi.update(_data);
	auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	ASSERT_EQ(1, _data.clusters->size());
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster0, { cellIds[0], cellIds[1], cellIds[2], cellIds[3], cellIds[4] }));
	auto cell0 = cluster0.cells->at(_navi.cellIndicesByCellIds.at(cellIds[0]));
	auto cell1 = cluster0.cells->at(_navi.cellIndicesByCellIds.at(cellIds[1]));
	auto cell2 = cluster0.cells->at(_navi.cellIndicesByCellIds.at(cellIds[2]));
	auto cell3 = cluster0.cells->at(_navi.cellIndicesByCellIds.at(cellIds[3]));
	auto cell4 = cluster0.cells->at(_navi.cellIndicesByCellIds.at(cellIds[4]));
	ASSERT_EQ(1, cell0.connectingCells.get().size());
	ASSERT_EQ(2, cell1.connectingCells.get().size());
	ASSERT_EQ(2, cell2.connectingCells.get().size());
	ASSERT_EQ(2, cell3.connectingCells.get().size());
	ASSERT_EQ(1, cell4.connectingCells.get().size());
}

TEST_F(CellConnectorTest, testMoveSeveralCellsOverOtherCells)
{
	vector<uint64_t> cellIds;
	for (int i = 0; i < 20; ++i) {
		cellIds.push_back(_numberGen->getTag());
	}

	for (int j = 0; j < 4; ++j) {
		_data.addCluster(ClusterDescription().setId(_numberGen->getTag()).addCells({
			CellDescription().setPos({ 100 + static_cast<float>(j) * 25, 100 }).setId(cellIds[0 + j * 5]).setMaxConnections(2).setConnectingCells({ cellIds[1 + j * 5] }),
			CellDescription().setPos({ 100 + static_cast<float>(j) * 25, 100 }).setId(cellIds[1 + j * 5]).setMaxConnections(3).setConnectingCells({ cellIds[0 + j * 5], cellIds[2 + j * 5] }),
			CellDescription().setPos({ 100 + static_cast<float>(j) * 25, 100 }).setId(cellIds[2 + j * 5]).setMaxConnections(3).setConnectingCells({ cellIds[1 + j * 5], cellIds[3 + j * 5] }),
			CellDescription().setPos({ 100 + static_cast<float>(j) * 25, 100 }).setId(cellIds[3 + j * 5]).setMaxConnections(3).setConnectingCells({ cellIds[2 + j * 5], cellIds[4 + j * 5] }),
			CellDescription().setPos({ 100 + static_cast<float>(j) * 25, 100 }).setId(cellIds[4 + j * 5]).setMaxConnections(2).setConnectingCells({ cellIds[3 + j * 5] })
		}));
	}

	for (int movement = 0; movement < 100; ++movement) {
		_navi.update(_data);

		list<uint64_t> ids;
		for (int i = 0; i < 10; ++i) {
			auto &cluster = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[i]));
			auto &cell = cluster.cells->at(_navi.cellIndicesByCellIds.at(cellIds[i]));
			auto pos = *cell.pos;
			pos.setX(pos.x() + 1);
			cell.pos = pos;
			ids.push_back(cell.id);
		}
		_connector->reconnect(_data, ids);
	}
	_navi.update(_data);

	ASSERT_EQ(4, _data.clusters->size());
	auto cluster0 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[0]));
	auto cluster1 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[5]));
	auto cluster2 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[10]));
	auto cluster3 = _data.clusters->at(_navi.clusterIndicesByCellIds.at(cellIds[15]));
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster0, { cellIds[0], cellIds[1], cellIds[2], cellIds[3], cellIds[4] }));
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster1, { cellIds[5], cellIds[6], cellIds[7], cellIds[8], cellIds[9] }));
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster2, { cellIds[10], cellIds[11], cellIds[12], cellIds[13], cellIds[14] }));
	ASSERT_TRUE(clusterConsistsOfFollowingCells(cluster3, { cellIds[15], cellIds[16], cellIds[17], cellIds[18], cellIds[19] }));

}
