#include <gtest/gtest.h>

#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/DescriptionHelper.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationContext.h"

#include "ModelGpu/SimulationControllerGpu.h"
#include "ModelGpu/SimulationAccessGpu.h"
#include "ModelGpu/ModelGpuData.h"
#include "ModelGpu/ModelGpuBuilderFacade.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"

class SimulationGpuTest
	: public IntegrationTestFramework
{
public:
	SimulationGpuTest();
	~SimulationGpuTest();

protected:
	SimulationControllerGpu* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SpaceProperties* _spaceProp = nullptr;
	SimulationAccessGpu* _access = nullptr;
	IntVector2D _gridSize{ 6, 6 };
};

SimulationGpuTest::SimulationGpuTest()
	: IntegrationTestFramework({ 600, 300 })
{
	_controller = _gpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelGpuData(), 0);
	_context = _controller->getContext();
	_spaceProp = _context->getSpaceProperties();
	_access = _gpuFacade->buildSimulationAccess();
	_parameters = _context->getSimulationParameters();
	_numberGen = _context->getNumberGenerator();
	_access->init(_controller);
}

SimulationGpuTest::~SimulationGpuTest()
{
	delete _access;
	delete _controller;
}

/**
* Situation: horizontal collision of two cells
* Expected result: direction of movement of both cells changed
*/
TEST_F(SimulationGpuTest, testCollisionOfSingleCells_horizontal)
{
	DataDescription origData;
	auto cellEnergy = _parameters->cellFunctionConstructorOffspringCellEnergy;

	auto cellId1 = _numberGen->getId();
	auto cell1 = CellDescription().setId(cellId1).setPos({ 100, 100 }).setMaxConnections(0).setEnergy(cellEnergy);
	auto cluster1 = ClusterDescription().setId(_numberGen->getId()).setVel({ 0.1f, 0 }).setAngle(0).setAngularVel(0)
		.addCell(cell1);
	cluster1.setPos(cluster1.getClusterPosFromCells());
	origData.addCluster(cluster1);

	auto cellId2 = _numberGen->getId();
	auto cell2 = CellDescription().setId(cellId2).setPos({ 110, 100 }).setMaxConnections(0).setEnergy(cellEnergy);
	auto cluster2 = ClusterDescription().setId(_numberGen->getId()).setVel({ -0.1f, 0 }).setAngle(0).setAngularVel(0)
		.addCell(cell2);
	cluster2.setPos(cluster2.getClusterPosFromCells());
	origData.addCluster(cluster2);

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(150, _controller);

	IntRect rect = { { 0, 0 },{ _universeSize.x, _universeSize.y } };
	DataDescription newData = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_EQ(2, newData.clusters->size());
	auto cellById = IntegrationTestHelper::getCellByCellId(newData);
	auto newCell1 = cellById.at(cellId1);
	auto newCell2 = cellById.at(cellId2);
	auto clusterById = IntegrationTestHelper::getClusterByCellId(newData);
	auto newCluster1 = clusterById.at(cellId1);
	auto newCluster2 = clusterById.at(cellId2);

	EXPECT_GE(99, newCell1.pos->x());
	EXPECT_TRUE(isCompatible(100.0f, newCell1.pos->y()));
	EXPECT_TRUE(isCompatible(QVector2D(-0.1f, 0), *newCluster1.vel));

	EXPECT_LE(111, newCell2.pos->x());
	EXPECT_TRUE(isCompatible(100.0f, newCell2.pos->y()));
	EXPECT_TRUE(isCompatible(QVector2D(0.1f, 0), *newCluster2.vel));
}

/**
* Situation: vertical collision of two cells
* Expected result: direction of movement of both cells changed
*/
TEST_F(SimulationGpuTest, testCollisionOfSingleCells_vertical)
{
	DataDescription origData;
	auto cellEnergy = _parameters->cellFunctionConstructorOffspringCellEnergy;

	auto cellId1 = _numberGen->getId();
	auto cell1 = CellDescription().setId(cellId1).setPos({ 100, 100 }).setMaxConnections(0).setEnergy(cellEnergy);
	auto cluster1 = ClusterDescription().setId(_numberGen->getId()).setVel({ 0, 0.1f }).setAngle(0).setAngularVel(0)
		.addCell(cell1);
	cluster1.setPos(cluster1.getClusterPosFromCells());
	origData.addCluster(cluster1);

	auto cellId2 = _numberGen->getId();
	auto cell2 = CellDescription().setId(cellId2).setPos({ 100, 110 }).setMaxConnections(0).setEnergy(cellEnergy);
	auto cluster2 = ClusterDescription().setId(_numberGen->getId()).setVel({ 0, -0.1f }).setAngle(0).setAngularVel(0)
		.addCell(cell2);
	cluster2.setPos(cluster2.getClusterPosFromCells());
	origData.addCluster(cluster2);

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(150, _controller);

	IntRect rect = { { 0, 0 },{ _universeSize.x, _universeSize.y } };
	DataDescription newData = IntegrationTestHelper::getContent(_access, rect);

	ASSERT_EQ(2, newData.clusters->size());
	auto cellById = IntegrationTestHelper::getCellByCellId(newData);
	auto newCell1 = cellById.at(cellId1);
	auto newCell2 = cellById.at(cellId2);
	auto clusterById = IntegrationTestHelper::getClusterByCellId(newData);
	auto newCluster1 = clusterById.at(cellId1);
	auto newCluster2 = clusterById.at(cellId2);

	EXPECT_GE(99, newCell1.pos->y());
	EXPECT_TRUE(isCompatible(100.0f, newCell1.pos->x()));
	EXPECT_TRUE(isCompatible(QVector2D(0, -0.1f), *newCluster1.vel));

	EXPECT_LE(111, newCell2.pos->y());
	EXPECT_TRUE(isCompatible(100.0f, newCell2.pos->x()));
	EXPECT_TRUE(isCompatible(QVector2D(0, 0.1f), *newCluster2.vel));
}

/**
* Situation:
*	- vertical center collision of two horizontal cell clusters
*	- first cluster has no velocity while second cluster moves upward
* Expected result: first cluster moves upward while second cluster stand stills
*/
TEST_F(SimulationGpuTest, testCenterCollisionOfTwoLineStructures)
{
	DataDescription origData;
	origData.addCluster(createHorizontalCluster(100, QVector2D{ 100, 100 }, QVector2D{ 0, 0 }));
	origData.addCluster(createHorizontalCluster(100, QVector2D{ 100, 110 }, QVector2D{ 0, -0.1f }));
	uint64_t clusterId1 = origData.clusters->at(0).id;
	uint64_t clusterId2 = origData.clusters->at(1).id;

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(150, _controller);

	IntRect rect = { { 0, 0 },{ _universeSize.x, _universeSize.y } };
	DataDescription newData = IntegrationTestHelper::getContent(_access, rect);
	ASSERT_EQ(2, newData.clusters->size());

	auto clusterById = IntegrationTestHelper::getClusterByClusterId(newData);
	{
		auto cluster = clusterById.at(clusterId1);
		EXPECT_EQ(100, cluster.pos->x());
		EXPECT_GE(99, cluster.pos->y());
		EXPECT_TRUE(isCompatible(0.0f, cluster.vel->x()));
		EXPECT_TRUE(isCompatible(-0.1f, cluster.vel->y()));
	}

	{
		auto cluster = clusterById.at(clusterId2);
		EXPECT_EQ(100, cluster.pos->x());
		EXPECT_LE(101, cluster.pos->y());
		EXPECT_TRUE(isCompatible(QVector2D(0, 0), *cluster.vel));
	}
}

/**
 * Situation: cluster with cross structure where middle cell connecting 4 parts has low energy            
 * Expected result: cluster decomposes into at least 4 parts
 */
TEST_F(SimulationGpuTest, testDecomposeClusterAfterLowEnergy)
{
	DataDescription origData;
	{
		auto cluster = ClusterDescription().setId(_numberGen->getId()).setVel({ 0, 0 }).setAngle(0).setAngularVel(0);
		for (int i = 0; i < 30; ++i) {
			auto cell = CellDescription().setId(_numberGen->getId()).setPos({ 100, 100 + float(i) }).setMaxConnections(4);
			if (15 == i) {
				cell.setEnergy(_parameters->cellMinEnergy / 2);
			}
			else {
				cell.setEnergy(_parameters->cellMinEnergy * 2);
			}
			cluster.addCell(cell);
		}
		auto leftCell = CellDescription().setId(_numberGen->getId()).setPos({ 99, 115 }).setMaxConnections(4).setEnergy(_parameters->cellMinEnergy * 2);
		cluster.addCell(leftCell);
		auto rightCell = CellDescription().setId(_numberGen->getId()).setPos({ 101, 115 }).setMaxConnections(4).setEnergy(_parameters->cellMinEnergy * 2);
		cluster.addCell(rightCell);

		for (int i = 0; i < 30; ++i) {
			list<uint64_t> connectingCells;
			if (i > 0) {
				connectingCells.emplace_back(cluster.cells->at(i - 1).id);
			}
			if (i < 30 - 1) {
				connectingCells.emplace_back(cluster.cells->at(i + 1).id);
			}
			cluster.cells->at(i).setConnectingCells(connectingCells);
		}
		cluster.cells->at(30).addConnection(cluster.cells->at(15).id);
		cluster.cells->at(15).addConnection(cluster.cells->at(30).id);
		cluster.cells->at(31).addConnection(cluster.cells->at(15).id);
		cluster.cells->at(15).addConnection(cluster.cells->at(31).id);

		cluster.setPos(cluster.getClusterPosFromCells());
		origData.addCluster(cluster);
	}

	IntegrationTestHelper::updateData(_access, origData);
	IntegrationTestHelper::runSimulation(2, _controller);

	IntRect rect = { { 0, 0 }, { _universeSize.x, _universeSize.y } };
	DataDescription newData = IntegrationTestHelper::getContent(_access, rect);

	auto numClusters = newData.clusters ? newData.clusters->size() : 0;
	ASSERT_LE(4, numClusters);

	unordered_map<int, vector<ClusterDescription>> clustersBySize;
	for (ClusterDescription const& cluster : *newData.clusters) {
		int numCells = cluster.cells ? cluster.cells->size() : 0;
		clustersBySize[numCells].emplace_back(cluster);
	};
	ASSERT_LE(2, clustersBySize.at(1).size());
	ASSERT_EQ(1, clustersBySize.at(14).size());
	ASSERT_EQ(1, clustersBySize.at(15).size());

	unordered_map<uint64_t, CellDescription> origCellById = IntegrationTestHelper::getCellByCellId(origData);
	for (ClusterDescription const& cluster : *newData.clusters) {
		EXPECT_EQ(cluster.getClusterPosFromCells(), *cluster.pos);
		for (CellDescription const& cell : *cluster.cells) {
			CellDescription const& origCell = origCellById.at(cell.id);
			EXPECT_TRUE(isCompatible(cell.pos, origCell.pos));
		}
	}
}
