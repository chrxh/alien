#include <boost/range/adaptors.hpp>
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

class DataDescriptionTransferGpuTest
	: public IntegrationTestFramework
{
public:
	DataDescriptionTransferGpuTest();
	~DataDescriptionTransferGpuTest();

protected:
	SimulationControllerGpu* _controller = nullptr;
	SimulationContext* _context = nullptr;
	SpaceProperties* _spaceProp = nullptr;
	SimulationAccessGpu* _access = nullptr;
	IntVector2D _gridSize{ 6, 6 };
};

DataDescriptionTransferGpuTest::DataDescriptionTransferGpuTest()
	: IntegrationTestFramework({ 600, 300 })
{
	_controller = _gpuFacade->buildSimulationController({ _universeSize, _symbols, _parameters }, ModelGpuData(), 0);
	_context = _controller->getContext();
	_spaceProp = _context->getSpaceProperties();
	_access = _gpuFacade->buildSimulationAccess();
	_access->init(_controller);
	_numberGen = _context->getNumberGenerator();
}

DataDescriptionTransferGpuTest::~DataDescriptionTransferGpuTest()
{
	delete _access;
	delete _controller;
}

TEST_F(DataDescriptionTransferGpuTest, DISABLED_testCreateClusterWithCompleteCell)
{
	DataDescription dataBefore;
	dataBefore.addCluster(createSingleCellClusterWithCompleteData());
	IntegrationTestHelper::updateData(_access, dataBefore);

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataBefore, dataAfter));
}

/**
* Situation: add token to cell
* Expected result: token in simulation added
*/
TEST_F(DataDescriptionTransferGpuTest, testAddToken)
{
	DataDescription dataBefore;
	auto cellId = _numberGen->getId();
	auto clusterId = _numberGen->getId();
	auto cluster = createSingleCellCluster(clusterId, cellId);
	dataBefore.addCluster(cluster);

	DataDescription dataChanged;
	cluster.cells->at(0).addToken(TokenDescription().setEnergy(30).setData(QByteArray(_parameters.tokenMemorySize, 0)));
	dataChanged.addCluster(cluster);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: change cell with token
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTest, testChangeCellWithToken_changeClusterId)
{
	auto cluster = createSingleCellCluster(_numberGen->getId(), _numberGen->getId());
	cluster.cells->at(0).addToken(TokenDescription().setEnergy(30).setData(QByteArray(_parameters.tokenMemorySize, 0)));

	DataDescription dataBefore;
	dataBefore.addCluster(cluster);

	DataDescription dataChanged;
	auto otherCluster = cluster;
	otherCluster.id = _numberGen->getId();
	dataChanged.addCluster(otherCluster);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: - one cell has a token
*			 - add further token
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTest, testChangeCellWithToken_addSecondToken)
{
	auto token = TokenDescription().setEnergy(30).setData(QByteArray(_parameters.tokenMemorySize, 0));

	auto cluster1 = createSingleCellCluster(_numberGen->getId(), _numberGen->getId());
	auto& cell1 = cluster1.cells->at(0);
	cell1.addToken(token);

	auto cluster2 = cluster1;
	cluster2.id = _numberGen->getId();
	auto& cell2 = cluster2.cells->at(0);
	cell2.addToken(token);

	DataDescription dataBefore;
	dataBefore.addCluster(cluster1);

	DataDescription dataChanged;
	dataChanged.addCluster(cluster2);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: - one cluster has two cells
*			 - one of its cells has a token, the other not
*			 - add further token to other cell
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTest, testChangeClusterWithToken_addSecondToken)
{
	auto token = TokenDescription().setEnergy(30).setData(QByteArray(_parameters.tokenMemorySize, 0));

	auto cluster1 = createHorizontalCluster(2);
	auto& cell1 = cluster1.cells->at(0);
	cell1.addToken(token);

	auto cluster2 = cluster1;
	auto& cell2 = cluster2.cells->at(1);
	cell2.addToken(token);

	DataDescription dataBefore;
	dataBefore.addCluster(cluster1);

	DataDescription dataChanged;
	dataChanged.addCluster(cluster2);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: - one cluster with one cell and one token
*			 - an other cluster with one cell and two tokens
*			 - position of first cluster is changed
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTest, testChangeCellWithSeveralTokens)
{
	auto token = TokenDescription().setEnergy(30).setData(QByteArray(_parameters.tokenMemorySize, 0));

	auto cluster1 = createSingleCellCluster(_numberGen->getId(), _numberGen->getId());
	auto& cell1 = cluster1.cells->at(0);
	cell1.addToken(token);

	auto cluster2 = createSingleCellCluster(_numberGen->getId(), _numberGen->getId());
	auto& cell2 = cluster2.cells->at(0);
	cell2.addToken(token);
	cell2.addToken(token);

	auto cluster3 = cluster1;
	cluster3.id = _numberGen->getId();
	*cluster3.pos = *cluster3.pos + QVector2D{ 1.0, 0.0 };
	*cluster3.cells->at(0).pos = *cluster3.cells->at(0).pos + QVector2D{ 1.0, 0.0 };

	DataDescription dataBefore;
	dataBefore.addCluster(cluster1);
	dataBefore.addCluster(cluster2);

	DataDescription dataChanged;
	dataChanged.addCluster(cluster3);
	dataChanged.addCluster(cluster2);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: - cluster with one cell and one token
*			 - an other cluster with one cell and two tokens
*			 - first cluster is removed
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTest, testRemoveCellWithToken)
{
	auto token = TokenDescription().setEnergy(30).setData(QByteArray(_parameters.tokenMemorySize, 0));

	auto cluster1 = createSingleCellCluster(_numberGen->getId(), _numberGen->getId());
	auto& cell1 = cluster1.cells->at(0);
	cell1.addToken(token);

	auto cluster2 = createSingleCellCluster(_numberGen->getId(), _numberGen->getId());
	auto& cell2 = cluster2.cells->at(0);
	cell2.addToken(token);
	cell2.addToken(token);

	DataDescription dataBefore;
	dataBefore.addCluster(cluster1);
	dataBefore.addCluster(cluster2);

	DataDescription dataChanged;
	dataChanged.addCluster(cluster2);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: change particle properties
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTest, testChangeParticle)
{
	DataDescription dataBefore;
	auto particleEnergy1 = _parameters.cellMinEnergy / 2.0;
	auto particleId = _numberGen->getId();
	auto particleBefore = ParticleDescription().setId(particleId).setEnergy(particleEnergy1).setPos({ 100, 100 }).setVel({ 0.5f, 0.0f });
	dataBefore.addParticle(particleBefore);
	
	DataDescription dataChanged;
	auto particleEnergy2 = _parameters.cellMinEnergy / 3.0;
	auto particleChange = ParticleDescription().setId(particleId).setEnergy(particleEnergy2).setPos({ 150, 150 }).setVel({ 0.0f, -0.3f });
	dataChanged.addParticle(particleChange);

	IntegrationTestHelper::updateData(_access, dataBefore);
	IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 }, { _universeSize.x, _universeSize.y } });

	ASSERT_TRUE(isCompatible(dataChanged, dataAfter));
}

/**
* Situation: create cluster and particle at a position outside universe
* Expected result: cluster and particle should be positioned inside universe due to torus topology
*/
TEST_F(DataDescriptionTransferGpuTest, testCreateDataOutsideBoundaries)
{
	auto universeSize = _spaceProp->getSize();
	DataDescription dataBefore;
	dataBefore.addCluster(createHorizontalCluster(3, QVector2D{ 2.5f * universeSize.x, 2.5f * universeSize.y}));
	dataBefore.addParticle(createParticle(QVector2D{ 2.5f * universeSize.x + 2.0f, 2.5f * universeSize.y + 2.0f }));

	IntegrationTestHelper::updateData(_access, dataBefore);
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 }, { _universeSize.x, _universeSize.y } });

	EXPECT_EQ(1, dataAfter.clusters->size());
	auto origCluster = dataBefore.clusters->at(0);
	auto newCluster = dataAfter.clusters->at(0);
	EXPECT_TRUE(isCompatible(*origCluster.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newCluster.pos));

	unordered_map<uint64_t, CellDescription> origCellById = IntegrationTestHelper::getCellByCellId(dataBefore);
	unordered_map<uint64_t, CellDescription> newCellById = IntegrationTestHelper::getCellByCellId(dataAfter);
	for (CellDescription const& origCell : origCellById | boost::adaptors::map_values) {
		CellDescription newCell = newCellById.at(origCell.id);
		EXPECT_TRUE(isCompatible(*origCell.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newCell.pos));
	}

	EXPECT_EQ(1, dataAfter.particles->size());
	auto origParticle = dataBefore.particles->at(0);
	auto newParticle = dataAfter.particles->at(0);
	EXPECT_TRUE(isCompatible(*origParticle.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newParticle.pos));
}

/**
* Situation:
* 	- one cluster with 10x10 cell and one particle
*	- particle and some cells are moved via update
* Fixed error: crash after moving cells in a cluster in item view
* Expected result: no crash
*/
TEST_F(DataDescriptionTransferGpuTest, regressionTest_changeData)
{
	auto descHelper = _basicFacade->buildDescriptionHelper();
	descHelper->init(_context);

	auto size = _spaceProp->getSize();
	DataDescription dataBefore;
	dataBefore.addCluster(createRectangularCluster({ 10, 10 }, QVector2D{ size.x / 2.0f, size.y / 2.0f }, QVector2D{}));
	dataBefore.addParticle(createParticle(QVector2D{ 0, 0 }));

	IntegrationTestHelper::updateData(_access, dataBefore);

	auto dataModified = dataBefore;
	unordered_set<uint64_t> idsOfChangedCells;
	for(int i = 0; i < 10; ++i) {
		auto& cluster = dataModified.clusters->at(0);
		auto& cell = cluster.cells->at(i);
		cell.pos->setX(cell.pos->x() + 50.0f);
		idsOfChangedCells.insert(cell.id);
	}
	auto& particle = dataModified.particles->at(0);
	particle.pos->setX(particle.pos->x() + 50.0f);

	descHelper->reconnect(dataModified, dataBefore, idsOfChangedCells);
	EXPECT_NO_THROW(IntegrationTestHelper::updateData(_access, DataChangeDescription(dataBefore, dataModified)));
}