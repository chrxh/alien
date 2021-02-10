#include <boost/range/adaptors.hpp>
#include <gtest/gtest.h>
#include <QEventLoop>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpaceProperties.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"

#include "EngineGpu/SimulationControllerGpu.h"
#include "EngineGpu/SimulationAccessGpu.h"
#include "EngineGpu/EngineGpuData.h"
#include "EngineGpu/EngineGpuBuilderFacade.h"

#include "Tests/Predicates.h"

#include "IntegrationTestHelper.h"
#include "IntegrationTestFramework.h"
#include "IntegrationGpuTestFramework.h"

class DataDescriptionTransferGpuTests
	: public IntegrationGpuTestFramework
{
public:
	DataDescriptionTransferGpuTests();
};

DataDescriptionTransferGpuTests::DataDescriptionTransferGpuTests()
	: IntegrationGpuTestFramework({ 600, 300 })
{
}

TEST_F(DataDescriptionTransferGpuTests, testCreateClusterWithCompleteCell)
{
	DataDescription dataBefore;
	dataBefore.addCluster(createSingleCellClusterWithCompleteData());
	IntegrationTestHelper::updateData(_access, _context, dataBefore);

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkCompatibility(dataBefore, dataAfter);
}

/**
* Situation: add token to cell
* Expected result: token in simulation added
*/
TEST_F(DataDescriptionTransferGpuTests, testAddToken)
{
	DataDescription dataBefore;
	auto cellId = _numberGen->getId();
	auto clusterId = _numberGen->getId();
	auto cluster = createSingleCellCluster(clusterId, cellId);
	dataBefore.addCluster(cluster);

	DataDescription dataChanged;
	auto token = createSimpleToken();
	cluster.cells->at(0).addToken(token);
	dataChanged.addCluster(cluster);

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkCompatibility(dataChanged, dataAfter);
}

/**
* Situation: change cell with token
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTests, testChangeCellWithToken_changeClusterId)
{
	auto cluster = createSingleCellCluster(_numberGen->getId(), _numberGen->getId());
	auto token = createSimpleToken();
	cluster.cells->at(0).addToken(token);

	DataDescription dataBefore;
	dataBefore.addCluster(cluster);

	DataDescription dataChanged;
	auto otherCluster = cluster;
	otherCluster.id = _numberGen->getId();
	dataChanged.addCluster(otherCluster);

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkCompatibility(dataChanged, dataAfter);
}

/**
* Situation: - one cell has a token
*			 - add further token
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTests, testChangeCellWithToken_addSecondToken)
{
	auto token = createSimpleToken();

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

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkCompatibility(dataChanged, dataAfter);
}

/**
* Situation: - one cluster has two cells
*			 - one of its cells has a token, the other not
*			 - add further token to other cell
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTests, testChangeClusterWithToken_addSecondToken)
{
	auto token = createSimpleToken();

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

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkCompatibility(dataChanged, dataAfter);
}

/**
* Situation: - one cluster with one cell and one token
*			 - an other cluster with one cell and two tokens
*			 - position of first cluster is changed
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTests, testChangeCellWithSeveralTokens)
{
	auto token = createSimpleToken();

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

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkCompatibility(dataChanged, dataAfter);
}

/**
* Situation: - cluster with one cell and one token
*			 - an other cluster with one cell and two tokens
*			 - first cluster is removed
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTests, testRemoveCellWithToken)
{
	auto token = createSimpleToken();

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

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkCompatibility(dataChanged, dataAfter);
}

/**
* Situation: change particle properties
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTests, testChangeParticle)
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

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataChanged));

	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 }, { _universeSize.x, _universeSize.y } });

    checkCompatibility(dataChanged, dataAfter);
}

/**
* Situation: create cluster and particle at a position outside universe
* Expected result: cluster and particle should be positioned inside universe due to torus topology
*/
TEST_F(DataDescriptionTransferGpuTests, testCreateDataOutsideBoundaries)
{
	auto universeSize = _spaceProp->getSize();
	DataDescription dataBefore;
	dataBefore.addCluster(createHorizontalCluster(3, QVector2D{ 2.5f * universeSize.x, 2.5f * universeSize.y}));
	dataBefore.addParticle(createParticle(QVector2D{ 2.5f * universeSize.x + 2.0f, 2.5f * universeSize.y + 2.0f }));

	IntegrationTestHelper::updateData(_access, _context, dataBefore);
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, { { 0, 0 }, { _universeSize.x, _universeSize.y } });

	EXPECT_EQ(1, dataAfter.clusters->size());
	auto origCluster = dataBefore.clusters->at(0);
	auto newCluster = dataAfter.clusters->at(0);
	checkCompatibility(*origCluster.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newCluster.pos);

	unordered_map<uint64_t, CellDescription> origCellById = IntegrationTestHelper::getCellByCellId(dataBefore);
	unordered_map<uint64_t, CellDescription> newCellById = IntegrationTestHelper::getCellByCellId(dataAfter);
	for (CellDescription const& origCell : origCellById | boost::adaptors::map_values) {
		CellDescription newCell = newCellById.at(origCell.id);
		checkCompatibility(*origCell.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newCell.pos);
	}

	EXPECT_EQ(1, dataAfter.particles->size());
	auto origParticle = dataBefore.particles->at(0);
	auto newParticle = dataAfter.particles->at(0);
	checkCompatibility(*origParticle.pos - QVector2D{ 2.0f * universeSize.x, 2.0f * universeSize.y }, *newParticle.pos);
}

/**
* Situation:
* 	- one cluster with 10x10 cell and one particle
*	- particle and some cells are moved via update
* Fixed error: crash after moving cells in a cluster in item view
* Expected result: no crash
*/
TEST_F(DataDescriptionTransferGpuTests, regressionTestChangeData)
{
	auto descHelper = _basicFacade->buildDescriptionHelper();
	descHelper->init(_context);

	auto size = _spaceProp->getSize();
	DataDescription dataBefore;
	dataBefore.addCluster(createRectangularCluster({ 10, 10 }, QVector2D{ size.x / 2.0f, size.y / 2.0f }, QVector2D{}));
	dataBefore.addParticle(createParticle(QVector2D{ 0, 0 }));

	IntegrationTestHelper::updateData(_access, _context, dataBefore);

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
	EXPECT_NO_THROW(IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(dataBefore, dataModified)));
}

/**
* Situation:
* 	- one cluster with one cell and one token is moved
* Fixed error: token was removed in DataConverter::processModifications
* Expected result: token is still there
*/
TEST_F(DataDescriptionTransferGpuTests, regressionTestMoveCellWithToken)
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(1, QVector2D{}, QVector2D{}, 0);
    auto& cell = cluster.cells->at(0);
    auto token = createSimpleToken();
    cell.addToken(token);
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, _context, origData);

    DataDescription changedData = origData;
    auto& changedCluster = changedData.clusters->at(0);
    setCenterPos(changedCluster, QVector2D{ 1.0f, 0 });
    IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(origData, changedData));

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    ASSERT_EQ(1, newData.clusters->size());

    auto const& newCluster = newData.clusters->at(0);
    EXPECT_EQ(1, newCluster.cells->size());

    auto const& newCell = newCluster.cells->at(0);
    EXPECT_EQ(1, newCell.tokens->size());
}

/**
* Situation:
* 	- two cluster with one cell each and one token
*   - move one cluster by updating only a part of the universe where that cluster is situated
* Fixed error: tokens were not correctly filtered in AccessKernel
* Expected result: changes are correctly transferred to simulation
*/
TEST_F(DataDescriptionTransferGpuTests, regressionTestMoveCellWithToken_partialUpdate)
{
    auto token = createSimpleToken();

    DataDescription origData;
    for (int i = 0; i < 2; ++i) {
        auto cluster = createHorizontalCluster(1, QVector2D{ static_cast<float>(10 + 40 * i), 0 }, QVector2D{}, 0);
        auto& firstCell = cluster.cells->at(0);
        firstCell.addToken(token);
        origData.addCluster(cluster);
    }
    IntegrationTestHelper::updateData(_access, _context, origData);

    auto intermediateData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ 20, _universeSize.y } });
    auto changedData = intermediateData;
    auto& changedCluster = changedData.clusters->at(0);
    setCenterPos(changedCluster, QVector2D{ 5, 0 });
    IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(intermediateData, changedData));

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    ASSERT_EQ(2, newData.clusters->size());

    int numToken = 0;
    for (auto const& cluster : *newData.clusters) {
        for (auto const& cell : *cluster.cells) {
            if (cell.tokens) {
                numToken += cell.tokens->size();
            }
        }
    }
    EXPECT_EQ(2, numToken);
}

/**
* Situation: partial update and running simulation several times
* Fixed error: particles with same id emerged (due to particles array swap in setSimulationAccessData)
*/
TEST_F(DataDescriptionTransferGpuTests, regressionTestRepeatingPartialUpdateAndRun)
{
    DataDescription origData;
    origData.addCluster(createRectangularCluster({ 10, 10 }, QVector2D{ 100, 100 }, QVector2D{}));
    IntegrationTestHelper::updateData(_access, _context, origData);

    for (int i = 0; i < 10; ++i) {
        auto const data = IntegrationTestHelper::getContent(_access, { { 70, 70 },{ 130, 130 } });
        auto newData = data;
        *newData.clusters->at(0).angularVel = _numberGen->getRandomReal(-0.01, 0.01);
        IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(data, newData));
        IntegrationTestHelper::runSimulation(100, _controller);
    }

    auto const newData = IntegrationTestHelper::getContent(_access, { { 70, 70 },{ 130, 130 } });
    std::unordered_set<uint64_t> ids;
    if (newData.particles) {
        for (auto const& particle : *newData.particles) {
            ASSERT_TRUE(ids.find(particle.id) == ids.end());
            ids.insert(particle.id);
        }
    }
}

namespace
{
    EngineGpuData getEngineGpuDataForMinClusterArraySizes()
    {
        CudaConstants cudaConstants;
        cudaConstants.NUM_THREADS_PER_BLOCK = 32;
        cudaConstants.NUM_BLOCKS = 32;
        cudaConstants.MAX_CLUSTERS = 3;
        cudaConstants.MAX_CELLS = 100;
        cudaConstants.MAX_PARTICLES = 10;
        cudaConstants.MAX_TOKENS = 10;
        cudaConstants.MAX_CELLPOINTERS = 100 * 10;
        cudaConstants.MAX_CLUSTERPOINTERS = 10 * 10;
        cudaConstants.MAX_PARTICLEPOINTERS = 100;
        cudaConstants.MAX_TOKENPOINTERS = 100;
        cudaConstants.DYNAMIC_MEMORY_SIZE = 1000;
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 1000;
        return EngineGpuData(cudaConstants);
    }
}

class DataDescriptionTransferGpuTestsWithMinClusterArraySizes : public IntegrationGpuTestFramework
{
public:
    DataDescriptionTransferGpuTestsWithMinClusterArraySizes()
        : IntegrationGpuTestFramework({1000, 1000}, getEngineGpuDataForMinClusterArraySizes())
    {}

    virtual ~DataDescriptionTransferGpuTestsWithMinClusterArraySizes() = default;
};

TEST_F(DataDescriptionTransferGpuTestsWithMinClusterArraySizes, testMaxCluster)
{
    {
        DataDescription data;
        data.addCluster(createHorizontalCluster(1, QVector2D{}, QVector2D{}, 0));
        data.addCluster(createHorizontalCluster(1, QVector2D{}, QVector2D{}, 0));
        IntegrationTestHelper::updateData(_access, _context, data);
    }
	{
        DataDescription data;
        data.addCluster(createHorizontalCluster(1, QVector2D{}, QVector2D{}, 0));
        IntegrationTestHelper::updateData(_access, _context, data);
    }
}
