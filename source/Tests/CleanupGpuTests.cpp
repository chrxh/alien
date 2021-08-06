
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/ServiceLocator.h"
#include "EngineGpu/EngineGpuBuilderFacade.h"
#include "EngineGpu/EngineGpuData.h"
#include "EngineGpu/SimulationAccessGpu.h"
#include "EngineGpu/SimulationControllerGpu.h"
#include "EngineInterface/DescriptionFactory.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpaceProperties.h"
#include "IntegrationGpuTestFramework.h"
#include "IntegrationTestHelper.h"
#include "Predicates.h"

class CleanupGpuTests : public IntegrationGpuTestFramework
{
public:
    CleanupGpuTests()
        : IntegrationGpuTestFramework({100, 100}, getModelDataForCleanup())
    {}

    virtual ~CleanupGpuTests() = default;

private:
    EngineGpuData getModelDataForCleanup()
    {
        CudaConstants cudaConstants;
        cudaConstants.NUM_THREADS_PER_BLOCK = 64;
        cudaConstants.NUM_BLOCKS = 64;
        cudaConstants.MAX_CLUSTERS = 2500;
        cudaConstants.MAX_CELLS = 3500;
        cudaConstants.MAX_PARTICLES = 25000;
        cudaConstants.MAX_TOKENS = 500;
        cudaConstants.MAX_CELLPOINTERS = cudaConstants.MAX_CELLS * 10;
        cudaConstants.MAX_CLUSTERPOINTERS = cudaConstants.MAX_CLUSTERS * 10;
        cudaConstants.MAX_PARTICLEPOINTERS = cudaConstants.MAX_PARTICLES * 10;
        cudaConstants.MAX_TOKENPOINTERS = cudaConstants.MAX_TOKENS * 10;
        cudaConstants.DYNAMIC_MEMORY_SIZE = 10000000;
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 10000;
        return EngineGpuData(cudaConstants);
    }
};


TEST_F(CleanupGpuTests, testCleanupCells)
{
    _parameters.cellFusionVelocity = 0;
    _parameters.radiationFactor = 0.002f;
    _context->setSimulationParameters(_parameters);

    DataDescription dataBefore;

    auto createRect = [&](auto const& pos, auto const& vel) {
        auto result = _factory->createRect(
            DescriptionFactory::CreateRectParameters().size({30, 25}).centerPosition(pos).velocity(vel),
            _context->getNumberGenerator());
        return result;
    };
    dataBefore.addCluster(createRect(QVector2D(10, 10), QVector2D(1, 0)));
    dataBefore.addCluster(createRect(QVector2D(50, 10), QVector2D(-1, 0)));
    dataBefore.addCluster(createRect(QVector2D(80, 40), QVector2D(0, -1)));

    IntegrationTestHelper::updateData(_access, _context, dataBefore);

    IntegrationTestHelper::runSimulation(2500, _controller);
    DataDescription dataAfter =
        IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
/*
    if (dataAfter.clusters) {
        DescriptionNavigator navi;
        navi.update(dataAfter);
        for (auto const& cluster : *dataAfter.clusters) {
            for (auto const& cell : *cluster.cells) {
                for (auto const& connection : *cell.connections) {
                    int connectedCellIndex = navi.cellIndicesByCellIds.at(connection.cellId);
                    auto connectedCell = cluster.cells->at(connectedCellIndex);
                    auto displacement = *connectedCell.pos - *cell.pos;
                    _spaceProp->correctDisplacement(displacement);
                    EXPECT_TRUE(displacement.length() < 7);
                }
            }
        }
    }
*/
}

TEST_F(CleanupGpuTests, testCleanupCellsWithToken)
{
    _parameters.radiationFactor = 0;
    _parameters.radiationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription dataBefore;

    auto createRect = [&](auto const& pos, auto const& vel) {
        auto result = _factory->createRect(
            DescriptionFactory::CreateRectParameters().size({55, 55}).centerPosition(pos).velocity(vel),
            _context->getNumberGenerator());
        return result;
    };
    auto rect = createRect(QVector2D(50, 50), QVector2D(0, 0));
    QByteArray expectedTokenMemory(_parameters.tokenMemorySize, 0);
    rect.cells->at(0).addToken(createSimpleToken());
    rect.cells->at(1).tokenBranchNumber = 1;
    dataBefore.addCluster(rect);

    IntegrationTestHelper::updateData(_access, _context, dataBefore);
    IntegrationTestHelper::runSimulation(1, _controller);
    DataDescription dataAfter =
        IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    auto beforeAndAfterCells = IntegrationTestHelper::getBeforeAndAfterCells(dataBefore, dataAfter);

    auto firstCellId = dataBefore.clusters->at(0).cells->at(1).id;
    for (auto const& [cellBefore, cellAfter] : beforeAndAfterCells) {
        if (cellBefore && cellBefore->id == firstCellId) {
            EXPECT_EQ(1, cellAfter->tokens->size());
        }
    }
}

/**
 * Situation: cluster where a token is moving in a cycle and branches
 * Expected result: no crash during the number of tokens of all times is growing
 */
TEST_F(CleanupGpuTests, testCleanupTokens)
{
    _parameters.radiationProb = 0;  //exclude radiation
    _parameters.cellMaxTokenBranchNumber = 4;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;

    auto token = createSimpleToken();
    auto cluster =
        _factory->createRect(
        DescriptionFactory::CreateRectParameters().size({3, 2}),
        _context->getNumberGenerator());
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    auto& thirdCell = cluster.cells->at(3);
    auto& fourthCell = cluster.cells->at(2);
    auto& fifthCell = cluster.cells->at(4);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    thirdCell.tokenBranchNumber = 2;
    fourthCell.tokenBranchNumber = 3;
    fifthCell.tokenBranchNumber = 0;
    firstCell.addToken(token);
    origData.addCluster(cluster);
    IntegrationTestHelper::updateData(_access, _context, origData);
    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(4000, _controller));
}


/*
void CleanupGpuTests::SetUp()
{
    _parameters.radiationExponent = 1;
    _parameters.radiationFactor = 0.0002f;
    _parameters.radiationProb = 0.03f;
    _context->setSimulationParameters(_parameters);
}

EngineGpuData CleanupGpuTests::getModelDataForCleanup()
{
    CudaConstants cudaConstants;
    cudaConstants.NUM_THREADS_PER_BLOCK = 64;
    cudaConstants.NUM_BLOCKS = 64;
    cudaConstants.MAX_CLUSTERS = 1000;
    cudaConstants.MAX_CELLS = 1000;
    cudaConstants.MAX_PARTICLES = 10000;
    cudaConstants.MAX_TOKENS = 100;
    cudaConstants.MAX_CELLPOINTERS = 1000 * 10;
    cudaConstants.MAX_CLUSTERPOINTERS = 1000 * 10;
    cudaConstants.MAX_PARTICLEPOINTERS = 1000 * 10;
    cudaConstants.MAX_TOKENPOINTERS = 100 * 10;
    cudaConstants.DYNAMIC_MEMORY_SIZE = 1000000;
    cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 10000;
    return EngineGpuData(cudaConstants);
}

/ **
* Situation: cluster emitting particles
* Expected result: no crash during the number of particles of all times is growing
* /
TEST_F(CleanupGpuTests, testCleanupParticles)
{
    _parameters.radiationExponent = 1;
    _parameters.radiationFactor = 0.0002f;
    _parameters.radiationProb = 0.3f;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    origData.addCluster(createRectangularCluster({10, 10}, QVector2D{}, QVector2D{0, 0}));
    IntegrationTestHelper::updateData(_access, _context, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(1000, _controller));
}

/ **
* Situation: few large clusters
* Expected result: no crash during the number of cells of all times is growing
* /
TEST_F(CleanupGpuTests, testCleanupCells)
{
    _parameters.radiationExponent = 1;
    _parameters.radiationFactor = 0.0002f;
    _parameters.radiationProb = 0.003f;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 9; ++i) {
        origData.addCluster(createRectangularCluster({10, 10}));
    }
    IntegrationTestHelper::updateData(_access, _context, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(2000, _controller));
}

/ **
* Situation: many small clusters
* Expected result: no crash during the number of clusters of all times is growing
* /
TEST_F(CleanupGpuTests, testCleanupClusters)
{
    _parameters.radiationProb = 0;
    _parameters.cellFusionVelocity = 100.0f;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 900; ++i) {
        origData.addCluster(createRectangularCluster({1, 1}));
    }
    IntegrationTestHelper::updateData(_access, _context, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(2000, _controller));
}

/ **
* Situation: few large fast moving clusters; radiate much energy with low number of particles
* Expected result: no crash during the number of cell pointers of all times is growing
* /
TEST_F(CleanupGpuTests, testCleanupCellPointers)
{
    _parameters.radiationExponent = 1;
    _parameters.radiationFactor = 0.02f;
    _parameters.radiationProb = 0.003f;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 5; ++i) {
        QVector2D vel(_numberGen->getRandomReal(-3, 3), _numberGen->getRandomReal(-4, 4));
        origData.addCluster(createRectangularCluster({10, 10}, boost::none, vel));
    }
    IntegrationTestHelper::updateData(_access, _context, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(2000, _controller));
}

/ **
* Situation: cluster where a token is moving in a cycle
* Expected result: no crash during the number of token pointers of all times is growing
* /
TEST_F(CleanupGpuTests, testCleanupTokenPointers)
{
    _parameters.radiationProb = 0;  //exclude radiation
    _parameters.cellMaxTokenBranchNumber = 4;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;

    auto token = createSimpleToken();
    auto cluster = createRectangularCluster({2, 2}, QVector2D{}, QVector2D{});
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    auto& thirdCell = cluster.cells->at(3);
    auto& fourthCell = cluster.cells->at(2);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    thirdCell.tokenBranchNumber = 2;
    fourthCell.tokenBranchNumber = 3;
    firstCell.addToken(token);
    origData.addCluster(cluster);
    IntegrationTestHelper::updateData(_access, _context, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(1100, _controller));
}

/ **
* Situation: cluster where a token is moving in a cycle and branches
* Expected result: no crash during the number of tokens of all times is growing
* /
TEST_F(CleanupGpuTests, testCleanupTokens)
{
    _parameters.radiationProb = 0;  //exclude radiation
    _parameters.cellMaxTokenBranchNumber = 4;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;

    auto token = createSimpleToken();
    auto cluster = createRectangularCluster({2, 3}, QVector2D{}, QVector2D{});
    auto& firstCell = cluster.cells->at(0);
    auto& secondCell = cluster.cells->at(1);
    auto& thirdCell = cluster.cells->at(3);
    auto& fourthCell = cluster.cells->at(2);
    auto& fifthCell = cluster.cells->at(4);
    firstCell.tokenBranchNumber = 0;
    secondCell.tokenBranchNumber = 1;
    thirdCell.tokenBranchNumber = 2;
    fourthCell.tokenBranchNumber = 3;
    fifthCell.tokenBranchNumber = 0;
    firstCell.addToken(token);
    origData.addCluster(cluster);
    IntegrationTestHelper::updateData(_access, _context, origData);
    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(440, _controller));
}

/ **
* Situation: one moving clusters and a particles which crosses old position of cluster
* Expected result: particle is not absorbed
* /
TEST_F(CleanupGpuTests, testCleanupCellMap)
{
    _parameters.radiationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    origData.addCluster(createRectangularCluster({2, 2}, QVector2D{0, 10}, QVector2D{0.5f, 0}));
    origData.addParticle(createParticle(QVector2D{5, 0}, QVector2D{0, 0.5f}));
    IntegrationTestHelper::updateData(_access, _context, origData);

    IntegrationTestHelper::runSimulation(30, _controller);
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(1, newData.particles->size());
}

/ **
* Situation: two moving particles where one particle crosses old position of the other one
* Expected result: particles do not fuse
* /
TEST_F(CleanupGpuTests, testCleanupParticleMap)
{
    DataDescription origData;
    origData.addParticle(createParticle(QVector2D{0, 10}, QVector2D{0.5f, 0}));
    origData.addParticle(createParticle(QVector2D{5, 0}, QVector2D{0, 0.5f}));
    IntegrationTestHelper::updateData(_access, _context, origData);

    IntegrationTestHelper::runSimulation(30, _controller);
    auto const newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(2, newData.particles->size());
}

TEST_F(CleanupGpuTests, testCleanupMetadata)
{
    DataDescription prevData;
    prevData.addCluster(createSingleCellCluster(_numberGen->getId(), _numberGen->getId()));

    DataDescription data;
    for (int i = 0; i < 100; ++i) {
        EXPECT_NO_THROW(IntegrationTestHelper::updateData(_access, _context, DataChangeDescription(data, prevData)));
        data = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
        checkCompatibility(prevData, data);

        //generate new metadata
        prevData.clusters->at(0).cells->at(0).setMetadata(CellMetadata().setSourceCode(
            QString(100, QChar('d')) + QString("%1").arg(i)));  //exceeding 10k byte memory after some iterations
        *prevData.clusters->at(0).cells->at(0).energy = i;
    }
}
*/
