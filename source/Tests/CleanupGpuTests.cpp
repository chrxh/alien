#include "IntegrationGpuTestFramework.h"

class CleanupGpuTests : public IntegrationGpuTestFramework
{
public:
    CleanupGpuTests()
        : IntegrationGpuTestFramework({600, 300}, getModelDataForCleanup())
    {}

    virtual ~CleanupGpuTests() = default;

protected:
    virtual void SetUp();

private:
    ModelGpuData getModelDataForCleanup();
};


void CleanupGpuTests::SetUp()
{
    _parameters.radiationExponent = 1;
    _parameters.radiationFactor = 0.0002f;
    _parameters.radiationProb = 0.03f;
    _context->setSimulationParameters(_parameters);
}

ModelGpuData CleanupGpuTests::getModelDataForCleanup()
{
    {
        ModelGpuData result;
        result.setNumThreadsPerBlock(64);
        result.setNumBlocks(64);
        result.setMaxClusters(1000);
        result.setMaxCells(1000);
        result.setMaxParticles(10000);
        result.setMaxTokens(100);
        result.setMaxCellPointers(10000);
        result.setMaxClusterPointers(10000);
        result.setMaxParticlePointers(10000);
        result.setMaxTokenPointers(1000);
        result.setDynamicMemorySize(1000000);
        return result;
    }
}

/**
* Situation: cluster emitting particles
* Expected result: no crash during the number of particles of all times is growing
*/
TEST_F(CleanupGpuTests, testCleanupParticles)
{
    _parameters.radiationExponent = 1;
    _parameters.radiationFactor = 0.0002f;
    _parameters.radiationProb = 0.3f;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    origData.addCluster(createRectangularCluster({10, 10}, QVector2D{}, QVector2D{0, 0}));
    IntegrationTestHelper::updateData(_access, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(1000, _controller));
}

/**
* Situation: few large clusters
* Expected result: no crash during the number of cells of all times is growing
*/
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
    IntegrationTestHelper::updateData(_access, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(2000, _controller));
}

/**
* Situation: many small clusters
* Expected result: no crash during the number of clusters of all times is growing
*/
TEST_F(CleanupGpuTests, testCleanupClusters)
{
    _parameters.radiationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 900; ++i) {
        origData.addCluster(createRectangularCluster({1, 1}));
    }
    IntegrationTestHelper::updateData(_access, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(2000, _controller));
}

/**
* Situation: few large fast moving clusters; radiate much energy with low number of particles
* Expected result: no crash during the number of cell pointers of all times is growing
*/
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
    IntegrationTestHelper::updateData(_access, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(2000, _controller));
}

/**
* Situation: cluster where a token is moving in a cycle
* Expected result: no crash during the number of token pointers of all times is growing
*/
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
    IntegrationTestHelper::updateData(_access, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(1100, _controller));
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
    IntegrationTestHelper::updateData(_access, origData);
    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(440, _controller));
}

/**
* Situation: one moving clusters and a particles which crosses old position of cluster
* Expected result: particle is not absorbed
*/
TEST_F(CleanupGpuTests, testCleanupCellMap)
{
    _parameters.radiationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    origData.addCluster(createRectangularCluster({2, 2}, QVector2D{0, 10}, QVector2D{0.5f, 0}));
    origData.addParticle(createParticle(QVector2D{5, 0}, QVector2D{0, 0.5f}));
    IntegrationTestHelper::updateData(_access, origData);

    IntegrationTestHelper::runSimulation(30, _controller);
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(1, newData.particles->size());
}

/**
* Situation: two moving particles where one particle crosses old position of the other one
* Expected result: particles do not fuse
*/
TEST_F(CleanupGpuTests, testCleanupParticleMap)
{
    DataDescription origData;
    origData.addParticle(createParticle(QVector2D{0, 10}, QVector2D{0.5f, 0}));
    origData.addParticle(createParticle(QVector2D{5, 0}, QVector2D{0, 0.5f}));
    IntegrationTestHelper::updateData(_access, origData);

    IntegrationTestHelper::runSimulation(30, _controller);
    auto const newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    ASSERT_EQ(2, newData.particles->size());
}

