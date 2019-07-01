#include "IntegrationGpuTestFramework.h"

class CleanupGpuTests
    : public IntegrationGpuTestFramework
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
        result.setNumClusterPointerArrays(1);
        result.setMaxClusters(1000);
        result.setMaxCells(1000);
        result.setMaxParticles(10000);
        result.setMaxTokens(1000);
        result.setMaxCellPointers(10000);
        result.setMaxClusterPointers(10000);
        result.setMaxParticlePointers(100000);
        result.setMaxTokenPointers(10000);
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
    origData.addCluster(createRectangularCluster({ 10, 10 }, QVector2D{}, QVector2D{ 0, 0 }));
    IntegrationTestHelper::updateData(_access, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(1000, _controller));
}

/**
* Situation: few large clusters
* Expected result: no crash during the number of cells of all times is growing
*/
TEST_F(CleanupGpuTests, testCleanupCells)
{
    DataDescription origData;
    for (int i = 0; i < 9; ++i) {
        origData.addCluster(createRectangularCluster({ 10, 10 }));
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
    DataDescription origData;
    for (int i = 0; i < 900; ++i) {
        origData.addCluster(createRectangularCluster({ 1, 1 }));
    }
    IntegrationTestHelper::updateData(_access, origData);

    EXPECT_NO_THROW(IntegrationTestHelper::runSimulation(2000, _controller));
}

