#include "Base/ServiceLocator.h"
#include "ModelBasic/QuantityConverter.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/SerializationHelper.h"

#include "IntegrationGpuTestFramework.h"

namespace
{
    ModelGpuData getModelGpuDataForReplicatorGpuTests()
    {
        ModelGpuData result;
        result.setNumThreadsPerBlock(16);
        result.setNumBlocks(512);
        result.setMaxClusters(100000);
        result.setMaxCells(2000000);
        result.setMaxParticles(2000000);
        result.setMaxTokens(500000);
        result.setMaxCellPointers(2000000 * 10);
        result.setMaxClusterPointers(100000 * 10);
        result.setMaxParticlePointers(2000000 * 10);
        result.setMaxTokenPointers(500000 * 10);
        result.setDynamicMemorySize(200000000);
        result.setStringByteSize(10000000);
        return result;
    }
}

class ReplicatorGpuTests
    : public IntegrationGpuTestFramework
{
public:
    ReplicatorGpuTests()
        : IntegrationGpuTestFramework({ 1000, 1000 }, getModelGpuDataForReplicatorGpuTests())
    {}

    virtual ~ReplicatorGpuTests() = default;

protected:
    virtual void SetUp() {}
};

/**
* Situation: - many concentrated replicator clusters
*			 - simulating 100 time steps
* Expected result: no crash
*/
TEST_F(ReplicatorGpuTests, regressionTestManyConcentratedReplicators)
{
    DataDescription loadData;
    auto serializer = _basicFacade->buildSerializer();
    auto filename = string{ "..\\..\\source\\Tests\\TestData\\replicator.aco" };
    SerializationHelper::loadFromFile<DataDescription>(
        filename, [&](string const& data) { return serializer->deserializeDataDescription(data); }, loadData);

    auto& replicator = loadData.clusters->at(0);

    DataDescription origData;
    for (int i = 0; i < 400; ++i) {
        setCenterPos(replicator, QVector2D{ static_cast<float>(_numberGen->getRandomReal(0, 30)),
            static_cast<float>(_numberGen->getRandomReal(0, 30)) });
        replicator.vel = QVector2D{ static_cast<float>(_numberGen->getRandomReal(-0.9, 0.9)),
            static_cast<float>(_numberGen->getRandomReal(-0.9, 0.9f)) };
        replicator.angularVel = _numberGen->getRandomReal(-1, 1);
        _descHelper->makeValid(replicator);
        origData.addCluster(replicator);
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(100, _controller);
}

/**
* Situation: many advanced replicators
* Expected result: no crash
* Fixed errors:
*   - energy is never negative in cells (is not checked here but can be checked in Cell::changeEnergy)
*   - constructor accessed connections of cells from other clusters with no locking
*/
TEST_F(ReplicatorGpuTests, regressionTestManyAdvancedReplicators)
{
    DataDescription loadData;
    auto serializer = _basicFacade->buildSerializer();
    auto filename = string{ "..\\..\\source\\Tests\\TestData\\replicator2.aco" };
    SerializationHelper::loadFromFile<DataDescription>(
        filename, [&](string const& data) { return serializer->deserializeDataDescription(data); }, loadData);

    auto& replicator = loadData.clusters->at(0);

    DataDescription origData;
    for (int i = 0; i < 20000; ++i) {
        setCenterPos(replicator, QVector2D{ static_cast<float>(_numberGen->getRandomReal(0, _universeSize.x)),
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.y)) });
        replicator.vel = QVector2D{ static_cast<float>(_numberGen->getRandomReal(-0.09, 0.09)),
            static_cast<float>(_numberGen->getRandomReal(-0.09, 0.09f)) };
        replicator.angularVel = 0;
        _descHelper->makeValid(replicator);
        origData.addCluster(replicator);
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1000, _controller);
}

