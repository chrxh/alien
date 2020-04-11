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
        result.setStringByteSize(100000000);
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

TEST_F(ReplicatorGpuTests, testManyReplicators)
{
    auto serializer = _basicFacade->buildSerializer();

    SimulationParameters parameters;
    {
        auto filename = string{ "..\\..\\source\\Tests\\TestData\\replicator.par" };
        SerializationHelper::loadFromFile<SimulationParameters>(
            filename, [&](string const& data) { return serializer->deserializeSimulationParameters(data); }, parameters);
    }
    _context->setSimulationParameters(parameters);

    DataDescription loadData;
    auto filename = string{ "..\\..\\source\\Tests\\TestData\\replicator.aco" };
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

TEST_F(ReplicatorGpuTests, testManyConcentratedReplicators)
{
    auto serializer = _basicFacade->buildSerializer();

    SimulationParameters parameters;
    {
        auto filename = string{ "..\\..\\source\\Tests\\TestData\\replicator.par" };
        SerializationHelper::loadFromFile<SimulationParameters>(
            filename, [&](string const& data) { return serializer->deserializeSimulationParameters(data); }, parameters);
    }
    _context->setSimulationParameters(parameters);

    DataDescription loadData;
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

namespace
{
    ModelGpuData getModelGpuDataWithManyThreads()
    {
        ModelGpuData result;
        result.setNumThreadsPerBlock(256);
        result.setNumBlocks(128);
        result.setMaxClusters(100000);
        result.setMaxCells(500000);
        result.setMaxParticles(500000);
        result.setMaxTokens(50000);
        result.setMaxCellPointers(500000 * 10);
        result.setMaxClusterPointers(100000 * 10);
        result.setMaxParticlePointers(500000 * 10);
        result.setMaxTokenPointers(50000 * 10);
        result.setDynamicMemorySize(100000000);
        result.setStringByteSize(1000);
        return result;
    }
}

class ReplicatorGpuTestsWithManyThreads : public IntegrationGpuTestFramework
{
public:
    ReplicatorGpuTestsWithManyThreads()
        : IntegrationGpuTestFramework({ 1000, 1000 }, getModelGpuDataForReplicatorGpuTests())
    { }

    virtual ~ReplicatorGpuTestsWithManyThreads() = default;
};

TEST_F(ReplicatorGpuTestsWithManyThreads, testManyReplicators)
{
    auto serializer = _basicFacade->buildSerializer();

    SimulationParameters parameters;
    {
        auto filename = string{ "..\\..\\source\\Tests\\TestData\\dna-replicator.par" };
        SerializationHelper::loadFromFile<SimulationParameters>(
            filename, [&](string const& data) { return serializer->deserializeSimulationParameters(data); }, parameters);
    }
    _context->setSimulationParameters(parameters);

    DataDescription loadData;
    {
        auto filename = string{ "..\\..\\source\\Tests\\TestData\\dna-replicator.aco" };
        SerializationHelper::loadFromFile<DataDescription>(
            filename, [&](string const& data) { return serializer->deserializeDataDescription(data); }, loadData);
    }

    auto& replicator = loadData.clusters->at(0);

    DataDescription origData;
    for (int i = 0; i < 2500; ++i) {
        origData.addCluster(createRectangularCluster(
            {8, 4},
            boost::none,
            QVector2D(_numberGen->getRandomReal(-0.1, 0.1), _numberGen->getRandomReal(-0.1, 0.1))));
    }

    for (int i = 0; i < 20; ++i) {
        setCenterPos(replicator, QVector2D{ static_cast<float>(_numberGen->getRandomReal(0, _universeSize.x)),
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.y)) });
        replicator.vel = QVector2D{ static_cast<float>(_numberGen->getRandomReal(-0.09, 0.09)),
            static_cast<float>(_numberGen->getRandomReal(-0.09, 0.09f)) };
        replicator.angularVel = 0;
        _descHelper->makeValid(replicator);
        origData.addCluster(replicator);
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(5000, _controller);
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    check(origData, newData);
}

