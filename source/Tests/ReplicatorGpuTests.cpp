#include "Base/ServiceLocator.h"
#include "EngineInterface/QuantityConverter.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SerializationHelper.h"

#include "IntegrationGpuTestFramework.h"

namespace
{
    EngineGpuData getEngineGpuDataForReplicatorGpuTests()
    {
        CudaConstants cudaConstants;
        cudaConstants.NUM_THREADS_PER_BLOCK = 16;
        cudaConstants.NUM_BLOCKS = 512;
        cudaConstants.MAX_CLUSTERS = 100000;
        cudaConstants.MAX_CELLS = 2000000;
        cudaConstants.MAX_PARTICLES = 2000000;
        cudaConstants.MAX_TOKENS = 500000;
        cudaConstants.MAX_CELLPOINTERS = 2000000 * 10;
        cudaConstants.MAX_CLUSTERPOINTERS = 100000 * 10;
        cudaConstants.MAX_PARTICLEPOINTERS = 2000000 * 10;
        cudaConstants.MAX_TOKENPOINTERS = 500000 * 10;
        cudaConstants.DYNAMIC_MEMORY_SIZE = 200000000;
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 100000000;
        return EngineGpuData(cudaConstants);
    }
}

class ReplicatorGpuTests
    : public IntegrationGpuTestFramework
{
public:
    ReplicatorGpuTests()
        : IntegrationGpuTestFramework({ 1000, 1000 }, getEngineGpuDataForReplicatorGpuTests())
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
        auto filename = string{ "..\\..\\..\\..\\source\\Tests\\TestData\\replicator.par" };
        SerializationHelper::loadFromFile<SimulationParameters>(
            filename, [&](string const& data) { return serializer->deserializeSimulationParameters(data); }, parameters);
    }
    _context->setSimulationParameters(parameters);

    DataDescription loadData;
    auto filename = string{ "..\\..\\..\\..\\source\\Tests\\TestData\\replicator.aco" };
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

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1000, _controller);
}

TEST_F(ReplicatorGpuTests, testManyConcentratedReplicators)
{
    auto serializer = _basicFacade->buildSerializer();

    SimulationParameters parameters;
    {
        auto filename = string{ "..\\..\\..\\..\\source\\Tests\\TestData\\replicator.par" };
        SerializationHelper::loadFromFile<SimulationParameters>(
            filename, [&](string const& data) { return serializer->deserializeSimulationParameters(data); }, parameters);
    }
    _context->setSimulationParameters(parameters);

    DataDescription loadData;
    auto filename = string{ "..\\..\\..\\..\\source\\Tests\\TestData\\replicator.aco" };
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

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(100, _controller);
}

namespace
{
    EngineGpuData getEngineGpuDataWithManyThreads()
    {
        CudaConstants cudaConstants;
        cudaConstants.NUM_THREADS_PER_BLOCK = 256;
        cudaConstants.NUM_BLOCKS = 128;
        cudaConstants.MAX_CLUSTERS = 100000;
        cudaConstants.MAX_CELLS = 500000;
        cudaConstants.MAX_PARTICLES = 500000;
        cudaConstants.MAX_TOKENS = 50000;
        cudaConstants.MAX_CELLPOINTERS = 500000 * 10;
        cudaConstants.MAX_CLUSTERPOINTERS = 100000 * 10;
        cudaConstants.MAX_PARTICLEPOINTERS = 500000 * 10;
        cudaConstants.MAX_TOKENPOINTERS = 50000 * 10;
        cudaConstants.DYNAMIC_MEMORY_SIZE = 100000000;
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 1000;
        return EngineGpuData(cudaConstants);
    }
}

class ReplicatorGpuTestsWithManyThreads : public IntegrationGpuTestFramework
{
public:
    ReplicatorGpuTestsWithManyThreads()
        : IntegrationGpuTestFramework({ 1000, 1000 }, getEngineGpuDataForReplicatorGpuTests())
    { }

    virtual ~ReplicatorGpuTestsWithManyThreads() = default;
};

TEST_F(ReplicatorGpuTestsWithManyThreads, testManyReplicators)
{
    auto serializer = _basicFacade->buildSerializer();

    SimulationParameters parameters;
    {
        auto filename = string{ "..\\..\\..\\..\\source\\Tests\\TestData\\dna-replicator.par" };
        SerializationHelper::loadFromFile<SimulationParameters>(
            filename, [&](string const& data) { return serializer->deserializeSimulationParameters(data); }, parameters);
    }
    _context->setSimulationParameters(parameters);

    DataDescription loadData;
    {
        auto filename = string{ "..\\..\\..\\..\\source\\Tests\\TestData\\dna-replicator.aco" };
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

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(5000, _controller);
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    check(origData, newData);
}

