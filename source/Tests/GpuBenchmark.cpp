#include <QElapsedTimer>

#include "IntegrationGpuTestFramework.h"

class GpuBenchmark
    : public IntegrationGpuTestFramework
{
public:
    GpuBenchmark(IntVector2D const& universeSize = { 1008, 504 }, optional<ModelGpuData> const& modelData = boost::none)
        : IntegrationGpuTestFramework(universeSize, modelData)
    {}

    virtual ~GpuBenchmark() = default;
};

TEST_F(GpuBenchmark, testClusterAndParticleMovement)
{
    DataDescription origData;
    for (int i = 0; i < 250; ++i) {
        origData.addCluster(createRectangularCluster({ 7, 40 },
            QVector2D{
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.x)),
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.y)) },
            QVector2D{
            static_cast<float>(_numberGen->getRandomReal(-1, 1)),
            static_cast<float>(_numberGen->getRandomReal(-1, 1)) }
        ));
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(400, _controller);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(200, _controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms"<< std::endl;
}

TEST_F(GpuBenchmark, testOnlyClusterMovement)
{
    _parameters.radiationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 250; ++i) {
        origData.addCluster(createRectangularCluster({ 7, 40 },
            QVector2D{
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.x)),
            static_cast<float>(_numberGen->getRandomReal(0, _universeSize.y)) },
            QVector2D{
            static_cast<float>(_numberGen->getRandomReal(-1, 1)),
            static_cast<float>(_numberGen->getRandomReal(-1, 1)) }
        ));
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(400, _controller);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(200, _controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;
}

namespace
{
    ModelGpuData getModelGpuDataWithOneBlock()
    {
        ModelGpuData result;
        result.setNumThreadsPerBlock(32);
        result.setNumBlocks(1);
        result.setMaxClusters(100);
        result.setMaxCells(1500000);
        result.setMaxParticles(500000);
        result.setMaxTokens(50000);
        result.setMaxCellPointers(500000 * 10);
        result.setMaxClusterPointers(100 * 10);
        result.setMaxParticlePointers(500000 * 10);
        result.setMaxTokenPointers(50000 * 10);
        result.setDynamicMemorySize(100000000);
        result.setMetadataDynamicMemorySize(1000);
        return result;
    }
}

class GpuBenchmarkForClusterDecomposition
    : public GpuBenchmark
{
public:
    GpuBenchmarkForClusterDecomposition() : GpuBenchmark({ 2010, 1000 }, getModelGpuDataWithOneBlock())
    {}

    virtual ~GpuBenchmarkForClusterDecomposition() = default;
};

TEST_F(GpuBenchmarkForClusterDecomposition, testClusterDecomposition)
{
    auto const lowEnergy = _parameters.cellMinEnergy / 2;
    DataDescription origData;
    auto cluster = createRectangularCluster({ 1000, 400 }, QVector2D{ 0, 0 }, QVector2D{});
    cluster.cells->at(10).energy = lowEnergy;
    cluster.cells->at(120).energy = lowEnergy;
    cluster.cells->at(5020).energy = lowEnergy;
    origData.addCluster(cluster);
    IntegrationTestHelper::updateData(_access, origData);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(1, _controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;
}
