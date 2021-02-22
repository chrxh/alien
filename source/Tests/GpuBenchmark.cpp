#include <QElapsedTimer>

#include "IntegrationGpuTestFramework.h"

class GpuBenchmark
    : public IntegrationGpuTestFramework
{
public:
    GpuBenchmark(IntVector2D const& universeSize = { 1008, 504 }, boost::optional<EngineGpuData> const& modelData = boost::none)
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

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(400, _controller);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(200, _controller);
    std::cerr << "Time elapsed during simulation: " << timer.elapsed() << " ms"<< std::endl;
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

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(400, _controller);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(200, _controller);
    std::cerr << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;
}

namespace
{
    EngineGpuData getEngineGpuDataWithOneBlock()
    {
        CudaConstants cudaConstants;
        cudaConstants.NUM_THREADS_PER_BLOCK = 32;
        cudaConstants.NUM_BLOCKS = 1;
        cudaConstants.MAX_CLUSTERS = 100;
        cudaConstants.MAX_CELLS = 1500000;
        cudaConstants.MAX_PARTICLES = 500000;
        cudaConstants.MAX_TOKENS = 50000;
        cudaConstants.MAX_CELLPOINTERS = 500000 * 10;
        cudaConstants.MAX_CLUSTERPOINTERS = 100 * 10;
        cudaConstants.MAX_PARTICLEPOINTERS = 500000 * 10;
        cudaConstants.MAX_TOKENPOINTERS = 50000 * 10;
        cudaConstants.DYNAMIC_MEMORY_SIZE = 100000000;
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 1000;
        return EngineGpuData(cudaConstants);
    }
}

class GpuBenchmarkForClusterDecomposition
    : public GpuBenchmark
{
public:
    GpuBenchmarkForClusterDecomposition() : GpuBenchmark({ 2010, 1000 }, getEngineGpuDataWithOneBlock())
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
    IntegrationTestHelper::updateData(_access, _context, origData);

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(1, _controller);
    std::cerr << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;
}
