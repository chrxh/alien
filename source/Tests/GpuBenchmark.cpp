#include <QElapsedTimer>

#include "SimulationGpuTestFramework.h"

class GpuBenchmark
    : public SimulationGpuTestFramework
{
public:
    GpuBenchmark() : SimulationGpuTestFramework({ 4000, 2000 })
    {}

    virtual ~GpuBenchmark() = default;
};

TEST_F(GpuBenchmark, testClusterAndParticleMovement)
{
    DataDescription origData;
    for (int i = 0; i < 3000; ++i) {
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

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(2000, _controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms"<< std::endl;
}

TEST_F(GpuBenchmark, testOnlyClusterMovement)
{
    _parameters.radiationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 3000; ++i) {
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

    QElapsedTimer timer;
    timer.start();
    IntegrationTestHelper::runSimulation(2000, _controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;
}
