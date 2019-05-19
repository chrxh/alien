#include <QElapsedTimer>

#include "SimulationGpuTest.h"

class BenchmarkGpu
    : public SimulationGpuTest
{
public:
    BenchmarkGpu() : SimulationGpuTest({ 2000, 1000 })
    {}

    virtual ~BenchmarkGpu() = default;
};

TEST_F(BenchmarkGpu, testClusterAndParticleMovement)
{
    DataDescription origData;
    for (int i = 0; i < 1000; ++i) {
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
    IntegrationTestHelper::runSimulation(1000, _controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms"<< std::endl;
}

TEST_F(BenchmarkGpu, testOnlyClusterMovement)
{
    _parameters.radiationProb = 0;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 1000; ++i) {
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
    IntegrationTestHelper::runSimulation(1000, _controller);
    std::cout << "Time elapsed during simulation: " << timer.elapsed() << " ms" << std::endl;
}
