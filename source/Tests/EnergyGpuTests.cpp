
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

class EnergyGpuTests : public IntegrationGpuTestFramework
{
public:
    EnergyGpuTests()
        : IntegrationGpuTestFramework({600, 300}, getModelDataForCleanup())
    {}

    virtual ~EnergyGpuTests() = default;

private:
    EngineGpuData getModelDataForCleanup()
    {
        CudaConstants cudaConstants;
        cudaConstants.NUM_THREADS_PER_BLOCK = 64;
        cudaConstants.NUM_BLOCKS = 64;
        cudaConstants.MAX_CLUSTERS = 100000;
        cudaConstants.MAX_CELLS = 100000;
        cudaConstants.MAX_PARTICLES = 100000;
        cudaConstants.MAX_TOKENS = 100000;
        cudaConstants.MAX_CELLPOINTERS = cudaConstants.MAX_CELLS * 10;
        cudaConstants.MAX_CLUSTERPOINTERS = cudaConstants.MAX_CLUSTERS * 10;
        cudaConstants.MAX_PARTICLEPOINTERS = cudaConstants.MAX_PARTICLES * 10;
        cudaConstants.MAX_TOKENPOINTERS = cudaConstants.MAX_TOKENS * 10;
        cudaConstants.DYNAMIC_MEMORY_SIZE = 10000000;
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE = 10000;
        return EngineGpuData(cudaConstants);
    }
};


TEST_F(EnergyGpuTests, testEnergyConservation)
{
    _parameters.cellFusionVelocity = 0;
    _parameters.radiationFactor = 0.002f;
    _context->setSimulationParameters(_parameters);

    auto const factory = ServiceLocator::getInstance().getService<DescriptionFactory>();

    DataDescription dataBefore;

    auto createRect = [&](auto const& pos, auto const& vel) {
        auto result = factory->createRect(
            DescriptionFactory::CreateRectParameters().size({25, 25}).centerPosition(pos).velocity(vel),
            _context->getNumberGenerator());
        for (auto& cell : *result.cells) {
            cell.maxConnections = cell.connections->size();
        }
        return result;
    };

    for (int i = 0; i < 1; ++i) {
        dataBefore.addCluster(createRect(
            QVector2D(_numberGen->getRandomReal(0, _universeSize.x), _numberGen->getRandomReal(0, _universeSize.y)),
            QVector2D(_numberGen->getRandomReal(-1, 1), _numberGen->getRandomReal(-1, 1))));
    }

    IntegrationTestHelper::updateData(_access, _context, dataBefore);
    IntegrationTestHelper::runSimulation(3000, _controller);
    DataDescription dataAfter = getDataFromSimulation();

    auto energyBefore = getEnergy(dataBefore);
    auto energyAfter = getEnergy(dataAfter);

    EXPECT_TRUE(abs(energyAfter - energyBefore) < 0.1);
}
