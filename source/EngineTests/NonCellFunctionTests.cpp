#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class NonCellFunctionTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.cellFunctionTransmitterEnergyDistributionSameCreature = true;
        result.innerFriction = 0;
        result.baseValues.friction = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationCellAgeStrength[i] = 0;
        }
        return result;
    }

    NonCellFunctionTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~NonCellFunctionTests() = default;
};

TEST_F(NonCellFunctionTests, distributeToOtherTransmitter)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setActivity({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f})
            .setEnergy(_parameters.cellNormalEnergy[0]),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto actualCell = getCell(actualData, 1);

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        EXPECT_EQ(0, actualCell.activity.channels[i]);
    }
}
