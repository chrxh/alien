#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/GenomeDescriptions.h"

class TransmitterTests : public IntegrationTestFramework
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

    TransmitterTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~TransmitterTests() = default;
};

TEST_F(TransmitterTests, distributeToOtherTransmitter)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(TransmitterDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors))
            .setEnergy(_parameters.cellNormalEnergy[0] * 2),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription()),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();

    auto origTransmitterCell1 = getCell(data, 1);
    auto actualTransmitterCell1 = getCell(actualData, 1);

    auto origTransmitterCell2 = getCell(data, 3);
    auto actualTransmitterCell2 = getCell(actualData, 3);

    auto origNerveCell = getCell(data, 2);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_TRUE(approxCompare(0.0f, actualTransmitterCell1.activity.channels[0]));
    EXPECT_TRUE(actualTransmitterCell1.energy < origTransmitterCell1.energy - NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell2.energy > origTransmitterCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(origNerveCell.energy, actualNerveCell.energy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(TransmitterTests, distributeToOneOtherTransmitter_forwardActivity)
{
    ActivityDescription activity;
    activity.setChannels({0.5f, -0.7f, 0, 0, 0, 0, 0, 0});

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(TransmitterDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors))
            .setEnergy(_parameters.cellNormalEnergy[0] * 2),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setActivity(activity),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();

    auto origTransmitterCell1 = getCell(data, 1);
    auto actualTransmitterCell1 = getCell(actualData, 1);

    auto origTransmitterCell2 = getCell(data, 3);
    auto actualTransmitterCell2 = getCell(actualData, 3);

    auto origNerveCell = getCell(data, 2);
    auto actualNerveCell = getCell(actualData, 2);

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        EXPECT_TRUE(approxCompare(activity.channels[i], actualTransmitterCell1.activity.channels[i]));
    }
    EXPECT_TRUE(actualTransmitterCell1.energy < origTransmitterCell1.energy - NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell2.energy > origTransmitterCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(origNerveCell.energy, actualNerveCell.energy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(TransmitterTests, distributeToConnectedCells)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(TransmitterDescription().setMode(EnergyDistributionMode_ConnectedCells))
            .setEnergy(_parameters.cellNormalEnergy[0] * 2),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription()),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();

    auto origTransmitterCell1 = getCell(data, 1);
    auto actualTransmitterCell1 = getCell(actualData, 1);

    auto origTransmitterCell2 = getCell(data, 3);
    auto actualTransmitterCell2 = getCell(actualData, 3);

    auto origNerveCell = getCell(data, 2);
    auto actualNerveCell = getCell(actualData, 2);

    EXPECT_TRUE(actualTransmitterCell1.energy < origTransmitterCell1.energy - NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell2.energy > origTransmitterCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(actualNerveCell.energy > origNerveCell.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(TransmitterTests, distributeToOtherTransmitterAndConstructor)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(TransmitterDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors))
            .setEnergy(_parameters.cellNormalEnergy[0] * 2),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(ConstructorDescription().setGenome(genome)),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();

    auto origTransmitterCell = getCell(data, 1);
    auto actualTransmitterCell = getCell(actualData, 1);

    auto origConstructorCell = getCell(data, 2);
    auto actualConstructorCell = getCell(actualData, 2);

    auto origOtherTransmitterCell = getCell(data, 3);
    auto actualOtherTransmitterCell = getCell(actualData, 3);

    EXPECT_TRUE(actualTransmitterCell.energy < origTransmitterCell.energy - NEAR_ZERO);
    EXPECT_TRUE(actualConstructorCell.energy > origConstructorCell.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(actualOtherTransmitterCell.energy, origOtherTransmitterCell.energy));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(TransmitterTests, distributeOnlyToActiveConstructors)
{
    auto genome = GenomeDescription().setHeader(GenomeHeaderDescription().setNumBranches(1));
    
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(TransmitterDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors))
            .setEnergy(_parameters.cellNormalEnergy[0] * 2),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(ConstructorDescription().setGenome(GenomeDescriptionService::convertDescriptionToBytes(genome))),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();

    auto origTransmitterCell = getCell(data, 1);
    auto actualTransmitterCell = getCell(actualData, 1);

    auto origConstructorCell = getCell(data, 2);
    auto actualConstructorCell = getCell(actualData, 2);

    auto origOtherTransmitterCell = getCell(data, 3);
    auto actualOtherTransmitterCell = getCell(actualData, 3);

    EXPECT_TRUE(actualTransmitterCell.energy < origTransmitterCell.energy - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(actualConstructorCell.energy, origConstructorCell.energy));
    EXPECT_TRUE(actualOtherTransmitterCell.energy > origOtherTransmitterCell.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(TransmitterTests, distributeToTwoTransmittersWithDifferentColor)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(TransmitterDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors))
            .setEnergy(_parameters.cellNormalEnergy[0] * 2),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(2).setExecutionOrderNumber(5).setCellFunction(TransmitterDescription()).setColor(1),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();

    auto origTransmitterCell = getCell(data, 1);
    auto actualTransmitterCell = getCell(actualData, 1);

    auto origOtherTransmitterCell1 = getCell(data, 2);
    auto actualOtherTransmitterCell1 = getCell(actualData, 2);

    auto origOtherTransmitterCell2 = getCell(data, 3);
    auto actualOtherTransmitterCell2 = getCell(actualData, 3);

    EXPECT_TRUE(actualTransmitterCell.energy < origTransmitterCell.energy - NEAR_ZERO);
    EXPECT_TRUE(actualOtherTransmitterCell1.energy > origOtherTransmitterCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(actualOtherTransmitterCell2.energy > origOtherTransmitterCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
