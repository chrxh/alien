#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/GenomeDescriptions.h"

#include "IntegrationTestFramework.h"

class AttackerTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction = 0;
        result.baseValues.friction = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationCellAgeStrength[i] = 0;
        }
        return result;
    }
    AttackerTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~AttackerTests() = default;
};

TEST_F(AttackerTests, nothingFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(AttackerDescription()),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.signal.channels[0]));
}

TEST_F(AttackerTests, successNoTransmitter)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .setId(3)
            .setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto origAttackCell = getCell(data, 1);
    auto actualAttackCell = getCell(actualData, 1);

    auto origTargetCell = getCell(data, 3);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualAttackCell.signal.channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualAttackCell.energy > origAttackCell.energy + NEAR_ZERO);
    EXPECT_TRUE(actualTargetCell.energy < origTargetCell.energy - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToOneTransmitter)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({12.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription().setId(4).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    auto origNerveCell = getCell(data, 2);
    auto actualNerveCell = getCell(actualData, 2);

    auto origTransmitterCell = getCell(data, 3);
    auto actualTransmitterCell = getCell(actualData, 3);

    EXPECT_TRUE(actualAttackCell.signal.channels[0] > NEAR_ZERO);
    EXPECT_TRUE(approxCompare(origNerveCell.energy, actualNerveCell.energy));
    EXPECT_TRUE(actualTransmitterCell.energy > origTransmitterCell.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToTwoTransmitters)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({12.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription().setId(4).setPos({11.0f, 9.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription().setId(5).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 4);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);
    auto origTransmitterCell1 = getCell(data, 3);
    auto actualTransmitterCell1 = getCell(actualData, 3);
    auto origTransmitterCell2 = getCell(data, 4);
    auto actualTransmitterCell2 = getCell(actualData, 4);

    EXPECT_TRUE(actualAttackCell.signal.channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell1.energy > origTransmitterCell1.energy + NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell2.energy > origTransmitterCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToTwoTransmittersWithDifferentColor)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({12.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription().setId(4).setPos({11.0f, 9.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()).setColor(1),
        CellDescription().setId(5).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 4);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);
    auto origTransmitterCell1 = getCell(data, 3);
    auto actualTransmitterCell1 = getCell(actualData, 3);
    auto origTransmitterCell2 = getCell(data, 4);
    auto actualTransmitterCell2 = getCell(actualData, 4);

    EXPECT_TRUE(actualAttackCell.signal.channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell1.energy > origTransmitterCell1.energy + NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell2.energy > origTransmitterCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}


TEST_F(AttackerTests, successDistributeToTransmitterAndConstructor)
{
    auto otherGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription().setMode(EnergyDistributionMode_TransmittersAndConstructors)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({12.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(TransmitterDescription()),
        CellDescription()
            .setId(4)
            .setPos({11.0f, 9.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(1)
            .setCellFunction(ConstructorDescription().setGenome(otherGenome)),
        CellDescription().setId(5).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 4);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);
    auto origTransmitterCell = getCell(data, 3);
    auto actualTransmitterCell = getCell(actualData, 3);
    auto origConstructorCell = getCell(data, 4);
    auto actualConstructorCell = getCell(actualData, 4);

    EXPECT_TRUE(actualAttackCell.signal.channels[0] > NEAR_ZERO);
    EXPECT_TRUE(approxCompare(actualTransmitterCell.energy, origTransmitterCell.energy));
    EXPECT_TRUE(actualConstructorCell.energy > origConstructorCell.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToConnectedCells)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription().setMode(EnergyDistributionMode_ConnectedCells)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({12.0f, 10.0f}).setMaxConnections(1).setExecutionOrderNumber(1).setCellFunction(NerveDescription()),
        CellDescription().setId(4).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    auto origNerveCell1 = getCell(data, 2);
    auto actualNerveCell1 = getCell(actualData, 2);

    auto origNerveCell2 = getCell(data, 3);
    auto actualNerveCell2 = getCell(actualData, 3);

    EXPECT_TRUE(actualAttackCell.signal.channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualNerveCell1.energy > origNerveCell1.energy + NEAR_ZERO);
    EXPECT_TRUE(actualNerveCell2.energy > origNerveCell2.energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successTwoTargets)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(2)
            .setExecutionOrderNumber(0)
            .setInputExecutionOrderNumber(5)
            .setCellFunction(AttackerDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setExecutionOrderNumber(5)
            .setCellFunction(NerveDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}),
        CellDescription().setId(4).setPos({9.0f, 11.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto origAttackCell = getCell(data, 1);
    auto actualAttackCell = getCell(actualData, 1);

    auto origTargetCell1 = getCell(data, 3);
    auto actualTargetCell1 = getCell(actualData, 3);

    auto origTargetCell2 = getCell(data, 4);
    auto actualTargetCell2 = getCell(actualData, 4);

    EXPECT_TRUE(actualAttackCell.signal.channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualAttackCell.energy > origAttackCell.energy + NEAR_ZERO);
    EXPECT_TRUE(actualTargetCell1.energy < origTargetCell1.energy - NEAR_ZERO);
    EXPECT_TRUE(actualTargetCell2.energy < origTargetCell2.energy - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
