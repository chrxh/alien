#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/GenomeDescriptions.h"

#include "IntegrationTestFramework.h"

class AttackerTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction = 0;
        result.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationType1_strength[i] = 0;
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
             .id(1)
             .pos({10.0f, 10.0f})
             .cellType(AttackerDescription()),
         CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .cellType(OscillatorDescription())
             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData._cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell._signal->_channels[0]));
}

TEST_F(AttackerTests, successNoTransmitter)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription()
            .id(3)
            .pos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto origAttackCell = getCell(data, 1);
    auto actualAttackCell = getCell(actualData, 1);

    auto origTargetCell = getCell(data, 3);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualAttackCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualAttackCell._energy > origAttackCell._energy + NEAR_ZERO);
    EXPECT_TRUE(actualTargetCell._energy < origTargetCell._energy - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToOneTransmitter)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({12.0f, 10.0f}).cellType(DepotDescription()),
        CellDescription().id(4).pos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    auto origOscillatorCell = getCell(data, 2);
    auto actualOscillatorCell = getCell(actualData, 2);

    auto origTransmitterCell = getCell(data, 3);
    auto actualTransmitterCell = getCell(actualData, 3);

    EXPECT_TRUE(actualAttackCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_TRUE(approxCompare(origOscillatorCell._energy, actualOscillatorCell._energy));
    EXPECT_TRUE(actualTransmitterCell._energy > origTransmitterCell._energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToTwoTransmitters)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({12.0f, 10.0f}).cellType(DepotDescription()),
        CellDescription().id(4).pos({11.0f, 9.0f}).cellType(DepotDescription()),
        CellDescription().id(5).pos({9.0f, 10.0f}),
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

    EXPECT_TRUE(actualAttackCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell1._energy > origTransmitterCell1._energy + NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell2._energy > origTransmitterCell2._energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToTwoTransmittersWithDifferentColor)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({12.0f, 10.0f}).cellType(DepotDescription()),
        CellDescription().id(4).pos({11.0f, 9.0f}).cellType(DepotDescription()).color(1),
        CellDescription().id(5).pos({9.0f, 10.0f}),
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

    EXPECT_TRUE(actualAttackCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell1._energy > origTransmitterCell1._energy + NEAR_ZERO);
    EXPECT_TRUE(actualTransmitterCell2._energy > origTransmitterCell2._energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}


TEST_F(AttackerTests, successDistributeToTransmitterAndConstructor)
{
    auto otherGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({12.0f, 10.0f}).cellType(DepotDescription()),
        CellDescription()
            .id(4)
            .pos({11.0f, 9.0f})
            .cellType(ConstructorDescription().genome(otherGenome)),
        CellDescription().id(5).pos({9.0f, 10.0f}),
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

    EXPECT_TRUE(actualAttackCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_TRUE(approxCompare(actualTransmitterCell._energy, origTransmitterCell._energy));
    EXPECT_TRUE(actualConstructorCell._energy > origConstructorCell._energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successDistributeToConnectedCells)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({12.0f, 10.0f}).cellType(OscillatorDescription()),
        CellDescription().id(4).pos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    auto origOscillatorCell1 = getCell(data, 2);
    auto actualOscillatorCell1 = getCell(actualData, 2);

    auto origOscillatorCell2 = getCell(data, 3);
    auto actualOscillatorCell2 = getCell(actualData, 3);

    EXPECT_TRUE(actualAttackCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualOscillatorCell1._energy > origOscillatorCell1._energy + NEAR_ZERO);
    EXPECT_TRUE(actualOscillatorCell2._energy > origOscillatorCell2._energy + NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(AttackerTests, successTwoTargets)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(AttackerDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}),
        CellDescription().id(4).pos({9.0f, 11.0f}),
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

    EXPECT_TRUE(actualAttackCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_TRUE(actualAttackCell._energy > origAttackCell._energy + NEAR_ZERO);
    EXPECT_TRUE(actualTargetCell1._energy < origTargetCell1._energy - NEAR_ZERO);
    EXPECT_TRUE(actualTargetCell2._energy < origTargetCell2._energy - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
