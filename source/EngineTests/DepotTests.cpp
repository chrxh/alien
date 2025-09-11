#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"
#include "EngineInterface/GenomeDescription.h"

class DepotTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.transmitterEnergyDistributionSameCreature.value = true;
        result.innerFriction.value = 0;
        result.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.radiationType1_strength.baseValue[i] = 0;
        }
        return result;
    }

    DepotTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~DepotTests() = default;
};

//TEST_F(DepotTests, distributeToOtherTransmitter)
//{
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_TransmittersAndConstructors))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellTypeData(GeneratorDescription()),
//        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(DepotDescription()),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origTransmitterCell1 = getCell(data, 1);
//    auto actualTransmitterCell1 = getCell(actualData, 1);
//
//    auto origTransmitterCell2 = getCell(data, 3);
//    auto actualTransmitterCell2 = getCell(actualData, 3);
//
//    auto origGeneratorCell = getCell(data, 2);
//    auto actualGeneratorCell = getCell(actualData, 2);
//
//    EXPECT_TRUE(approxCompare(0.0f, actualTransmitterCell1._signal->_channels[0]));
//    EXPECT_TRUE(actualTransmitterCell1._energy < origTransmitterCell1._energy - NEAR_ZERO);
//    EXPECT_TRUE(actualTransmitterCell2._energy > origTransmitterCell2._energy + NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(origGeneratorCell._energy, actualGeneratorCell._energy));
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
//
//TEST_F(DepotTests, distributeToOneOtherTransmitter_forwardSignal)
//{
//    SignalDescription signal;
//    signal.channels({0.5f, -0.7f, 0, 0, 0, 0, 0, 0});
//
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_TransmittersAndConstructors))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellTypeData(GeneratorDescription())
//            .signal(signal),
//        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(DepotDescription()),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origTransmitterCell1 = getCell(data, 1);
//    auto actualTransmitterCell1 = getCell(actualData, 1);
//
//    auto origTransmitterCell2 = getCell(data, 3);
//    auto actualTransmitterCell2 = getCell(actualData, 3);
//
//    auto origGeneratorCell = getCell(data, 2);
//    auto actualGeneratorCell = getCell(actualData, 2);
//
//    for (int i = 0; i < MAX_CHANNELS; ++i) {
//        EXPECT_TRUE(approxCompare(signal._channels[i], actualTransmitterCell1._signal->_channels[i]));
//    }
//    EXPECT_TRUE(actualTransmitterCell1._energy < origTransmitterCell1._energy - NEAR_ZERO);
//    EXPECT_TRUE(actualTransmitterCell2._energy > origTransmitterCell2._energy + NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(origGeneratorCell._energy, actualGeneratorCell._energy));
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
//
//TEST_F(DepotTests, distributeToConnectedCells)
//{
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_ConnectedCells))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellTypeData(GeneratorDescription()),
//        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(DepotDescription()),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origTransmitterCell1 = getCell(data, 1);
//    auto actualTransmitterCell1 = getCell(actualData, 1);
//
//    auto origTransmitterCell2 = getCell(data, 3);
//    auto actualTransmitterCell2 = getCell(actualData, 3);
//
//    auto origGeneratorCell = getCell(data, 2);
//    auto actualGeneratorCell = getCell(actualData, 2);
//
//    EXPECT_TRUE(actualTransmitterCell1._energy < origTransmitterCell1._energy - NEAR_ZERO);
//    EXPECT_TRUE(actualTransmitterCell2._energy > origTransmitterCell2._energy + NEAR_ZERO);
//    EXPECT_TRUE(actualGeneratorCell._energy > origGeneratorCell._energy + NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
//
//TEST_F(DepotTests, distributeToOtherTransmitterAndConstructor)
//{
//    //auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({CellGenomeDescription()}));
//
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_TransmittersAndConstructors))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription().id(2).pos({11.0f, 10.0f}).cellTypeData(ConstructorDescription()/*.genome(genome)*/),
//        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(DepotDescription()),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origTransmitterCell = getCell(data, 1);
//    auto actualTransmitterCell = getCell(actualData, 1);
//
//    auto origConstructorCell = getCell(data, 2);
//    auto actualConstructorCell = getCell(actualData, 2);
//
//    auto origOtherTransmitterCell = getCell(data, 3);
//    auto actualOtherTransmitterCell = getCell(actualData, 3);
//
//    EXPECT_TRUE(actualTransmitterCell._energy < origTransmitterCell._energy - NEAR_ZERO);
//    EXPECT_TRUE(actualConstructorCell._energy > origConstructorCell._energy + NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(actualOtherTransmitterCell._energy, origOtherTransmitterCell._energy));
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
//
//TEST_F(DepotTests, distributeOnlyToActiveConstructors)
//{
//    //auto genome = GenomeDescription().header(GenomeHeaderDescription().numBranches(1));
//    
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_TransmittersAndConstructors))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription()
//            .id(2)
//            .pos({11.0f, 10.0f})
//            .cellTypeData(ConstructorDescription()/*.genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(genome))*/),
//        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(DepotDescription()),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origTransmitterCell = getCell(data, 1);
//    auto actualTransmitterCell = getCell(actualData, 1);
//
//    auto origConstructorCell = getCell(data, 2);
//    auto actualConstructorCell = getCell(actualData, 2);
//
//    auto origOtherTransmitterCell = getCell(data, 3);
//    auto actualOtherTransmitterCell = getCell(actualData, 3);
//
//    EXPECT_TRUE(actualTransmitterCell._energy < origTransmitterCell._energy - NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(actualConstructorCell._energy, origConstructorCell._energy));
//    EXPECT_TRUE(actualOtherTransmitterCell._energy > origOtherTransmitterCell._energy + NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
//
//TEST_F(DepotTests, distributeToTwoTransmittersWithDifferentColor)
//{
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_TransmittersAndConstructors))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription().id(2).pos({11.0f, 10.0f}).cellTypeData(DepotDescription()).color(1),
//        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(DepotDescription()),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(1);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origTransmitterCell = getCell(data, 1);
//    auto actualTransmitterCell = getCell(actualData, 1);
//
//    auto origOtherTransmitterCell1 = getCell(data, 2);
//    auto actualOtherTransmitterCell1 = getCell(actualData, 2);
//
//    auto origOtherTransmitterCell2 = getCell(data, 3);
//    auto actualOtherTransmitterCell2 = getCell(actualData, 3);
//
//    EXPECT_TRUE(actualTransmitterCell._energy < origTransmitterCell._energy - NEAR_ZERO);
//    EXPECT_TRUE(actualOtherTransmitterCell1._energy > origOtherTransmitterCell2._energy + NEAR_ZERO);
//    EXPECT_TRUE(actualOtherTransmitterCell2._energy > origOtherTransmitterCell2._energy + NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
//
//TEST_F(DepotTests, distributeNotToNotReadyConstructors)
//{
//    _parameters.constructorCompletenessCheck.value = true;
//    _simulationFacade->setSimulationParameters(_parameters);
//
//    auto subgenome = GenomeDescription().cells({CellGenomeDescription()});
//
//    auto genome = GenomeDescription().cells({
//        CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()),
//        CellGenomeDescription().cellType(DepotGenomeDescription()),
//        CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(subgenome))),
//    });
//
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({9.0f, 10.0f})
//            .cellTypeData(ConstructorDescription().numExpectedCells(4)/*.genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(genome))*/),
//        CellDescription()
//            .id(2)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_TransmittersAndConstructors))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription()
//            .id(3)
//            .pos({11.0f, 10.0f})
//            .cellTypeData(ConstructorDescription()/*.genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(subgenome))*/),
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(2);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origReplicator = getCell(data, 1);
//    auto actualReplicator = getCell(actualData, 1);
//
//    auto origTransmitter = getCell(data, 2);
//    auto actualTransmitter = getCell(actualData, 2);
//
//    auto origConstructor = getCell(data, 3);
//    auto actualConstructor = getCell(actualData, 3);
//
//    EXPECT_TRUE(actualTransmitter._energy < origTransmitter._energy - NEAR_ZERO);
//    EXPECT_TRUE(approxCompare(actualReplicator._energy, origReplicator._energy));
//    EXPECT_TRUE(actualConstructor._energy > origConstructor._energy + NEAR_ZERO);
//
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
//
//TEST_F(DepotTests, distributeToReadyConstructors)
//{
//    _parameters.constructorCompletenessCheck.value = true;
//    _simulationFacade->setSimulationParameters(_parameters);
//
//    //auto subgenome = GenomeDescription().cells({CellGenomeDescription()});
//
//    //auto genome = GenomeDescription().cells({
//    //    CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()),
//    //    CellGenomeDescription().cellType(DepotGenomeDescription()),
//    //    CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(subgenome))),
//    //});
//
//    Description data;
//    data.cells() = {
//        CellDescription()
//            .id(1)
//            .pos({9.0f, 10.0f})
//            .cellTypeData(ConstructorDescription().numExpectedCells(4)/*.genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(genome))*/),
//        CellDescription()
//            .id(2)
//            .pos({10.0f, 10.0f})
//            .cellTypeData(DepotDescription().mode(EnergyDistributionMode_TransmittersAndConstructors))
//            .energy(_parameters.normalCellEnergy.value[0] * 2),
//        CellDescription()
//            .id(3)
//            .pos({11.0f, 10.0f})
//            .cellTypeData(ConstructorDescription()/*.genome(GenomeDescriptionConverterService::get().convertDescriptionToBytes(subgenome))*/),
//        CellDescription()
//            .id(4)
//            .pos({12.0f, 10.0f})
//            ,
//    });
//    data.addConnection(1, 2);
//    data.addConnection(2, 3);
//    data.addConnection(3, 4);
//
//    _simulationFacade->setSimulationData(data);
//    _simulationFacade->calcTimesteps(2);
//
//    auto actualData = _simulationFacade->getSimulationData();
//
//    auto origReplicator = getCell(data, 1);
//    auto actualReplicator = getCell(actualData, 1);
//
//    auto origTransmitter = getCell(data, 2);
//    auto actualTransmitter = getCell(actualData, 2);
//
//    auto origConstructor = getCell(data, 3);
//    auto actualConstructor = getCell(actualData, 3);
//
//    EXPECT_TRUE(actualTransmitter._energy < origTransmitter._energy - NEAR_ZERO);
//    EXPECT_TRUE(actualReplicator._energy > origReplicator._energy + NEAR_ZERO);
//    EXPECT_TRUE(actualConstructor._energy > origConstructor._energy + NEAR_ZERO);
//
//    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
//}
