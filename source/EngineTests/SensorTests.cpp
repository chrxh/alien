#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class SensorTests : public IntegrationTestFramework
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction.value = 0;
        result.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.radiationType1_strength.baseValue[i] = 0;
        }
        return result;
    }
    SensorTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~SensorTests() = default;
};

TEST_F(SensorTests, scanNeighborhood_noSignal)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription()),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({0, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData._cells.size());
    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_noOtherCell)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription()),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_densityTooLow)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().minDensity(0.3f)),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(10).height(10).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_wrongColor)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToColor(1)),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(10).height(10).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_foundAtFront)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription()),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
    EXPECT_TRUE(actualSensorCell._signal->_channels[1] > 0.3f);
    EXPECT_TRUE(actualSensorCell._signal->_channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensorCell._signal->_channels[3] < 15.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtRightHandSide)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription()),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell._signal->_channels[0]));
    EXPECT_TRUE(actualAttackCell._signal->_channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell._signal->_channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell._signal->_channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell._signal->_channels[3] > 70.0f / 365);
    EXPECT_TRUE(actualAttackCell._signal->_channels[3] < 105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtLeftHandSide)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription()),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell._signal->_channels[0]));
    EXPECT_TRUE(actualAttackCell._signal->_channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell._signal->_channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell._signal->_channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell._signal->_channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualAttackCell._signal->_channels[3] > -105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtBack)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription()),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({190.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell._signal->_channels[0]));
    EXPECT_TRUE(actualAttackCell._signal->_channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell._signal->_channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell._signal->_channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell._signal->_channels[3] < -165.0f / 365 || actualAttackCell._signal->_channels[3] > 165.0f / 365);
}


TEST_F(SensorTests, scanNeighborhood_twoMasses)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().minDensity(0.7f)),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.5f)));
    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 200.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
    EXPECT_TRUE(actualSensorCell._signal->_channels[1] > 0.7f);
    EXPECT_TRUE(actualSensorCell._signal->_channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualSensorCell._signal->_channels[3] > -105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_found)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(6)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .id(2)
             .mutationId(6)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
    EXPECT_TRUE(actualSensorCell._signal->_channels[1] > 0.3f);
    EXPECT_TRUE(actualSensorCell._signal->_channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell._signal->_channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensorCell._signal->_channels[3] < 15.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_found_wallBehind)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(6)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .id(2)
             .mutationId(6)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    //data.add(DescriptionEditService::get().createRect(
    //    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).mutationId(0)));

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_notFound)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(7)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .id(2)
             .mutationId(7)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_notFound_wallInBetween)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(7)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .id(2)
             .mutationId(7)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({50.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).mutationId(0)));

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_sameMutant_found)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(6)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
         CellDescription()
             .id(2)
             .mutationId(6)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(6)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_sameMutant_notFound)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto const MutantId = 6;
    for (int otherMutantId = 0; otherMutantId < 100; ++otherMutantId) {
        if (otherMutantId == MutantId) {
            continue;
        }
        CollectionDescription data;
        data.addCells(
            {CellDescription()
                 .id(1)
                 .mutationId(MutantId)
                 .pos({100.0f, 100.0f})
                 .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
             CellDescription()
                 .id(2)
                 .mutationId(MutantId)
                 .pos({101.0f, 100.0f})
                 .cellTypeData(OscillatorDescription())
                 .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(
            DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(otherMutantId)));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_zeroMutant_found)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(6)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToStructures)),
         CellDescription()
             .id(2)
             .mutationId(6)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_zeroMutant_notFound)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(6)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToStructures)),
         CellDescription()
             .id(2)
             .mutationId(6)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_respawnedMutant_found)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(6)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
         CellDescription()
             .id(2)
             .mutationId(6)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_respawnedMutant_notFound)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(6)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
         CellDescription()
             .id(2)
             .mutationId(6)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_found)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCells(
            {CellDescription()
                 .id(1)
                 .mutationId(5)
                 .pos({100.0f, 100.0f})
                 .genomeComplexity(1000.0f)
                 .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
             CellDescription()
                 .id(2)
                 .mutationId(5)
                 .pos({101.0f, 100.0f})
                 .cellTypeData(OscillatorDescription())
                 .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_otherMutant)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCells(
            {CellDescription()
                 .id(1)
                 .mutationId(5)
                 .pos({100.0f, 100.0f})
                 .genomeComplexity(1000.0f)
                 .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
             CellDescription()
                 .id(2)
                 .mutationId(5)
                 .pos({101.0f, 100.0f})
                 .cellTypeData(OscillatorDescription())
                 .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_zeroMutant)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(100)
             .pos({100.0f, 100.0f})
             .genomeComplexity(1000.0f)
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription()
             .id(2)
             .mutationId(100)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0).genomeComplexity(100.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_respawnedCell)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(100)
             .genomeComplexity(1000.0f)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription()
             .id(2)
             .mutationId(100)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1).genomeComplexity(100.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_found)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCells(
            {CellDescription()
                 .id(1)
                 .mutationId(5)
                 .pos({100.0f, 100.0f})
                 .genomeComplexity(500.0f)
                 .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
             CellDescription()
                 .id(2)
                 .mutationId(5)
                 .pos({101.0f, 100.0f})
                 .cellTypeData(OscillatorDescription())
                 .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_otherMutant)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        CollectionDescription data;
        data.addCells(
            {CellDescription()
                 .id(1)
                 .mutationId(5)
                 .pos({100.0f, 100.0f})
                 .genomeComplexity(500.0f)
                 .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
             CellDescription()
                 .id(2)
                 .mutationId(5)
                 .pos({101.0f, 100.0f})
                 .cellTypeData(OscillatorDescription())
                 .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_zeroMutant)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(100)
             .genomeComplexity(100.0f)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
         CellDescription()
             .id(2)
             .mutationId(100)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0).genomeComplexity(1000.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_respawnedCell)
{
    _parameters.attackerSensorDetectionFactor.value[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .mutationId(100)
             .genomeComplexity(100.0f)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().restrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
         CellDescription()
             .id(2)
             .mutationId(100)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1).genomeComplexity(1000.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_minRange_found)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().minRange(50)),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_minRange_notFound)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().minRange(120)),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_maxRange_found)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().maxRange(120)),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell._signal->_channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_maxRange_notFound)
{
    CollectionDescription data;
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({100.0f, 100.0f})
             .cellTypeData(SensorDescription().maxRange(50)),
         CellDescription()
             .id(2)
             .pos({101.0f, 100.0f})
             .cellTypeData(OscillatorDescription())
             .signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell._signal->_channels[0]));
}
