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
        result.innerFriction = 0;
        result.baseValues.friction = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationCellAgeStrength[i] = 0;
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
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({0, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_noOtherCell)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_densityTooLow)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setMinDensity(0.3f)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(10).height(10).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_wrongColor)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setColor(1)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(10).height(10).cellDistance(2.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_foundAtFront)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    EXPECT_TRUE(actualSensorCell.signal->channels[1] > 0.3f);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] < 15.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtRightHandSide)
{
    _parameters.cellFunctionMuscleMovementTowardTargetedObject = false;
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.signal->channels[0]));
    EXPECT_TRUE(actualAttackCell.signal->channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell.signal->channels[3] > 70.0f / 365);
    EXPECT_TRUE(actualAttackCell.signal->channels[3] < 105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtLeftHandSide)
{
    _parameters.cellFunctionMuscleMovementTowardTargetedObject = false;
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.signal->channels[0]));
    EXPECT_TRUE(actualAttackCell.signal->channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell.signal->channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualAttackCell.signal->channels[3] > -105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtBack)
{
    _parameters.cellFunctionMuscleMovementTowardTargetedObject = false;
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({190.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.signal->channels[0]));
    EXPECT_TRUE(actualAttackCell.signal->channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell.signal->channels[3] < -165.0f / 365 || actualAttackCell.signal->channels[3] > 165.0f / 365);
}


TEST_F(SensorTests, scanNeighborhood_twoMasses)
{
    _parameters.cellFunctionMuscleMovementTowardTargetedObject = false;
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setMinDensity(0.7f)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(1.5f)));
    data.add(DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 200.0f}).width(16).height(16).cellDistance(1.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    EXPECT_TRUE(actualSensorCell.signal->channels[1] > 0.7f);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] > -105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    EXPECT_TRUE(actualSensorCell.signal->channels[1] > 0.3f);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensorCell.signal->channels[3] < 15.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_found_wallBehind)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    //data.add(DescriptionEditService::get().createRect(
    //    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).mutationId(0)));

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(7)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(7)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_notFound_wallInBetween)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(7)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(7)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({50.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).mutationId(0)));

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_sameMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(6)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_sameMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto const MutantId = 6;
    for (int otherMutantId = 0; otherMutantId < 100; ++otherMutantId) {
        if (otherMutantId == MutantId) {
            continue;
        }
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(MutantId)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(MutantId)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setCellFunction(NerveDescription())
                 .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::get().createRect(
            DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(otherMutantId)));

        _simulationFacade->clear();
        _simulationFacade->setCurrentTimestep(0ull);
        _simulationFacade->setSimulationData(data);
        _simulationFacade->calcTimesteps(1);

        auto actualData = _simulationFacade->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_zeroMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToHandcraftedCells)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_zeroMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToHandcraftedCells)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_respawnedMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_respawnedMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToFreeCells)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(1000.0f)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setCellFunction(NerveDescription())
                 .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
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

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_otherMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(1000.0f)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setCellFunction(NerveDescription())
                 .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
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

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_zeroMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setGenomeComplexity(1000.0f)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0).genomeComplexity(100.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_respawnedCell)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(1000.0f)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1).genomeComplexity(100.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(500.0f)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setCellFunction(NerveDescription())
                 .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
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

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_otherMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(500.0f)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setCellFunction(NerveDescription())
                 .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
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

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_zeroMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(100.0f)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0).genomeComplexity(1000.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_respawnedCell)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(100.0f)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1).genomeComplexity(1000.0f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_minRange_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setMinRange(50)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_minRange_notFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setMinRange(120)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_maxRange_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setMaxRange(120)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.signal->channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_maxRange_notFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setCellFunction(SensorDescription().setMaxRange(50)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setCellFunction(NerveDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(
        DescriptionEditService::get().createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.signal->channels[0]));
}
