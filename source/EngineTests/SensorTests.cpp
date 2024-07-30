#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
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

TEST_F(SensorTests, scanNeighborhood_noActivity)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({0, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_noOtherCell)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_densityTooLow)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setMinDensity(0.3f)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(10).height(10).cellDistance(2.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_wrongColor)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setColor(1)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(10).height(10).cellDistance(2.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_foundAtFront)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
    EXPECT_TRUE(actualSensorCell.activity.channels[1] > 0.3f);
    EXPECT_TRUE(actualSensorCell.activity.channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell.activity.channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell.activity.channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensorCell.activity.channels[3] < 15.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtRightHandSide)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(0.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualAttackCell.activity.channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[3] > 70.0f / 365);
    EXPECT_TRUE(actualAttackCell.activity.channels[3] < 105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtLeftHandSide)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(0.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualAttackCell.activity.channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[3] < -70.0f / 365);
    EXPECT_TRUE(actualAttackCell.activity.channels[3] > -105.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_foundAtBack)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription()),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({190.0f, 100.0f}).width(16).height(16).cellDistance(0.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualAttackCell.activity.channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[3] < -165.0f / 365 || actualAttackCell.activity.channels[3] > 165.0f / 365);
}


TEST_F(SensorTests, scanNeighborhood_twoMasses)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setMinDensity(0.3f)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 10.0f}).width(16).height(16).cellDistance(0.8f)));
    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 200.0f}).width(16).height(16).cellDistance(0.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualAttackCell.activity.channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[3] > 70.0f / 365);
    EXPECT_TRUE(actualAttackCell.activity.channels[3] < 105.0f / 365);
}

TEST_F(SensorTests, scanByAngle_found)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setFixedAngle(-90.0f)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(0.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualAttackCell.activity.channels[0]));
    EXPECT_TRUE(actualAttackCell.activity.channels[1] > 0.3f);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] > 80.0f / 256);
    EXPECT_TRUE(actualAttackCell.activity.channels[2] < 105.0f / 256);
}

TEST_F(SensorTests, scanByAngle_wrongAngle)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setFixedAngle(90.0f)),
         CellDescription()
             .setId(2)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters().center({100.0f, 190.0f}).width(16).height(16).cellDistance(0.5f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualAttackCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualAttackCell.activity.channels[0]));
}


TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
    EXPECT_TRUE(actualSensorCell.activity.channels[1] > 0.3f);
    EXPECT_TRUE(actualSensorCell.activity.channels[2] < 1.0f - 80.0f / 256);
    EXPECT_TRUE(actualSensorCell.activity.channels[2] > 1.0f - 105.0f / 256);
    EXPECT_TRUE(actualSensorCell.activity.channels[3] > -15.0f / 365);
    EXPECT_TRUE(actualSensorCell.activity.channels[3] < 15.0f / 365);
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_found_wallBehind)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    //data.add(DescriptionEditService::createRect(
    //    DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).mutationId(0)));

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(7)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(7)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_otherMutant_notFound_wallInBetween)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(7)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToOtherMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(7)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({50.0f, 100.0f}).width(1).height(16).cellDistance(0.5f).mutationId(0)));

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(7)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_sameMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(6)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_sameMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);

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
                 .setExecutionOrderNumber(0)
                 .setInputExecutionOrderNumber(5)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToSameMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(MutantId)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setExecutionOrderNumber(5)
                 .setCellFunction(NerveDescription())
                 .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::createRect(
            DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(otherMutantId)));

        _simController->clear();
        _simController->setCurrentTimestep(0ull);
        _simController->setSimulationData(data);
        _simController->calcTimesteps(1);

        auto actualData = _simController->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_zeroMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToZeroMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_zeroMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToZeroMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_respawnedMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToNutrient)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_respawnedMutant_notFound)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(6)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToNutrient)),
         CellDescription()
             .setId(2)
             .setMutationId(6)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(1000.0f)
                 .setExecutionOrderNumber(0)
                 .setInputExecutionOrderNumber(5)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setExecutionOrderNumber(5)
                 .setCellFunction(NerveDescription())
                 .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simController->clear();
        _simController->setCurrentTimestep(0ull);
        _simController->setSimulationData(data);
        _simController->calcTimesteps(1);

        auto actualData = _simController->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_otherMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(1000.0f)
                 .setExecutionOrderNumber(0)
                 .setInputExecutionOrderNumber(5)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setExecutionOrderNumber(5)
                 .setCellFunction(NerveDescription())
                 .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simController->clear();
        _simController->setCurrentTimestep(0ull);
        _simController->setSimulationData(data);
        _simController->calcTimesteps(1);

        auto actualData = _simController->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_zeroMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setGenomeComplexity(1000.0f)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0).genomeComplexity(100.0f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_lessComplexMutant_notFound_respawnedCell)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(1000.0f)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToLessComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1).genomeComplexity(100.0f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_found)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 1000; otherGenomeComplexity < 2001; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(500.0f)
                 .setExecutionOrderNumber(0)
                 .setInputExecutionOrderNumber(5)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setExecutionOrderNumber(5)
                 .setCellFunction(NerveDescription())
                 .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simController->clear();
        _simController->setCurrentTimestep(0ull);
        _simController->setSimulationData(data);
        _simController->calcTimesteps(1);

        auto actualData = _simController->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(1.0f, actualSensorCell.activity.channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_otherMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);

    for (int otherGenomeComplexity = 0; otherGenomeComplexity < 500; ++otherGenomeComplexity) {
        DataDescription data;
        data.addCells(
            {CellDescription()
                 .setId(1)
                 .setMutationId(5)
                 .setPos({100.0f, 100.0f})
                 .setMaxConnections(2)
                 .setGenomeComplexity(500.0f)
                 .setExecutionOrderNumber(0)
                 .setInputExecutionOrderNumber(5)
                 .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
             CellDescription()
                 .setId(2)
                 .setMutationId(5)
                 .setPos({101.0f, 100.0f})
                 .setMaxConnections(1)
                 .setExecutionOrderNumber(5)
                 .setCellFunction(NerveDescription())
                 .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
        data.addConnection(1, 2);

        data.add(DescriptionEditService::createRect(DescriptionEditService::CreateRectParameters()
                                                        .center({10.0f, 100.0f})
                                                        .width(16)
                                                        .height(16)
                                                        .cellDistance(0.5f)
                                                        .mutationId(6)
                                                        .genomeComplexity(toFloat(otherGenomeComplexity))));

        _simController->clear();
        _simController->setCurrentTimestep(0ull);
        _simController->setSimulationData(data);
        _simController->calcTimesteps(1);

        auto actualData = _simController->getSimulationData();
        auto actualSensorCell = getCell(actualData, 1);

        EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
    }
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_zeroMutant)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(100.0f)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(0).genomeComplexity(1000.0f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}

TEST_F(SensorTests, scanNeighborhood_targetedCreature_moreComplexMutant_notFound_respawnedCell)
{
    _parameters.cellFunctionAttackerSensorDetectionFactor[0] = 1.0f;
    _simController->setSimulationParameters(_parameters);
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setMutationId(100)
             .setGenomeComplexity(100.0f)
             .setPos({100.0f, 100.0f})
             .setMaxConnections(2)
             .setExecutionOrderNumber(0)
             .setInputExecutionOrderNumber(5)
             .setCellFunction(SensorDescription().setRestrictToMutants(SensorRestrictToMutants_RestrictToMoreComplexMutants)),
         CellDescription()
             .setId(2)
             .setMutationId(100)
             .setPos({101.0f, 100.0f})
             .setMaxConnections(1)
             .setExecutionOrderNumber(5)
             .setCellFunction(NerveDescription())
             .setActivity({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    data.add(DescriptionEditService::createRect(
        DescriptionEditService::CreateRectParameters().center({10.0f, 100.0f}).width(16).height(16).cellDistance(0.5f).mutationId(1).genomeComplexity(1000.0f)));

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);

    auto actualData = _simController->getSimulationData();
    auto actualSensorCell = getCell(actualData, 1);

    EXPECT_TRUE(approxCompare(0.0f, actualSensorCell.activity.channels[0]));
}
