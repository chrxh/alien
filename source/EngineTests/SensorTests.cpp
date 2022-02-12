#include <chrono>

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class SensorTests : public IntegrationTestFramework
{
public:
    SensorTests()
        : IntegrationTestFramework({1000, 1000})
    {}

    ~SensorTests() = default;

    struct SensorParameters
    {
        MEMBER_DECLARATION(SensorParameters, RealVector2D, center, RealVector2D());
        MEMBER_DECLARATION(SensorParameters, unsigned char, command, Enums::SensorIn_DoNothing);
        MEMBER_DECLARATION(SensorParameters, int, angle, 0);
        MEMBER_DECLARATION(SensorParameters, int, minDensity, 0);
        MEMBER_DECLARATION(SensorParameters, int, color, 0);
    };
    std::string runSensor(DataDescription& world, SensorParameters const& parameters) const;

    void addMass(DataDescription& world, int width, int height, RealVector2D const& center, int color = 0);

protected:
    void SetUp() override;
};


std::string SensorTests::runSensor(DataDescription& world, SensorParameters const& parameters) const
{
    DataDescription sensorData = DescriptionHelper::createRect(
        DescriptionHelper::CreateRectParameters().width(2).height(1).center(parameters._center));
    auto& origFirstCell = sensorData.cells.at(0);
    origFirstCell.tokenBranchNumber = 0;
    auto& origSecondCell = sensorData.cells.at(1);
    origSecondCell.tokenBranchNumber = 1;
    origSecondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction_Sensor);
    auto token = createSimpleToken();
    token.data[Enums::Sensor_Input] = parameters._command;
    token.data[Enums::Sensor_InOutAngle] = parameters._angle;
    token.data[Enums::Sensor_InMinDensity] = parameters._minDensity;
    token.data[Enums::Sensor_InColor] = parameters._color;
    origFirstCell.addToken(token);
    world.add(sensorData);

    _simController->setSimulationData(world);
    _simController->calcSingleTimestep();

    DataDescription data = _simController->getSimulationData({0, 0}, _simController->getWorldSize());
    auto cellById = getCellById(data);
    auto const& secondCell = cellById.at(origSecondCell.id);
    EXPECT_EQ(1, secondCell.tokens.size());
    return secondCell.tokens.at(0).data;
}

void SensorTests::addMass(DataDescription& world, int width, int height, RealVector2D const& center, int color)
{
    world.add(DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters().width(width).height(height).center(center).color(color)));
}

void SensorTests::SetUp()
{
    auto parameters = _simController->getSimulationParameters();
    //exclude radiation and mutations
    parameters.radiationProb = 0;
    parameters.spotValues.tokenMutationRate = 0;
    parameters.spotValues.cellMutationRate = 0;
    _simController->setSimulationParameters_async(parameters);
}

TEST_F(SensorTests, doNothing)
{
    DataDescription world;
    addMass(world, 10, 10, {100, 0});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_DoNothing));

    EXPECT_EQ(Enums::SensorOut_NothingFound, result[Enums::Sensor_Output]);
}

TEST_F(SensorTests, searchVicinity_nothingFound)
{
    DataDescription world;
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchVicinity));

    EXPECT_EQ(Enums::SensorOut_NothingFound, result[Enums::Sensor_Output]);
}

TEST_F(SensorTests, searchVicinity_massAtFront)
{
    DataDescription world;
    addMass(world, 10, 10, {100, 0});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchVicinity));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_TRUE(angle < 10 && angle > -10);
}

TEST_F(SensorTests, searchVicinity_massAtFront_wrongColor)
{
    DataDescription world;
    addMass(world, 10, 10, {100, 0}, 1);
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchVicinity));

    EXPECT_EQ(Enums::SensorOut_NothingFound, result[Enums::Sensor_Output]);
}

TEST_F(SensorTests, searchVicinity_massAtBottom)
{
    DataDescription world;
    addMass(world, 10, 10, {0, 100});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchVicinity));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_TRUE(angle < 64 + 10 && angle > 64 - 10);
}

TEST_F(SensorTests, searchVicinity_twoMasses1)
{
    DataDescription world;
    addMass(world, 3, 3, {50, 0});
    addMass(world, 10, 10, {0, 100});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchVicinity).minDensity(10));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_TRUE(angle < 64 + 10 && angle > 64 - 10);
}

TEST_F(SensorTests, searchVicinity_twoMasses2)
{
    DataDescription world;
    addMass(world, 10, 10, {50, 0});
    addMass(world, 3, 3, {0, 100});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchVicinity).minDensity(10));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(50 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(50 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_TRUE(angle < 10 && angle > -10);
}

TEST_F(SensorTests, searchByAngle_nothingFound)
{
    DataDescription world;
    addMass(world, 10, 10, {0, 100});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchByAngle).angle(0));

    EXPECT_EQ(Enums::SensorOut_NothingFound, result[Enums::Sensor_Output]);
}

TEST_F(SensorTests, searchByAngle_massAtFront)
{
    DataDescription world;
    addMass(world, 10, 10, {100, 0});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchByAngle).angle(0));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_EQ(0, angle);
}

TEST_F(SensorTests, searchByAngle_twoMassesAtFront)
{
    DataDescription world;
    addMass(world, 10, 10, {100, 0});
    addMass(world, 10, 10, {50, 0});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchByAngle).angle(0));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(50 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(50 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_EQ(0, angle);
}

TEST_F(SensorTests, searchByAngle_twoMassesAtFront_differentColor)
{
    DataDescription world;
    addMass(world, 10, 10, {100, 0});
    addMass(world, 10, 10, {50, 0}, 1);
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchByAngle).angle(0));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_EQ(0, angle);
}

TEST_F(SensorTests, searchByAngle_massAtBottom)
{
    DataDescription world;
    addMass(world, 10, 10, {0, 100});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchByAngle).angle(64));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_EQ(64, angle);
}

TEST_F(SensorTests, searchByAngle_massAtTop)
{
    DataDescription world;
    addMass(world, 10, 10, {0, -100});
    auto result = runSensor(world, SensorParameters().center({0, 0}).command(Enums::SensorIn_SearchByAngle).angle(-64));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 15, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 15, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_EQ(-64, angle);
}
