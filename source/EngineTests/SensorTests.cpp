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
    };
    std::string runSensor(DataDescription& world, SensorParameters const& parameters) const;

    void addMass(DataDescription& world, int width, int height, RealVector2D const& center);

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
    token.data[Enums::Sensor_Input] = Enums::SensorIn_SearchVicinity;
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

void SensorTests::addMass(DataDescription& world, int width, int height, RealVector2D const& center)
{
    world.add(DescriptionHelper::createRect(DescriptionHelper::CreateRectParameters().width(width).height(height).center(center)));
}

void SensorTests::SetUp()
{
    auto parameters = _simController->getSimulationParameters();
    parameters.radiationProb = 0;  //exclude radiation
    _simController->setSimulationParameters_async(parameters);
}

TEST_F(SensorTests, nothingFound)
{
    DataDescription world;
    auto result = runSensor(world, SensorParameters().center({0, 0}));

    EXPECT_EQ(Enums::SensorOut_NothingFound, result[Enums::Sensor_Output]);
}

TEST_F(SensorTests, massAtFront)
{
    DataDescription world;
    addMass(world, 10, 10, {200, 100});
    auto result = runSensor(world, SensorParameters().center({100,100}));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 10, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 10, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_TRUE(angle < 10 && angle > -10);
}

TEST_F(SensorTests, massAtBottom)
{
    DataDescription world;
    addMass(world, 10, 10, {0, 100});
    auto result = runSensor(world, SensorParameters().center({0, 0}));

    EXPECT_EQ(Enums::SensorOut_ClusterFound, result[Enums::Sensor_Output]);
    EXPECT_LE(100 - 10, result[Enums::Sensor_OutDistance]);
    EXPECT_GE(100 + 10, result[Enums::Sensor_OutDistance]);
    auto angle = static_cast<char>(result[Enums::Sensor_InOutAngle]);
    EXPECT_TRUE(angle < 64 + 10 && angle > 64 - 10);
}
