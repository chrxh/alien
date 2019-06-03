#include "Base/ServiceLocator.h"
#include "ModelBasic/CellComputerCompiler.h"
#include "ModelBasic/QuantityConverter.h"

#include "IntegrationGpuTestFramework.h"

class PropulsionGpuTests
    : public IntegrationGpuTestFramework
{
public:
    PropulsionGpuTests() : IntegrationGpuTestFramework({ 10, 10 })
    {}

    virtual ~PropulsionGpuTests() = default;

protected:
    virtual void SetUp();

    DataDescription runPropulsion(Enums::PropIn::Type command, unsigned char angle, unsigned char power) const;

    pair<Physics::Velocities, Enums::PropOut::Type> extractResult(DataDescription const& data);
};


void PropulsionGpuTests::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

DataDescription PropulsionGpuTests::runPropulsion(Enums::PropIn::Type command,
    unsigned char angle, unsigned char power) const
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::PROPULSION);
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    tokenData[Enums::Prop::IN] = command;
    tokenData[Enums::Prop::IN_ANGLE] = angle;
    tokenData[Enums::Prop::IN_POWER] = power;
    firstCell.addToken(token);
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    checkEnergy(origData, newData);
    return newData;
}

pair<Physics::Velocities, Enums::PropOut::Type> PropulsionGpuTests::extractResult(DataDescription const & data)
{
    pair<Physics::Velocities, Enums::PropOut::Type> result;
    auto const& cluster = data.clusters->at(0);
    CellDescription cell;
    for (auto const& cellToCheck : *cluster.cells) {
        if (cellToCheck.tokens && 1 == cellToCheck.tokens->size()) {
            cell = cellToCheck;
        }
    }
    auto token = cell.tokens->at(0);
    result.first = { *cluster.vel, *cluster.angularVel };
    result.second = static_cast<Enums::PropOut::Type>(token.data->at(Enums::Prop::OUT));
    return result;
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle1)
{
    auto data = runPropulsion(Enums::PropIn::BY_ANGLE, 0, 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_GT(-0.005f, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < 0.001f);
    EXPECT_TRUE(abs(velocities.angular) < 0.01f);
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle2)
{
    auto data = runPropulsion(Enums::PropIn::BY_ANGLE, QuantityConverter::convertAngleToData(90), 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < 0.001f);
    EXPECT_GT(-0.005f, velocities.linear.y());
    EXPECT_GT(-0.005f, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle3)
{
    auto data = runPropulsion(Enums::PropIn::BY_ANGLE, QuantityConverter::convertAngleToData(180), 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_LT(0.005f, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < 0.001f);
    EXPECT_TRUE(abs(velocities.angular) < 0.01f);
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle4)
{
    auto data = runPropulsion(Enums::PropIn::BY_ANGLE, QuantityConverter::convertAngleToData(270), 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < 0.001f);
    EXPECT_LT(0.005f, velocities.linear.y());
    EXPECT_LT(0.005f, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlFromCenter)
{
    auto data = runPropulsion(Enums::PropIn::FROM_CENTER, 0, 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_LT(0.005f, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < 0.001f);
    EXPECT_TRUE(abs(velocities.angular) < 0.01f);
}

TEST_F(PropulsionGpuTests, testThrustControlTowardCenter)
{
    auto data = runPropulsion(Enums::PropIn::TOWARD_CENTER, 0, 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_GT(-0.005f, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < 0.001f);
    EXPECT_TRUE(abs(velocities.angular) < 0.01f);
}

TEST_F(PropulsionGpuTests, testThrustControlRotationClockwise)
{
    auto data = runPropulsion(Enums::PropIn::ROTATION_CLOCKWISE, 0, 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < 0.001f);
    EXPECT_LT(0.005f, velocities.linear.y());
    EXPECT_LT(0.005f, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlRotationCounterClockwise)
{
    auto data = runPropulsion(Enums::PropIn::ROTATION_COUNTERCLOCKWISE, 0, 100);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < 0.001f);
    EXPECT_GT(-0.005f, velocities.linear.y());
    EXPECT_GT(-0.005f, velocities.angular);
}
