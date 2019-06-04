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
    const float SmallVelocity = 0.005f;
    const float SmallAngularVelocity = 0.05f;
    const float NeglectableVelocity = 0.001f;
    const float NeglectableAngularVelocity = 0.01f;

protected:
    virtual void SetUp();

    DataDescription runStandardPropulsionTest(Enums::PropIn::Type command, unsigned char propAngle,
        unsigned char propPower, QVector2D const& vel = QVector2D{}, float angle = 0,
        float angularVel = 0.0f) const;

    ClusterDescription createClusterForPropulsionTest(Enums::PropIn::Type command,
        unsigned char propAngle, unsigned char propPower, QVector2D const& vel, 
        float angle, float angularVel, int numTokens) const;

    pair<Physics::Velocities, Enums::PropOut::Type> extractResult(DataDescription const& data);
};


void PropulsionGpuTests::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

ClusterDescription PropulsionGpuTests::createClusterForPropulsionTest(Enums::PropIn::Type command, 
    unsigned char propAngle, unsigned char propPower, QVector2D const& vel, float angle, 
    float angularVel, int numTokens) const
{
    auto result = createLineCluster(2, QVector2D{}, vel, angle, angularVel);
    auto& firstCell = result.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto& secondCell = result.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::PROPULSION);
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    tokenData[Enums::Prop::IN] = command;
    tokenData[Enums::Prop::IN_ANGLE] = propAngle;
    tokenData[Enums::Prop::IN_POWER] = propPower;
    for (int i = 0; i < numTokens; ++i) {
        firstCell.addToken(token);
    }
    return result;
}

DataDescription PropulsionGpuTests::runStandardPropulsionTest(Enums::PropIn::Type command,
    unsigned char propAngle, unsigned char propPower, QVector2D const& vel,
    float angle, float angularVel) const
{
    DataDescription origData;
    origData.addCluster(
        createClusterForPropulsionTest(command, propAngle, propPower, vel, angle, angularVel, 1));

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

TEST_F(PropulsionGpuTests, testDoNothing)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::DO_NOTHING, 0, 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(isCompatible(QVector2D{}, velocities.linear));
    EXPECT_TRUE(isCompatible(0.0, velocities.angular));
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle1)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::BY_ANGLE, 0, 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_LT(SmallVelocity, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < NeglectableVelocity);
    EXPECT_TRUE(abs(velocities.angular) < NeglectableAngularVelocity);
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle2)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::BY_ANGLE, QuantityConverter::convertAngleToData(90), 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < NeglectableVelocity);
    EXPECT_LT(SmallVelocity, velocities.linear.y());
    EXPECT_GT(-SmallAngularVelocity, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle3)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::BY_ANGLE, QuantityConverter::convertAngleToData(180), 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_GT(-SmallVelocity, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < NeglectableVelocity);
    EXPECT_TRUE(abs(velocities.angular) < NeglectableAngularVelocity);
}

TEST_F(PropulsionGpuTests, testThrustControlByAngle4)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::BY_ANGLE, QuantityConverter::convertAngleToData(270), 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < NeglectableVelocity);
    EXPECT_GT(-SmallVelocity, velocities.linear.y());
    EXPECT_LT(SmallAngularVelocity, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlFromCenter)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::FROM_CENTER, 0, 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_GT(-SmallVelocity, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < NeglectableVelocity);
    EXPECT_TRUE(abs(velocities.angular) < NeglectableAngularVelocity);
}

TEST_F(PropulsionGpuTests, testThrustControlTowardCenter)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::TOWARD_CENTER, 0, 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_LT(SmallVelocity, velocities.linear.x());
    EXPECT_TRUE(abs(velocities.linear.y()) < NeglectableVelocity);
    EXPECT_TRUE(abs(velocities.angular) < NeglectableAngularVelocity);
}

TEST_F(PropulsionGpuTests, testThrustControlRotationClockwise)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::ROTATION_CLOCKWISE, 0, 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < NeglectableVelocity);
    EXPECT_GT(-SmallVelocity, velocities.linear.y());
    EXPECT_LT(SmallAngularVelocity, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlRotationCounterClockwise)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::ROTATION_COUNTERCLOCKWISE, 0, 100, QVector2D{}, 180);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_TRUE(abs(velocities.linear.x()) < NeglectableVelocity);
    EXPECT_LT(SmallVelocity, velocities.linear.y());
    EXPECT_GT(-SmallAngularVelocity, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlDampRotation1)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::DAMP_ROTATION, 0, 100, QVector2D{}, 180, 10);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_GT(10 - SmallAngularVelocity, velocities.angular);
}

TEST_F(PropulsionGpuTests, testThrustControlDampRotation2)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::DAMP_ROTATION, 0, 100, QVector2D{}, 180, -10);
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_LT(-10 + SmallAngularVelocity, velocities.angular);
}

TEST_F(PropulsionGpuTests, testPowerControl)
{
    DataDescription origData;
    auto&& cluster1 = createClusterForPropulsionTest(Enums::PropIn::FROM_CENTER, 0, 1, QVector2D{}, 0, 0, 1);
    auto&& cluster2 = createClusterForPropulsionTest(Enums::PropIn::FROM_CENTER, 0, 10, QVector2D{}, 0, 0, 1);
    auto&& cluster3 = createClusterForPropulsionTest(Enums::PropIn::FROM_CENTER, 0, 255, QVector2D{}, 0, 0, 1);
    setCenterPos(cluster1, { 0, 0 });
    setCenterPos(cluster2, { 5, 0 });
    setCenterPos(cluster3, { 0, 5 });
    origData.addCluster(cluster1);
    origData.addCluster(cluster2);
    origData.addCluster(cluster3);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    checkEnergy(origData, newData);

    auto clusterByClusterId = IntegrationTestHelper::getClusterByClusterId(newData);
    auto const& newCluster1 = clusterByClusterId.at(cluster1.id);
    auto const& newCluster2 = clusterByClusterId.at(cluster2.id);
    auto const& newCluster3 = clusterByClusterId.at(cluster2.id);

    EXPECT_GT(newCluster1.vel->length() * 2, newCluster2.vel->length() );
    EXPECT_GT(newCluster2.vel->length() * 2, newCluster3.vel->length());
}

TEST_F(PropulsionGpuTests, testSlowdown)
{
    auto data = runStandardPropulsionTest(Enums::PropIn::TOWARD_CENTER, 0, 100, QVector2D{0.3f, 0});
    auto result = extractResult(data);
    auto const& velocities = result.first;
    auto const& propOut = result.second;

    EXPECT_EQ(Enums::PropOut::SUCCESS, propOut);
    EXPECT_GT(0.3f - SmallVelocity, velocities.linear.length());
}

/**
* Situation: - two clusters with identical cells
*			 - first cluster has "cellMaxToken" tokens on one cell
*            - second cluster has one token
*			 - simulating one time steps executing same propulsion function
* Expected result: first cluster has "cellMaxToken" times higher velocity than second one
*/
TEST_F(PropulsionGpuTests, testParallelization1)
{
    auto const& cellMaxToken = _parameters.cellMaxToken;

    DataDescription origData;
    auto&& cluster1 = createClusterForPropulsionTest(Enums::PropIn::FROM_CENTER, 0, 1, QVector2D{}, 0, 0, cellMaxToken);
    auto&& cluster2 = createClusterForPropulsionTest(Enums::PropIn::FROM_CENTER, 0, 1, QVector2D{}, 0, 0, 1);
    setCenterPos(cluster1, { 0, 0 });
    setCenterPos(cluster2, { 5, 0 });
    origData.addCluster(cluster1);
    origData.addCluster(cluster2);

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    checkEnergy(origData, newData);

    auto clusterByClusterId = IntegrationTestHelper::getClusterByClusterId(newData);
    auto const& newCluster1 = clusterByClusterId.at(cluster1.id);
    auto const& newCluster2 = clusterByClusterId.at(cluster2.id);

    EXPECT_TRUE(isCompatible(static_cast<float>(cellMaxToken), newCluster1.vel->length() / newCluster2.vel->length()));
}

/**
* Situation: - two rectangular clusters with identical cells
*			 - first cluster has two tokens on different cells
*            - second cluster has one token
*			 - simulating one time steps executing same propulsion function
* Expected result: first cluster has two times higher velocity than second one
*/
TEST_F(PropulsionGpuTests, testParallelization2)
{
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    tokenData[Enums::Prop::IN] = Enums::PropIn::ROTATION_CLOCKWISE;
    tokenData[Enums::Prop::IN_ANGLE] = 0;
    tokenData[Enums::Prop::IN_POWER] = 100;

    DataDescription origData;
    ClusterDescription cluster1;
    ClusterDescription cluster2;
    {
        cluster1 = createRectangularCluster({ 2, 2 }, QVector2D{}, QVector2D{});
        auto& firstCell = cluster1.cells->at(0);
        auto& secondCell = cluster1.cells->at(1);
        auto& thirdCell = cluster1.cells->at(3);
        auto& fourthCell = cluster1.cells->at(2);
        firstCell.tokenBranchNumber = 0;
        secondCell.tokenBranchNumber = 1;
        secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::PROPULSION);
        thirdCell.tokenBranchNumber = 2;
        fourthCell.tokenBranchNumber = 3;
        fourthCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::PROPULSION);
        firstCell.addToken(token);
        thirdCell.addToken(token);
        origData.addCluster(cluster1);
    }
    {
        cluster2 = createRectangularCluster({ 2, 2 }, QVector2D{5.0f, 0}, QVector2D{});
        auto& firstCell = cluster2.cells->at(0);
        auto& secondCell = cluster2.cells->at(1);
        auto& thirdCell = cluster2.cells->at(3);
        auto& fourthCell = cluster2.cells->at(2);
        firstCell.tokenBranchNumber = 0;
        secondCell.tokenBranchNumber = 1;
        secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::PROPULSION);
        thirdCell.tokenBranchNumber = 2;
        fourthCell.tokenBranchNumber = 3;
        fourthCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::PROPULSION);
        firstCell.addToken(token);
        origData.addCluster(cluster2);
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    checkEnergy(origData, newData);

    auto clusterByClusterId = IntegrationTestHelper::getClusterByClusterId(newData);
    auto const& newCluster1 = clusterByClusterId.at(cluster1.id);
    auto const& newCluster2 = clusterByClusterId.at(cluster2.id);

    EXPECT_TRUE(isCompatible(2.0, *newCluster1.angularVel / *newCluster2.angularVel));
}
