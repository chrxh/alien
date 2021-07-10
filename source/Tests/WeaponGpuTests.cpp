/*
#include "Base/ServiceLocator.h"
#include "EngineInterface/QuantityConverter.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SerializationHelper.h"

#include "IntegrationGpuTestFramework.h"

class WeaponGpuTests
    : public IntegrationGpuTestFramework
{
public:
    WeaponGpuTests(IntVector2D const& universeSize = {50, 50}, boost::optional<EngineGpuData> const& modelData = boost::none)
        : IntegrationGpuTestFramework(universeSize, modelData)
    {}

    virtual ~WeaponGpuTests() = default;

protected:
    virtual void SetUp();

    struct WeaponTestResult {
        Enums::WeaponOut::Type tokenOutput;
        boost::optional<float> energyDiffOfWeapon;
        boost::optional<float> energyDiffOfTarget1;
        boost::optional<float> energyDiffOfTarget2;
    };
    struct WeaponTestParameters {
        MEMBER_DECLARATION(WeaponTestParameters, int, minMass, 0);
        MEMBER_DECLARATION(WeaponTestParameters, int, maxMass, 0);
        struct Target {
            Target(QVector2D position, int mass = 1)
                : position(position)
                , mass(mass)
            {}
            QVector2D position;
            int mass;
        };
        MEMBER_DECLARATION(WeaponTestParameters, boost::optional<Target>, target1, boost::none);
        MEMBER_DECLARATION(WeaponTestParameters, boost::optional<Target>, target2, boost::none);
    };
    WeaponTestResult runWeaponTest(WeaponTestParameters const& parameters) const;

    //creates 2x2 clusters, maxBonds should be 4
    ClusterDescription createRectangularWeaponCluster(QVector2D const& pos, QVector2D const& vel);
};

void WeaponGpuTests::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _parameters.cellFunctionWeaponGeometryDeviationExponent = 0;
    _parameters.cellFunctionWeaponInhomogeneousColorFactor = 1.0f;
    _context->setSimulationParameters(_parameters);
}

auto WeaponGpuTests::runWeaponTest(WeaponTestParameters const& parameters) const -> WeaponTestResult
{
    auto origCluster = createLineCluster(2, addSmallDisplacement(QVector2D{}), QVector2D{}, 0.0f, 0.0f);
    auto& firstCell = origCluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    auto& secondCell = origCluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::WEAPON);
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    firstCell.addToken(token);

    DataDescription origData;
    origData.addCluster(origCluster);

    boost::optional<ClusterDescription> target1;
    boost::optional<ClusterDescription> target2;
    if (parameters._target1) {
        target1 = createRectangularCluster(
            {parameters._target1->mass, 1}, addSmallDisplacement(parameters._target1->position), QVector2D{});
        origData.addCluster(*target1);
    }
    if (parameters._target2) {
        target2 = createRectangularCluster(
            {parameters._target2->mass, 1}, addSmallDisplacement(parameters._target2->position), QVector2D{});
        origData.addCluster(*target2);
    }

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    check(origData, newData);

    auto const newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const newClusterByClusterId = IntegrationTestHelper::getClusterByClusterId(newData);
    auto const& newSecondCell = newCellByCellId.at(secondCell.id);
    auto const& newToken = newSecondCell.tokens->at(0);

    WeaponTestResult result;
    result.tokenOutput = static_cast<Enums::WeaponOut::Type>(newToken.data->at(Enums::Weapon::OUTPUT));
    if (parameters._target1) {
        auto const& newTarget1 = newClusterByClusterId.at(target1->id);
        result.energyDiffOfTarget1 = calcAndCheckEnergy(newTarget1) - calcAndCheckEnergy(*target1);
    }
    if (parameters._target2) {
        auto const& newTarget2 = newClusterByClusterId.at(target2->id);
        result.energyDiffOfTarget2 = calcAndCheckEnergy(newTarget2) - calcAndCheckEnergy(*target2);
    }
    result.energyDiffOfWeapon = *newSecondCell.energy - *secondCell.energy;

    return result;
}

ClusterDescription WeaponGpuTests::createRectangularWeaponCluster(QVector2D const & pos, QVector2D const & vel)
{
    auto result = createRectangularCluster({2, 2}, pos, vel);
    auto& cells = *result.cells;
    cells[0].tokenBranchNumber = 0;
    cells[1].tokenBranchNumber = 1;
    cells[3].tokenBranchNumber = 2;
    cells[2].tokenBranchNumber = 3;

    for (auto& cell : cells) {
        cell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::WEAPON);
    }

    cells[0].addToken(createSimpleToken());

    return result;
}

TEST_F(WeaponGpuTests, testNoTarget)
{
    auto const result = runWeaponTest(WeaponTestParameters().target1(WeaponTestParameters::Target{ QVector2D{ 3, 0 } }));
    EXPECT_EQ(Enums::WeaponOut::NO_TARGET, result.tokenOutput);
    EXPECT_EQ(-_parameters.cellFunctionWeaponEnergyCost, result.energyDiffOfWeapon);
}

TEST_F(WeaponGpuTests, testNoTargetBelowMinMass)
{
    auto const result = runWeaponTest(
        WeaponTestParameters().minMass(2).maxMass(4).target1(WeaponTestParameters::Target{ QVector2D{ 1, 0 }, 1 }));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);
}

TEST_F(WeaponGpuTests, testNoTargetAboveMaxMass)
{
    auto const result = runWeaponTest(
        WeaponTestParameters().minMass(2).maxMass(4).target1(WeaponTestParameters::Target{ QVector2D{ 3, 0 }, 5 }));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);
}


TEST_F(WeaponGpuTests, testStrike)
{
    auto const result = runWeaponTest(WeaponTestParameters().target1(WeaponTestParameters::Target{ QVector2D{1, 0} }));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);

    auto const expectedEnergyLoss =
        _parameters.cellFunctionConstructorOffspringCellEnergy * _parameters.cellFunctionWeaponStrength + 1;
    EXPECT_EQ(-expectedEnergyLoss, *result.energyDiffOfTarget1);
}

TEST_F(WeaponGpuTests, testStrikeMinMass)
{
    auto const result = runWeaponTest(
        WeaponTestParameters().minMass(2).maxMass(4).target1(WeaponTestParameters::Target{QVector2D{1.5, 0}, 2}));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);
}

TEST_F(WeaponGpuTests, testStrikeMaxMass)
{
    auto const result = runWeaponTest(
        WeaponTestParameters().minMass(2).maxMass(4).target1(WeaponTestParameters::Target{ QVector2D{ 2.5, 0 }, 4 }));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);
}

TEST_F(WeaponGpuTests, testDoubleStrike)
{
    auto const result = runWeaponTest(WeaponTestParameters()
                                          .target1(WeaponTestParameters::Target{QVector2D{1, 0}})
                                          .target2(WeaponTestParameters::Target{QVector2D{0, 1}}));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);

    auto const expectedEnergyLoss =
        _parameters.cellFunctionConstructorOffspringCellEnergy * _parameters.cellFunctionWeaponStrength + 1;
    EXPECT_EQ(-expectedEnergyLoss, *result.energyDiffOfTarget1);
    EXPECT_EQ(-expectedEnergyLoss, *result.energyDiffOfTarget2);
}

/ **
* Situation: many 2x2 clusters with rotating tokens invoking weapons
* Expected result: energy balance fulfilled
* Fixed error: atomic operations for energy changes
* /
TEST_F(WeaponGpuTests, regressionTestManyClustersWithWeapons)
{
    _parameters.cellFusionVelocity = 100;    //exclude fusion
    _parameters.cellMaxTokenBranchNumber = 4;
    _context->setSimulationParameters(_parameters);

    DataDescription origData;
    for (int i = 0; i < 1000; ++i) {
        origData.addCluster(createRectangularWeaponCluster(
            QVector2D(_numberGen->getRandomReal(0, _universeSize.x), _numberGen->getRandomReal(0, _universeSize.y)), 
            QVector2D(_numberGen->getRandomReal(-0.3, 0.3), _numberGen->getRandomReal(-0.3, 0.3))));
    }

    IntegrationTestHelper::updateData(_access, _context, origData);
    IntegrationTestHelper::runSimulation(200, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    check(origData, newData);
}
*/
