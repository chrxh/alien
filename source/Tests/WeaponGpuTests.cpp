#include "Base/ServiceLocator.h"
#include "ModelBasic/QuantityConverter.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/SerializationHelper.h"

#include "IntegrationGpuTestFramework.h"

class WeaponGpuTests
    : public IntegrationGpuTestFramework
{
public:
    WeaponGpuTests(IntVector2D const& universeSize = {50, 50}, optional<ModelGpuData> const& modelData = boost::none)
        : IntegrationGpuTestFramework(universeSize, modelData)
    {}

    virtual ~WeaponGpuTests() = default;

protected:
    virtual void SetUp();

    struct WeaponTestResult {
        Enums::WeaponOut::Type tokenOutput;
        optional<float> energyDiffOfWeapon;
        optional<float> energyDiffOfTarget1;
        optional<float> energyDiffOfTarget2;
    };
    struct WeaponTestParameters {
        MEMBER_DECLARATION(WeaponTestParameters, optional<QVector2D>, target1, boost::none);
        MEMBER_DECLARATION(WeaponTestParameters, optional<QVector2D>, target2, boost::none);
    };
    WeaponTestResult runWeaponTest(WeaponTestParameters const& parameters) const;

    //creates 2x2 clusters, maxBonds should be 4
    ClusterDescription createRectangularWeaponCluster(QVector2D const& pos, QVector2D const& vel);
};

void WeaponGpuTests::SetUp()
{
    _parameters.radiationProb = 0;    //exclude radiation
    _context->setSimulationParameters(_parameters);
}

auto WeaponGpuTests::runWeaponTest(WeaponTestParameters const& parameters) const -> WeaponTestResult
{
    auto origCluster = createLineCluster(2, QVector2D{}, QVector2D{}, 0.0f, 0.0f);
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

    optional<CellDescription> targetCell1;
    optional<CellDescription> targetCell2;
    if (parameters._target1) {
        auto const targetCluster1 = createRectangularCluster({ 1, 1 }, *parameters._target1, QVector2D{});
        origData.addCluster(targetCluster1);
        targetCell1 = targetCluster1.cells->at(0);
    }
    if (parameters._target2) {
        auto const targetCluster2 = createRectangularCluster({ 1, 1 }, *parameters._target2, QVector2D{});
        origData.addCluster(targetCluster2);
        targetCell2 = targetCluster2.cells->at(0);
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    check(origData, newData);

    auto const newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const& newSecondCell = newCellByCellId.at(secondCell.id);
    auto const& newToken = newSecondCell.tokens->at(0);

    WeaponTestResult result;
    result.tokenOutput = static_cast<Enums::WeaponOut::Type>(newToken.data->at(Enums::Weapon::OUT));
    if (parameters._target1) {
        auto const& newTarget1 = newCellByCellId.at(targetCell1->id);
        result.energyDiffOfTarget1 = *newTarget1.energy - *targetCell1->energy;
    }
    if (parameters._target2) {
        auto const& newTarget2 = newCellByCellId.at(targetCell2->id);
        result.energyDiffOfTarget2 = *newTarget2.energy - *targetCell2->energy;
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
    auto const result = runWeaponTest(WeaponTestParameters().target1(QVector2D{ 3, 0 }));
    EXPECT_EQ(Enums::WeaponOut::NO_TARGET, result.tokenOutput);
    EXPECT_EQ(-_parameters.cellFunctionWeaponEnergyCost, result.energyDiffOfWeapon);
}

TEST_F(WeaponGpuTests, testStrike)
{
    auto const result = runWeaponTest(WeaponTestParameters().target1(QVector2D{1, 0}));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);

    auto const expectedEnergyLoss =
        _parameters.cellFunctionConstructorOffspringCellEnergy * _parameters.cellFunctionWeaponStrength + 1;
    EXPECT_EQ(-expectedEnergyLoss, *result.energyDiffOfTarget1);
}

TEST_F(WeaponGpuTests, testDoubleStrike)
{
    auto const result = runWeaponTest(WeaponTestParameters().target1(QVector2D{1, 0}).target2(QVector2D{0, 1}));
    EXPECT_EQ(Enums::WeaponOut::STRIKE_SUCCESSFUL, result.tokenOutput);

    auto const expectedEnergyLoss =
        _parameters.cellFunctionConstructorOffspringCellEnergy * _parameters.cellFunctionWeaponStrength + 1;
    EXPECT_EQ(-expectedEnergyLoss, *result.energyDiffOfTarget1);
    EXPECT_EQ(-expectedEnergyLoss, *result.energyDiffOfTarget2);
}

/**
* Situation: many 2x2 clusters with rotating tokens invoking weapons
* Expected result: energy balance fulfilled
* Fixed error: atomic operations for energy changes
*/
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

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(200, _controller);

    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });

    check(origData, newData);
}
