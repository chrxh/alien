#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class AttackerTests : public IntegrationTestFramework
{
public:
    AttackerTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
            _parameters.attackerEnergyCost.baseValue[i] = 0;
            _parameters.attackerStrength.value[i] = 0.5f;
            _parameters.attackerRadius.value[i] = 3.5f;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~AttackerTests() = default;

protected:
    // Helper to create an attacker creature with a generator that triggers it
    Description createAttackerWithGenerator(RealVector2D const& attackerPos, float attackerRawEnergy = 0.0f, int attackerColor = 0)
    {
        auto data = Description().addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos(attackerPos).color(attackerColor).cellType(AttackerDescription()).rawEnergy(attackerRawEnergy),
            CellDescription().id(2).pos({attackerPos.x + 1.0f, attackerPos.y}).color(attackerColor).cellType(GeneratorDescription().autoTriggerInterval(3)),
        }));
        data.addConnection(1, 2);
        return data;
    }

    // Helper to create a target creature at a given position
    Description createTargetCreature(RealVector2D const& pos, uint64_t creatureId = 2, int color = 0, float usableEnergy = 100.0f, bool fixed = false)
    {
        auto data = Description().addCreature(CreatureDescription().id(creatureId).cells({
            CellDescription().id(100).pos(pos).color(color).usableEnergy(usableEnergy).fixed(fixed),
            CellDescription().id(101).pos({pos.x + 1.0f, pos.y}).color(color).usableEnergy(usableEnergy).fixed(fixed),
        }));
        data.addConnection(100, 101);
        return data;
    }
};

/**
 * Test: attackerMaxRawEnergyThreshold
 * The attacker should not attack when its rawEnergy exceeds the threshold (2.0f)
 */
TEST_F(AttackerTests, maxRawEnergyThreshold_belowThreshold)
{
    // Create attacker with rawEnergy below threshold
    auto data = createAttackerWithGenerator({100.0f, 100.0f}, SimulationParameters::attackerMaxRawEnergyThreshold / 2);  // Below threshold

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);
    auto& origTarget = data.getCellRef(100);
    origTarget.rawEnergy(100.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getCellRef(1);
    auto actualTarget = actualData.getCellRef(100);

    // Attacker should attack because rawEnergy is below threshold
    EXPECT_TRUE(actualTarget._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(actualTarget._rawEnergy, origTarget._rawEnergy));

    // Attacker should have a signal with success value > 0
    ASSERT_TRUE(actualAttacker._signal.has_value());
    EXPECT_TRUE(actualAttacker._signal->_channels[Channels::AttackerSuccess] > NEAR_ZERO);
}

TEST_F(AttackerTests, maxRawEnergyThreshold_aboveThreshold)
{
    // Create attacker with rawEnergy above threshold
    auto data = createAttackerWithGenerator({100.0f, 100.0f}, SimulationParameters::attackerMaxRawEnergyThreshold + NEAR_ZERO);  // Above threshold

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getCellRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getCellRef(1);
    auto actualTarget = actualData.getCellRef(100);

    // Attacker should NOT attack because rawEnergy is above threshold
    EXPECT_TRUE(approxCompare(origTarget._usableEnergy, actualTarget._usableEnergy));
    EXPECT_TRUE(approxCompare(actualTarget._rawEnergy, origTarget._rawEnergy));

    // Attacker should have a signal with success value = 0
    ASSERT_TRUE(actualAttacker._signal.has_value());
    EXPECT_TRUE(approxCompare(0.0f, actualAttacker._signal->_channels[Channels::AttackerSuccess]));
}

TEST_F(AttackerTests, maxRawEnergyThreshold_outsideRange)
{
    // Create attacker with rawEnergy above threshold
    auto data = createAttackerWithGenerator({100.0f, 100.0f});

    // Add target creature outside attack radius
    data.add(createTargetCreature({100.0f, 100.0f + _parameters.attackerRadius.value[0] + 0.01f}), false);

    auto origTarget = data.getCellRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getCellRef(1);
    auto actualTarget = actualData.getCellRef(100);

    // Attacker should NOT attack because rawEnergy is above threshold
    EXPECT_TRUE(approxCompare(origTarget._usableEnergy, actualTarget._usableEnergy));
}

/**
 * Test: attackerFoodChainColorMatrix
 * The attack energy transfer should be modulated by the color matrix
 */
TEST_F(AttackerTests, foodChainColorMatrix_fullStrength)
{
    // Set color matrix to full strength for attacker color 0 attacking target color 1
    _parameters.attackerFoodChainColorMatrix.baseValue[0][1] = 1.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = createAttackerWithGenerator({100.0f, 100.0f}, 0.0f, 0);  // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);       // Color 1 target

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getCellRef(100);

    // Attack should happen at full strength
    EXPECT_TRUE(actualTarget._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, foodChainColorMatrix_zeroStrength)
{
    // Set color matrix to zero strength for attacker color 0 attacking target color 1
    _parameters.attackerFoodChainColorMatrix.baseValue[0][1] = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = createAttackerWithGenerator({100.0f, 100.0f}, 0.0f, 0);  // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);       // Color 1 target

    auto origTarget = data.getCellRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getCellRef(100);

    // No attack should happen because color matrix is zero
    EXPECT_TRUE(approxCompare(origTarget._usableEnergy, actualTarget._usableEnergy));
}

TEST_F(AttackerTests, outputSignal_noTarget)
{
    auto data = createAttackerWithGenerator({100.0f, 100.0f});
    // No target creature - nothing to attack

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getCellRef(1);

    // Attacker may have signal from generator but AttackerSuccess should be 0
    if (actualAttacker._signalState == SignalState_Active) {
        EXPECT_TRUE(approxCompare(0.0f, actualAttacker._signal._channels[Channels::AttackerSuccess]));
    }
}

/**
 * Test: No attacking of own creature cells
 * Cells belonging to the same creature should not be attacked
 */
TEST_F(AttackerTests, noAttackOnOwnCreatureCells)
{
    // Create a single creature with attacker and potential targets
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(AttackerDescription()),
        CellDescription().id(2).pos({101.0f, 100.0f}).cellType(GeneratorDescription().autoTriggerInterval(3)),
        CellDescription().id(3).pos({100.0f, 103.0f}).usableEnergy(100.0f),  // Same creature, in attack range
        CellDescription().id(4).pos({100.5f, 103.0f}).usableEnergy(100.0f),  // Same creature, in attack range
    }));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);

    auto origCell3 = data.getCellRef(3);
    auto origCell4 = data.getCellRef(4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell3 = actualData.getCellRef(3);
    auto actualCell4 = actualData.getCellRef(4);

    // Own creature cells should NOT be attacked
    EXPECT_TRUE(approxCompare(origCell3._usableEnergy, actualCell3._usableEnergy));
    EXPECT_TRUE(approxCompare(origCell4._usableEnergy, actualCell4._usableEnergy));
}

/**
 * Test: No attacking of offspring
 * Cells whose creature's ancestorId matches the attacker's creature id should not be attacked
 */
TEST_F(AttackerTests, noAttackOnOffspring)
{
    // Create parent creature with attacker
    auto data = createAttackerWithGenerator({100.0f, 100.0f});
    auto parentId = data._creatures.at(0)._id;

    // Create offspring creature with ancestorId pointing to parent
    data.addCreature(CreatureDescription().id(2).ancestorId(parentId).cells({
        CellDescription().id(100).pos({100.0f, 103.0f}).usableEnergy(100.0f),
        CellDescription().id(101).pos({100.5f, 103.0f}).usableEnergy(100.0f),
    }));
    data.addConnection(100, 101);

    auto origCell = data.getCellRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getCellRef(100);

    // Offspring cells should NOT be attacked
    EXPECT_TRUE(approxCompare(origCell._usableEnergy, actualCell._usableEnergy));
}

TEST_F(AttackerTests, attackOnNonOffspring)
{
    // Create attacker creature
    auto data = createAttackerWithGenerator({100.0f, 100.0f});

    // Create unrelated creature (no ancestorId relationship)
    data.addCreature(CreatureDescription().id(2).ancestorId(3).cells({
        CellDescription().id(100).pos({100.0f, 103.0f}).usableEnergy(100.0f),
        CellDescription().id(101).pos({100.5f, 103.0f}).usableEnergy(100.0f),
    }));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getCellRef(100);

    // Non-offspring cells should be attacked
    EXPECT_TRUE(actualTarget._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: No attacking of fixed cells
 * Cells with fixed=true should not be attacked
 */
TEST_F(AttackerTests, noAttackOnFixedCells)
{
    auto data = createAttackerWithGenerator({100.0f, 100.0f});
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 0, 100.0f, true), false);  // fixed=true

    auto origTarget = data.getCellRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getCellRef(100);

    // Fixed cells should NOT be attacked
    EXPECT_TRUE(approxCompare(origTarget._usableEnergy, actualTarget._usableEnergy));
}

/**
 * Test: Visible cone blocking by same creature connections
 * Attacks should be blocked when same-creature cell connections cross the ray to the target
 */
TEST_F(AttackerTests, rayBlockedBySameCreatureConnections)
{
    // Create attacker with connections that block the attack ray
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(AttackerDescription()),
        CellDescription().id(2).pos({101.0f, 100.0f}).cellType(GeneratorDescription().autoTriggerInterval(3)),
        // Create a connection that crosses the ray path to target at (100, 99)
        CellDescription().id(3).pos({99.0f, 99.0f}),
        CellDescription().id(4).pos({101.0f, 99.0f}),
    }));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);
    data.addConnection(1, 4);

    // Add target creature below (ray to target is blocked by connection 3-4)
    data.add(createTargetCreature({100.0f, 97.0f}), false);

    auto origTarget = data.getCellRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getCellRef(100);

    // Target should NOT be attacked because ray is blocked by same-creature connections
    EXPECT_TRUE(approxCompare(origTarget._usableEnergy, actualTarget._usableEnergy));
}

TEST_F(AttackerTests, rayNotBlockedByDifferentCreatureConnections)
{
    // Create attacker creature
    auto data = createAttackerWithGenerator({100.0f, 100.0f});

    // Create a different creature with connections that would cross the ray path
    data.addCreature(CreatureDescription().id(3).cells({
        CellDescription().id(50).pos({99.0f, 98.5f}),
        CellDescription().id(51).pos({101.0f, 98.5f}),
    }));
    data.addConnection(50, 51);

    // Add target creature below
    data.add(createTargetCreature({100.0f, 97.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget1 = actualData.getCellRef(100);
    auto actualTarget2 = actualData.getCellRef(50);

    // Both targets should be attacked because blocking connections belong to different creature
    EXPECT_TRUE(actualTarget1._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(actualTarget2._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, rayNotBlocked_noIntersection)
{
    // Create attacker with connections that do NOT block the attack ray
    auto data = Description().addCreature(CreatureDescription().id(1).cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(AttackerDescription()),
        CellDescription().id(2).pos({101.0f, 100.0f}).cellType(GeneratorDescription().autoTriggerInterval(3)),
        // Connections that don't intersect the ray to target
        CellDescription().id(3).pos({102.0f, 99.0f}),
        CellDescription().id(4).pos({103.0f, 99.0f}),
    }));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    // Add target creature at a position not blocked by connections
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getCellRef(100);

    // Target should be attacked because ray is not blocked
    EXPECT_TRUE(actualTarget._usableEnergy < 100.0f - NEAR_ZERO);
}
