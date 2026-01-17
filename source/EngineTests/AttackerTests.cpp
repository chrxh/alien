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
    Description createAttackerWithIncomingSignal(RealVector2D const& attackerPos, float attackerRawEnergy = 0.0f, int attackerColor = 0)
    {
        auto data = Description().addCreature({
            ObjectDescription().id(1).pos(attackerPos).color(attackerColor).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription())).rawEnergy(attackerRawEnergy)),
            ObjectDescription().id(2).pos({attackerPos.x + 1.0f, attackerPos.y}).color(attackerColor).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        }, CreatureDescription().id(1));
        data.addConnection(1, 2);
        return data;
    }

    // Helper to create a target creature at a given position
    Description createTargetCreature(RealVector2D const& pos, uint64_t creatureId = 2, int color = 0, float usableEnergy = 100.0f, bool fixed = false)
    {
        auto data = Description().addCreature({
            ObjectDescription().id(100).pos(pos).color(color).fixed(fixed).type(CellDescription().usableEnergy(usableEnergy)),
            ObjectDescription().id(101).pos({pos.x + 1.0f, pos.y}).color(color).fixed(fixed).type(CellDescription().usableEnergy(usableEnergy)),
        }, CreatureDescription().id(creatureId));
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
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f}, SimulationParameters::attackerMaxRawEnergyThreshold / 2);  // Below threshold

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);
    auto& origTarget = data.getObjectRef(100);
    origTarget.getCellRef()._rawEnergy = 100.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);  

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    // Attacker should attack because rawEnergy is below threshold
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(actualTarget.getCellRef()._rawEnergy, origTarget.getCellRef()._rawEnergy));

    // Attacker should have a signal with success value > 0
    ASSERT_TRUE(actualAttacker.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(actualAttacker.getCellRef()._signal._channels[Channels::AttackerSuccess] > NEAR_ZERO);
}

TEST_F(AttackerTests, maxRawEnergyThreshold_aboveThreshold)
{
    // Create attacker with rawEnergy above threshold
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f}, SimulationParameters::attackerMaxRawEnergyThreshold + NEAR_ZERO);  // Above threshold

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    // Attacker should NOT attack because rawEnergy is above threshold
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(actualTarget.getCellRef()._rawEnergy, origTarget.getCellRef()._rawEnergy));

    // Attacker should have a signal with success value = 0
    ASSERT_TRUE(actualAttacker.getCellRef()._signalState == SignalState_Active);
    EXPECT_TRUE(approxCompare(0.0f, actualAttacker.getCellRef()._signal._channels[Channels::AttackerSuccess]));
}

TEST_F(AttackerTests, maxRawEnergyThreshold_outsideRange)
{
    // Create attacker with rawEnergy above threshold
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f});

    // Add target creature outside attack radius
    data.add(createTargetCreature({100.0f, 100.0f + _parameters.attackerRadius.value[0] + 0.01f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    // Attacker should NOT attack because rawEnergy is above threshold
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
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

    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f}, 0.0f, 0);  // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);       // Color 1 target

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Attack should happen at full strength
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, foodChainColorMatrix_zeroStrength)
{
    // Set color matrix to zero strength for attacker color 0 attacking target color 1
    _parameters.attackerFoodChainColorMatrix.baseValue[0][1] = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f}, 0.0f, 0);  // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);       // Color 1 target

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // No attack should happen because color matrix is zero
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, outputSignal_noTarget)
{
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f});
    // No target creature - nothing to attack

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);

    // Attacker may have signal from generator but AttackerSuccess should be 0
    if (actualAttacker.getCellRef()._signalState == SignalState_Active) {
        EXPECT_TRUE(approxCompare(0.0f, actualAttacker.getCellRef()._signal._channels[Channels::AttackerSuccess]));
    }
}

/**
 * Test: No attacking of own creature cells
 * Cells belonging to the same creature should not be attacked
 */
TEST_F(AttackerTests, noAttackOnOwnCreatureCells)
{
    // Create a single creature with attacker and potential targets
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        ObjectDescription().id(3).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),  // Same creature, in attack range
        ObjectDescription().id(4).pos({100.5f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),  // Same creature, in attack range
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);

    auto origCell3 = data.getObjectRef(3);
    auto origCell4 = data.getObjectRef(4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell3 = actualData.getObjectRef(3);
    auto actualCell4 = actualData.getObjectRef(4);

    // Own creature cells should NOT be attacked
    EXPECT_TRUE(approxCompare(origCell3.getCellRef()._usableEnergy, actualCell3.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(origCell4.getCellRef()._usableEnergy, actualCell4.getCellRef()._usableEnergy));
}

/**
 * Test: No attacking of offspring
 * Cells whose creature's ancestorId matches the attacker's creature id should not be attacked
 */
TEST_F(AttackerTests, noAttackOnOffspring)
{
    // Create parent creature with attacker
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f});
    auto parentId = data._creatures.at(0)._id;

    // Create offspring creature with ancestorId pointing to parent
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
        ObjectDescription().id(101).pos({100.5f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
    }, CreatureDescription().id(2).ancestorId(parentId));
    data.addConnection(100, 101);

    auto origCell = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(100);

    // Offspring cells should NOT be attacked
    EXPECT_TRUE(approxCompare(origCell.getCellRef()._usableEnergy, actualCell.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, attackOnNonOffspring)
{
    // Create attacker creature
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f});

    // Create unrelated creature (no ancestorId relationship)
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
        ObjectDescription().id(101).pos({100.5f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
    }, CreatureDescription().id(2).ancestorId(3));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Non-offspring cells should be attacked
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: No attacking of fixed cells
 * Cells with fixed=true should not be attacked
 */
TEST_F(AttackerTests, noAttackOnFixedCells)
{
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f});
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 0, 100.0f, true), false);  // fixed=true

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Fixed cells should NOT be attacked
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: Visible cone blocking by same creature connections
 * Attacks should be blocked when same-creature cell connections cross the ray to the target
 */
TEST_F(AttackerTests, rayBlockedBySameCreatureConnections)
{
    // Create attacker with connections that block the attack ray
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        // Create a connection that crosses the ray path to target at (100, 99)
        ObjectDescription().id(3).pos({99.0f, 99.0f}),
        ObjectDescription().id(4).pos({101.0f, 99.0f}),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);
    data.addConnection(1, 4);

    // Add target creature below (ray to target is blocked by connection 3-4)
    data.add(createTargetCreature({100.0f, 97.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because ray is blocked by same-creature connections
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, rayNotBlockedByDifferentCreatureConnections)
{
    // Create attacker creature
    auto data = createAttackerWithIncomingSignal({100.0f, 100.0f});

    // Create a different creature with connections that would cross the ray path
    data.addCreature({
        ObjectDescription().id(50).pos({99.0f, 98.5f}),
        ObjectDescription().id(51).pos({101.0f, 98.5f}),
    }, CreatureDescription().id(3));
    data.addConnection(50, 51);

    // Add target creature below
    data.add(createTargetCreature({100.0f, 97.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget1 = actualData.getObjectRef(100);
    auto actualTarget2 = actualData.getObjectRef(50);

    // Both targets should be attacked because blocking connections belong to different creature
    EXPECT_TRUE(actualTarget1.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(actualTarget2.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, rayNotBlocked_noIntersection)
{
    // Create attacker with connections that do NOT block the attack ray
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        // Connections that don't intersect the ray to target
        ObjectDescription().id(3).pos({102.0f, 99.0f}),
        ObjectDescription().id(4).pos({103.0f, 99.0f}),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    // Add target creature at a position not blocked by connections
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because ray is not blocked
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: restrictToColor
 * The attacker should only attack cells of the specified color
 */
TEST_F(AttackerTests, restrictToColor_matchingColor)
{
    // Create attacker that only attacks color 1
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).color(0).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToColor(1)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with matching color (color 1)
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because color matches restriction
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, restrictToColor_nonMatchingColor)
{
    // Create attacker that only attacks color 1
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).color(0).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToColor(1)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with non-matching color (color 0)
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 0), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because color does not match restriction
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, restrictToColor_noRestriction)
{
    // Create attacker with no color restriction
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).color(0).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with different color (color 2)
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because there is no color restriction
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: minNumCells
 * The attacker should only attack creatures with at least the specified number of cells
 */
TEST_F(AttackerTests, minNumCells_creatureAboveMinimum)
{
    // Create attacker that only attacks creatures with at least 2 cells
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().minNumCells(2)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with 2 cells (meets minimum)
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because creature has at least 2 cells
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, minNumCells_creatureBelowMinimum)
{
    // Create attacker that only attacks creatures with at least 3 cells
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().minNumCells(3)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with only 2 cells (below minimum)
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because creature has fewer cells than minimum
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: maxNumCells
 * The attacker should only attack creatures with at most the specified number of cells
 */
TEST_F(AttackerTests, maxNumCells_creatureBelowMaximum)
{
    // Create attacker that only attacks creatures with at most 5 cells
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().maxNumCells(5)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with 2 cells (below maximum)
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because creature has at most 5 cells
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, maxNumCells_creatureAboveMaximum)
{
    // Create attacker that only attacks creatures with at most 1 cell
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().maxNumCells(1)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with 2 cells (above maximum)
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because creature has more cells than maximum
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: restrictToLineage
 * The attacker should attack only creatures of the same or different lineage
 */
TEST_F(AttackerTests, restrictToLineage_sameLineage_matching)
{
    // Create attacker creature with lineage 42, restricted to same lineage
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToLineage(LineageRestriction_SameLineage)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1).lineageId(42));
    data.addConnection(1, 2);

    // Add target creature with same lineage (42)
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
        ObjectDescription().id(101).pos({101.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
    }, CreatureDescription().id(2).lineageId(42));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because lineage matches
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, restrictToLineage_sameLineage_notMatching)
{
    // Create attacker creature with lineage 42, restricted to same lineage
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToLineage(LineageRestriction_SameLineage)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1).lineageId(42));
    data.addConnection(1, 2);

    // Add target creature with different lineage (43)
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
        ObjectDescription().id(101).pos({101.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
    }, CreatureDescription().id(2).lineageId(43));
    data.addConnection(100, 101);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because lineage does not match
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, restrictToLineage_otherLineage_matching)
{
    // Create attacker creature with lineage 42, restricted to other lineage
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToLineage(LineageRestriction_OtherLineage)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1).lineageId(42));
    data.addConnection(1, 2);

    // Add target creature with different lineage (43)
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
        ObjectDescription().id(101).pos({101.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
    }, CreatureDescription().id(2).lineageId(43));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because lineage is different
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, restrictToLineage_otherLineage_notMatching)
{
    // Create attacker creature with lineage 42, restricted to other lineage
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToLineage(LineageRestriction_OtherLineage)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1).lineageId(42));
    data.addConnection(1, 2);

    // Add target creature with same lineage (42)
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
        ObjectDescription().id(101).pos({101.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
    }, CreatureDescription().id(2).lineageId(42));
    data.addConnection(100, 101);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because lineage is the same
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, restrictToLineage_noRestriction)
{
    // Create attacker creature with lineage 42, no lineage restriction
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1).lineageId(42));
    data.addConnection(1, 2);

    // Add target creature with different lineage (43)
    data.addCreature({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
        ObjectDescription().id(101).pos({101.0f, 103.0f}).type(CellDescription().usableEnergy(100.0f)),
    }, CreatureDescription().id(2).lineageId(43));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because there is no lineage restriction
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: Combined restrictions
 * The attacker should only attack cells that meet all restrictions
 */
TEST_F(AttackerTests, combinedRestrictions_allMatch)
{
    // Create attacker with multiple restrictions:
    // - restrictToColor: 1
    // - minNumCells: 2
    // - maxNumCells: 5
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).color(0).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToColor(1).minNumCells(2).maxNumCells(5)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature that meets all restrictions: color 1, 2 cells
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because all restrictions are met
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, combinedRestrictions_colorMismatch)
{
    // Create attacker with restrictions: color 1, minNumCells 2
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).color(0).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription().restrictToColor(1).minNumCells(2)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature with wrong color but correct cell count: color 0, 2 cells
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 0), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because color does not match
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: AttackerMode_FreeCell tests
 * The attacker in FreeCell mode should only attack free cells (not part of a creature)
 */
TEST_F(AttackerTests, freeCellMode_attackFreeCell)
{
    // Create attacker creature in FreeCell mode
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackFreeCellDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add a free cell (not part of a creature) - using FreeCellDescription
    data.addObjects({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(FreeCellDescription()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should be attacked in FreeCell mode
    EXPECT_TRUE(actualTarget.getFreeCellRef()._rawEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, freeCellMode_attackFreeCell_matchingColor)
{
    // Create attacker creature in FreeCell mode with color restriction to color 1
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).color(0).type(CellDescription().cellType(AttackerDescription().mode(AttackFreeCellDescription().restrictToColor(1)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add a free cell with matching color (color 1)
    data.addObjects({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).color(1).type(FreeCellDescription()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should be attacked because color matches restriction
    EXPECT_TRUE(actualTarget.getFreeCellRef()._rawEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, freeCellMode_attackFreeCell_nonMatchingColor)
{
    // Create attacker creature in FreeCell mode with color restriction to color 1
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).color(0).type(CellDescription().cellType(AttackerDescription().mode(AttackFreeCellDescription().restrictToColor(1)))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).color(0).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add a free cell with non-matching color (color 0)
    data.addObjects({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).color(0).type(FreeCellDescription()),
    });

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should NOT be attacked because color does not match restriction
    EXPECT_TRUE(approxCompare(origTarget.getFreeCellRef()._rawEnergy, actualTarget.getFreeCellRef()._rawEnergy));
}

TEST_F(AttackerTests, freeCellMode_doesNotAttackCreature)
{
    // Create attacker creature in FreeCell mode
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackFreeCellDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add target creature (cells that are part of a creature, not free cells)
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Creature should NOT be attacked in FreeCell mode
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, creatureMode_doesNotAttackFreeCell)
{
    // Create attacker creature in Creature mode
    auto data = Description().addCreature({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(AttackerDescription().mode(AttackCreatureDescription()))),
        ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
    }, CreatureDescription().id(1));
    data.addConnection(1, 2);

    // Add a free cell (not part of a creature)
    data.addObjects({
        ObjectDescription().id(100).pos({100.0f, 103.0f}).type(FreeCellDescription()),
    });

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should NOT be attacked in Creature mode
    EXPECT_TRUE(approxCompare(origTarget.getFreeCellRef()._rawEnergy, actualTarget.getFreeCellRef()._rawEnergy));
}
