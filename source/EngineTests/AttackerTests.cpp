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
    // Helper to create an attacker creature with neural net bias for activation and a sensor cell with lastMatch
    // For creature attack mode, the sensor's lastMatch.creatureId should match the lower 16 bits of the target creature's id
    Desc createAttacker(
        RealVector2D const& attackerPos,
        RealVector2D const& targetPos,
        uint64_t targetCreatureId = 2,
        float attackerRawEnergy = 0.0f,
        int attackerColor = 0)
    {
        // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
        NeuralNetworkDesc nn;
        nn._biases[Channels::CellTypeActivation] = 1.0f;

        // Create a sensor with lastMatch pointing to the target creature
        SensorLastMatchDesc lastMatch;
        lastMatch._creatureId = targetCreatureId & 0xffff;  // Sensor stores only lower 16 bits
        lastMatch._pos = targetPos;

        auto data = Desc().addCreature(
            {
                ObjectDesc()
                    .id(1)
                    .pos(attackerPos)
                    .color(attackerColor)
                    .type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).rawEnergy(attackerRawEnergy).neuralNetwork(nn)),
                ObjectDesc()
                    .id(2)
                    .pos({attackerPos.x + 1.0f, attackerPos.y})
                    .color(attackerColor)
                    .type(CellDesc().cellType(SensorDesc().autoTriggerInterval(std::nullopt).lastMatch(lastMatch))),
            },
            CreatureDesc().id(1));
        data.addConnection(1, 2);
        return data;
    }

    // Helper to create a target creature at a given position
    Desc createTargetCreature(RealVector2D const& pos, uint64_t creatureId = 2, int color = 0, float usableEnergy = 100.0f, bool fixed = false)
    {
        auto data = Desc().addCreature(
            {
                ObjectDesc().id(100).pos(pos).color(color).fixed(fixed).type(CellDesc().usableEnergy(usableEnergy)),
            },
            CreatureDesc().id(creatureId));
        return data;
    }
};

/**
 * Test: attackerMaxRawEnergyThreshold
 * The attacker should not attack when its rawEnergy exceeds the threshold (2.0f)
 */
TEST_F(AttackerTests, maxRawEnergyThreshold_belowThreshold)
{
    // Create attacker with rawEnergy below threshold and sensor targeting creature 2
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2, SimulationParameters::attackerMaxRawEnergyThreshold / 2);

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
    EXPECT_TRUE(actualAttacker.getCellRef()._signal._channels[Channels::AttackerSuccess] > NEAR_ZERO);
}

TEST_F(AttackerTests, maxRawEnergyThreshold_aboveThreshold)
{
    // Create attacker with rawEnergy above threshold
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2, SimulationParameters::attackerMaxRawEnergyThreshold + NEAR_ZERO);

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
    EXPECT_TRUE(approxCompare(0.0f, actualAttacker.getCellRef()._signal._channels[Channels::AttackerSuccess]));
}

TEST_F(AttackerTests, maxRawEnergyThreshold_outsideRange)
{
    // Create attacker with sensor targeting creature 2
    auto targetPos = RealVector2D{100.0f, 100.0f + _parameters.attackerRadius.value[0] + 0.01f};
    auto data = createAttacker({100.0f, 100.0f}, targetPos, 2);

    // Add target creature outside attack radius
    data.add(createTargetCreature(targetPos), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    // Attacker should NOT attack because target is outside attack radius
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

    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2, 0.0f, 0);  // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);               // Color 1 target

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Attack should happen at full strength
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_EQ(1.0f, actualTarget.getCellRef()._signal._channels.at(Channels::AttackerNotify));  // Notify attacked cell
}

TEST_F(AttackerTests, foodChainColorMatrix_zeroStrength)
{
    // Set color matrix to zero strength for attacker color 0 attacking target color 1
    _parameters.attackerFoodChainColorMatrix.baseValue[0][1] = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2, 0.0f, 0);  // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);               // Color 1 target

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
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 999);  // Sensor targets non-existent creature

    // No target creature - nothing to attack

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);

    // Attacker may have signal from generator but AttackerSuccess should be 0
    EXPECT_TRUE(approxCompare(0.0f, actualAttacker.getCellRef()._signal._channels[Channels::AttackerSuccess]));
}

/**
 * Test: No attacking of own creature cells
 * Cells belonging to the same creature should not be attacked
 */
TEST_F(AttackerTests, noAttackOnOwnCreatureCells)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create a sensor with lastMatch pointing to creature 1 (same creature)
    SensorLastMatchDesc lastMatch;
    lastMatch._creatureId = 1;  // Same creature id
    lastMatch._pos = {100.0f, 103.0f};

    // Create a single creature with attacker, sensor, and potential targets
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTriggerInterval(std::nullopt).lastMatch(lastMatch))),
            ObjectDesc().id(3).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),  // Same creature, in attack range
            ObjectDesc().id(4).pos({100.5f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),  // Same creature, in attack range
        },
        CreatureDesc().id(1));
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
    // Create parent creature with attacker and sensor targeting creature 2
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2);
    auto parentId = data._creatures.at(0)._id;

    // Create offspring creature with ancestorId pointing to parent
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
            ObjectDesc().id(101).pos({100.5f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
        },
        CreatureDesc().id(2).ancestorId(parentId));
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
    // Create attacker creature with sensor targeting creature 2
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2);

    // Create unrelated creature (no ancestorId relationship)
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
            ObjectDesc().id(101).pos({100.5f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
        },
        CreatureDesc().id(2).ancestorId(3));
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
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2);
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
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create a sensor with lastMatch pointing to creature 2
    SensorLastMatchDesc lastMatch;
    lastMatch._creatureId = 2;
    lastMatch._pos = {100.0f, 97.0f};

    // Create attacker with connections that block the attack ray
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().lastMatch(lastMatch))),
            // Create a connection that crosses the ray path to target at (100, 99)
            ObjectDesc().id(3).pos({99.0f, 99.0f}),
            ObjectDesc().id(4).pos({101.0f, 99.0f}),
        },
        CreatureDesc().id(1));
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
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature with sensor targeting creatures 2 and 3
    SensorLastMatchDesc lastMatch1;
    lastMatch1._creatureId = 2;
    lastMatch1._pos = {100.0f, 97.0f};

    SensorLastMatchDesc lastMatch2;
    lastMatch2._creatureId = 3;
    lastMatch2._pos = {99.0f, 98.5f};

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTriggerInterval(std::nullopt).lastMatch(lastMatch2))),
            ObjectDesc().id(3).pos({99.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTriggerInterval(std::nullopt).lastMatch(lastMatch1))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    // Create a different creature with connections that would cross the ray path
    data.addCreature(
        {
            ObjectDesc().id(50).pos({99.0f, 98.5f}),
            ObjectDesc().id(51).pos({101.0f, 98.5f}),
        },
        CreatureDesc().id(3));
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
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create a sensor with lastMatch pointing to creature 2
    SensorLastMatchDesc lastMatch;
    lastMatch._creatureId = 2;
    lastMatch._pos = {100.0f, 103.0f};

    // Create attacker with connections that do NOT block the attack ray
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTriggerInterval(std::nullopt).lastMatch(lastMatch))),
            // Connections that don't intersect the ray to target
            ObjectDesc().id(3).pos({102.0f, 99.0f}),
            ObjectDesc().id(4).pos({103.0f, 99.0f}),
        },
        CreatureDesc().id(1));
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
 * Test: Sensor-based targeting
 * The attacker should only attack creatures whose creatureId matches sensor lastMatch
 */
TEST_F(AttackerTests, sensorTargeting_matchingCreatureId)
{
    // Create attacker with sensor targeting creature 2
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 2);

    // Add target creature with matching creatureId
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because creatureId matches sensor lastMatch
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, sensorTargeting_nonMatchingCreatureId)
{
    // Create attacker with sensor targeting creature 3
    auto data = createAttacker({100.0f, 100.0f}, {100.0f, 103.0f}, 3);

    // Add target creature with non-matching creatureId (creature 2)
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because creatureId does not match sensor lastMatch
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, sensorTargeting_noSensorWithLastMatch)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker without sensor (single cell is sufficient for this test)
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).color(0).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because no sensor has lastMatch
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, sensorTargeting_multipleTargets)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create a sensor with lastMatch pointing to creature 2 and another to creature 3
    SensorLastMatchDesc lastMatch1;
    lastMatch1._creatureId = 2;
    lastMatch1._pos = {100.0f, 103.0f};

    SensorLastMatchDesc lastMatch2;
    lastMatch2._creatureId = 4;
    lastMatch2._pos = {100.0f, 97.0f};

    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTriggerInterval(std::nullopt).lastMatch(lastMatch2))),
            ObjectDesc().id(3).pos({99.0f, 100.0f}).type(CellDesc().cellType(SensorDesc().autoTriggerInterval(std::nullopt).lastMatch(lastMatch1))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    // Add target creature 2 and creature 4
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);
    data.addCreature(
        {
            ObjectDesc().id(200).pos({100.0f, 97.0f}).type(CellDesc().usableEnergy(100.0f)),
        },
        CreatureDesc().id(4));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget1 = actualData.getObjectRef(100);
    auto actualTarget2 = actualData.getObjectRef(200);

    // Both targets should be attacked because both creatureIds match sensor lastMatches
    EXPECT_TRUE(actualTarget1.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(actualTarget2.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: AttackerMode_FreeCell tests
 * The attacker in FreeCell mode should only attack free cells (not part of a creature)
 */
TEST_F(AttackerTests, freeCellMode_attackFreeCell)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode (single cell is sufficient)
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc())).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell (not part of a creature) - using FreeCellDesc
    data.addObjects({
        ObjectDesc().id(100).pos({100.0f, 103.0f}).type(FreeCellDesc()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should be attacked in FreeCell mode
    EXPECT_TRUE(actualTarget.getFreeCellRef()._energy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, freeCellMode_attackFreeCell_matchingColor)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode with color restriction to color 1 (single cell is sufficient)
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc().restrictToColor(1))).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell with matching color (color 1)
    data.addObjects({
        ObjectDesc().id(100).pos({100.0f, 103.0f}).color(1).type(FreeCellDesc()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should be attacked because color matches restriction
    EXPECT_TRUE(actualTarget.getFreeCellRef()._energy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, freeCellMode_attackFreeCell_nonMatchingColor)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode with color restriction to color 1 (single cell is sufficient)
    auto data = Desc().addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .color(0)
                .type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc().restrictToColor(1))).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell with non-matching color (color 0)
    data.addObjects({
        ObjectDesc().id(100).pos({100.0f, 103.0f}).color(0).type(FreeCellDesc()),
    });

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should NOT be attacked because color does not match restriction
    EXPECT_TRUE(approxCompare(origTarget.getFreeCellRef()._energy, actualTarget.getFreeCellRef()._energy));
}

TEST_F(AttackerTests, freeCellMode_doesNotAttackCreature)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode (single cell is sufficient)
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc())).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

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
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetworkDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in Creature mode (single cell is sufficient)
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell (not part of a creature)
    data.addObjects({
        ObjectDesc().id(100).pos({100.0f, 103.0f}).type(FreeCellDesc()),
    });

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should NOT be attacked in Creature mode
    EXPECT_TRUE(approxCompare(origTarget.getFreeCellRef()._energy, actualTarget.getFreeCellRef()._energy));
}
