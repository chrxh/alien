#include <gtest/gtest.h>

#include <Data/Interface/DescEditService.h>
#include <Data/Interface/Descs.h>

#include <Engine/Interface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class AttackerTests : public IntegrationTestFramework
{
public:
    AttackerTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        _parameters.attackerStrength.value = 0.1f;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
            _parameters.attackerEnergyCost.baseValue[i] = 0;
            _parameters.attackerRadius.value[i] = 3.5f;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~AttackerTests() override = default;

protected:
    // Helper to create an attacker creature with neural net bias for activation and a sensor cell which scans in every cycle
    ContentDesc createAttacker(const RealVector2D& attackerPos, float attackerRawEnergy = 0.0f, int attackerColor = 0, int sensorRestrictToColors = 0x3FF)
    {
        auto data = ContentDesc().addCreature(
            {
                ObjectDesc()
                .id(1)
                .pos(attackerPos)
                .color(attackerColor)
                .type(createAttackerCell().rawEnergy(attackerRawEnergy)),
                ObjectDesc()
                .id(2)
                .pos({attackerPos.x + 1.0f, attackerPos.y})
                .color(attackerColor)
                .type(createSensorCell(SensorDesc().mode(DetectCreatureDesc().restrictToColors(sensorRestrictToColors)))),
            },
            CreatureDesc().id(1));
        data.addConnection(1, 2);
        return data;
    }

    // Attacker cell which is triggered by a neural net bias and ignores the signals of connected cells
    CellDesc createAttackerCell()
    {
        auto nn = NeuralNetDesc().bias(Channels::CellTypeActivation, 1.0f).connectionWeight(0, 0.0f);
        return CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn);
    }

    // A sensor only scans with a front angle
    CellDesc createSensorCell(SensorDesc const& sensor) { return CellDesc().frontAngle(0.0f).cellType(sensor); }

    std::optional<int> getLastMatchedCreatureIdPart(ContentDesc const& data, uint64_t sensorId)
    {
        auto const& lastMatch = std::get<SensorDesc>(data.getObjectRef(sensorId).getCellRef()._cellType)._lastMatch;
        if (!lastMatch.has_value()) {
            return std::nullopt;
        }
        return lastMatch->_creatureIdPart;
    }

    // Helper to create a target creature at a given position
    ContentDesc createTargetCreature(const RealVector2D& pos, uint64_t creatureId = 2, int color = 0, float usableEnergy = 100.0f, bool fixed = false)
    {
        auto data = ContentDesc().addCreature(
            {
                ObjectDesc().id(100).pos(pos).color(color).isStatic(fixed).type(CellDesc().usableEnergy(usableEnergy)),
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
    // Create attacker with rawEnergy below threshold
    auto data = createAttacker({100.0f, 100.0f}, SimulationParameters::attackerMaxRawEnergyThreshold / 2);

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);
    auto& origTarget = data.getObjectRef(100);
    origTarget.getCellRef()._rawEnergy = 100.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    // Attacker should attack because rawEnergy is below threshold
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(actualTarget.getCellRef()._rawEnergy, origTarget.getCellRef()._rawEnergy));

    // Attacker should have a signal with success value > 0
    EXPECT_TRUE(actualAttacker.getCellRef()._neuralActivity._signals[Channels::AttackerSuccess] > NEAR_ZERO);
}

TEST_F(AttackerTests, maxRawEnergyThreshold_aboveThreshold)
{
    // Create attacker with rawEnergy above threshold
    auto data = createAttacker({100.0f, 100.0f}, SimulationParameters::attackerMaxRawEnergyThreshold + NEAR_ZERO);

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    // Attacker should NOT attack because rawEnergy is above threshold
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(actualTarget.getCellRef()._rawEnergy, origTarget.getCellRef()._rawEnergy));

    // Attacker should have a signal with success value = 0
    EXPECT_TRUE(approxCompare(0.0f, actualAttacker.getCellRef()._neuralActivity._signals[Channels::AttackerSuccess]));
}

TEST_F(AttackerTests, maxRawEnergyThreshold_outsideRange)
{
    auto targetPos = RealVector2D{100.0f, 100.0f + _parameters.attackerRadius.value[0] + 0.01f};
    auto data = createAttacker({100.0f, 100.0f});

    // Add target creature outside attack radius
    data.add(createTargetCreature(targetPos), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Attacker should NOT attack because target is outside attack radius although the sensor detected it
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
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

    auto data = createAttacker({100.0f, 100.0f}, 0.0f, 0);                     // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);              // Color 1 target

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Attack should happen at full strength
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_EQ(CellEvent_Attacked, actualTarget.getCellRef()._event);  // Notify attacked cell
    EXPECT_TRUE(actualTarget.getCellRef()._eventCounter > 0);
}

TEST_F(AttackerTests, foodChainColorMatrix_zeroStrength)
{
    // Set color matrix to zero strength for attacker color 0 attacking target color 1
    _parameters.attackerFoodChainColorMatrix.baseValue[0][1] = 0.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = createAttacker({100.0f, 100.0f}, 0.0f, 0);                     // Color 0 attacker
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);              // Color 1 target

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // No attack should happen because color matrix is zero
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, outputSignal_noTarget)
{
    auto data = createAttacker({100.0f, 100.0f});

    // No target creature - nothing to attack

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);

    // Attacker may have signal from generator but AttackerSuccess should be 0
    EXPECT_TRUE(approxCompare(0.0f, actualAttacker.getCellRef()._neuralActivity._signals[Channels::AttackerSuccess]));
}

/**
 * Test: No attacking of own creature cells
 * Cells belonging to the same creature should not be attacked
 */
TEST_F(AttackerTests, noAttackOnOwnCreatureCells)
{
    // Create a single creature with attacker, sensor, and potential targets
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(createSensorCell(SensorDesc())),
            ObjectDesc().id(3).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f)), // Same creature, in attack range
            ObjectDesc().id(4).pos({100.5f, 103.0f}).type(CellDesc().usableEnergy(100.0f)), // Same creature, in attack range
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);

    auto origCell3 = data.getObjectRef(3);
    auto origCell4 = data.getObjectRef(4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

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
    auto data = createAttacker({100.0f, 100.0f});
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
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(100);

    // Offspring cells should NOT be attacked although the sensor detected them
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_TRUE(approxCompare(origCell.getCellRef()._usableEnergy, actualCell.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, attackOnNonOffspring)
{
    auto data = createAttacker({100.0f, 100.0f});

    // Create unrelated creature (no ancestorId relationship)
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
            ObjectDesc().id(101).pos({100.5f, 103.0f}).type(CellDesc().usableEnergy(100.0f)),
        },
        CreatureDesc().id(2).ancestorId(3));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Non-offspring cells should be attacked
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, attackDrainsDepotStoredUsableEnergy)
{
    auto data = createAttacker({100.0f, 100.0f});
    data.addCreature(
        {ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f).cellType(DepotDesc().storedUsableEnergy(200.0f)))},
        CreatureDesc().id(2));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_TRUE(approxCompare(100.0f, actualTarget.getCellRef()._usableEnergy));
    EXPECT_TRUE(std::get<DepotDesc>(actualTarget.getCellRef()._cellType)._storedUsableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(actualAttacker.getCellRef()._rawEnergy > NEAR_ZERO);
}

TEST_F(AttackerTests, attackDrainsConstructorReservedEnergy)
{
    auto data = createAttacker({100.0f, 100.0f});
    data.addCreature(
        {ObjectDesc().id(100).pos({100.0f, 103.0f}).type(CellDesc().usableEnergy(100.0f).constructor(ConstructorDesc().reservedEnergy(200.0f)))},
        CreatureDesc().id(2));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualAttacker = actualData.getObjectRef(1);
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_TRUE(approxCompare(100.0f, actualTarget.getCellRef()._usableEnergy));
    EXPECT_TRUE(actualTarget.getCellRef()._constructor->_reservedEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(actualAttacker.getCellRef()._rawEnergy > NEAR_ZERO);
}

/**
 * Test: No attacking of fixed cells
 * Cells with fixed=true should not be attacked
 */
TEST_F(AttackerTests, noAttackOnFixedCells)
{
    auto data = createAttacker({100.0f, 100.0f});
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 0, 100.0f, true), false); // fixed=true

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Fixed cells should NOT be attacked although the sensor detected them
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: Visible cone blocking by same creature connections
 * Attacks should be blocked when same-creature cell connections cross the ray to the target
 */
TEST_F(AttackerTests, rayBlockedBySameCreatureConnections)
{
    // Create attacker with connections that block the attack ray, the sensor sees the target past them
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({103.0f, 100.0f}).type(createSensorCell(SensorDesc())),
            // Create a connection that crosses the ray path to target at (100, 99)
            ObjectDesc().id(3).pos({99.0f, 99.0f}),
            ObjectDesc().id(4).pos({101.0f, 99.0f}),
        },
        CreatureDesc().id(1));
    data.addConnection(4, 2);
    data.addConnection(1, 3);
    data.addConnection(3, 4);
    data.addConnection(1, 4);

    // Add target creature below (ray to target is blocked by connection 3-4)
    data.add(createTargetCreature({100.0f, 97.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because ray is blocked by same-creature connections
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, rayNotBlockedByDifferentCreatureConnections)
{
    // The sensor detects creature 3 as match and creature 2 as nearby creature
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(createSensorCell(SensorDesc())),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);

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
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

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
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(createSensorCell(SensorDesc())),
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
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because ray is not blocked
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: Sensor-based targeting
 * The attacker should only attack creatures which a sensor of its creature has detected in the current cycle
 */
TEST_F(AttackerTests, sensorTargeting_detectedCreature)
{
    auto data = createAttacker({100.0f, 100.0f});
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, sensorTargeting_undetectedCreature)
{
    auto data = createAttacker({100.0f, 100.0f});
    std::get<SensorDesc>(data.getObjectRef(2).getCellRef()._cellType)._maxRange = 2;

    // Target is within the attack radius but beyond the sensor range
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_FALSE(getLastMatchedCreatureIdPart(actualData, 2).has_value());
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, sensorTargeting_sensorWithoutScan)
{
    // The sensor has a last match but does not scan in this cycle
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc()
                .id(2)
                .pos({101.0f, 100.0f})
                .type(createSensorCell(SensorDesc().autoTrigger(false).lastMatch(SensorLastMatchDesc().creatureIdPart(2).pos({100.0f, 103.0f})))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, sensorTargeting_detectionsOfPreviousCycleExpire)
{
    auto data = createAttacker({100.0f, 100.0f});
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_TRUE(actualData.getObjectRef(100).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);

    // The sensor no longer detects the target
    std::get<SensorDesc>(actualData.getObjectRef(2).getCellRef()._cellType)._maxRange = 2;

    // Reset the energies so that only the missing detection prevents another attack
    actualData.getObjectRef(1).getCellRef()._rawEnergy = 0.0f;
    actualData.getObjectRef(100).getCellRef()._usableEnergy = 100.0f;

    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(100.0f, actualData.getObjectRef(100).getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, sensorTargeting_noSensor)
{
    // Create attacker without sensor (single cell is sufficient for this test)
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).color(0).type(createAttackerCell()),
        },
        CreatureDesc().id(1));

    // Add target creature within attack radius
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, sensorTargeting_sensorFarFromAttacker)
{
    // The sensor is connected to the attacker through a chain of cells
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(3).pos({101.0f, 100.0f}),
            ObjectDesc().id(4).pos({102.0f, 100.0f}),
            ObjectDesc().id(5).pos({103.0f, 100.0f}),
            ObjectDesc().id(6).pos({104.0f, 100.0f}),
            ObjectDesc().id(7).pos({105.0f, 100.0f}),
            ObjectDesc().id(2).pos({106.0f, 100.0f}).type(createSensorCell(SensorDesc())),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 3);
    data.addConnection(3, 4);
    data.addConnection(4, 5);
    data.addConnection(5, 6);
    data.addConnection(6, 7);
    data.addConnection(7, 2);
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, sensorTargeting_multipleSensors)
{
    // Each sensor detects the creature nearest to it
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.5f, 100.5f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.5f, 100.5f}).type(createSensorCell(SensorDesc())),
            ObjectDesc().id(3).pos({99.5f, 100.5f}).type(createSensorCell(SensorDesc())),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    data.add(createTargetCreature({101.5f, 103.5f}, 2), false);
    data.addCreature({ObjectDesc().id(200).pos({99.5f, 97.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(4));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget1 = actualData.getObjectRef(100);
    auto actualTarget2 = actualData.getObjectRef(200);

    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_EQ(std::optional(4), getLastMatchedCreatureIdPart(actualData, 3));
    EXPECT_TRUE(actualTarget1.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(actualTarget2.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Tests for the nearby creatures which a sensor in SensorMode_DetectCreature detects together with the match
 * The sensor is at (101.5, 100.5), the match is creature 2 at a distance of 2, the other creatures are farther away from the sensor
 */
TEST_F(AttackerTests, sensorTargeting_nearbyCreatures)
{
    auto data = createAttacker({100.5f, 100.5f});
    data.addCreature({ObjectDesc().id(100).pos({101.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(2));
    data.addCreature({ObjectDesc().id(200).pos({102.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(3));
    data.addCreature({ObjectDesc().id(300).pos({99.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(4));
    data.addCreature({ObjectDesc().id(400).pos({100.5f, 97.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(5));  // Beyond the nearby radius

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    for (auto id : {100, 200, 300}) {
        auto const& target = actualData.getObjectRef(id).getCellRef();
        EXPECT_TRUE(target._usableEnergy < 100.0f - NEAR_ZERO);
        EXPECT_EQ(CellEvent_Attacked, target._event);
        EXPECT_TRUE(target._eventCounter > 0);
        EXPECT_TRUE(approxCompare(RealVector2D{100.5f, 100.5f}, target._eventPos));
    }
    auto const& notTarget = actualData.getObjectRef(400).getCellRef();
    EXPECT_TRUE(approxCompare(100.0f, notTarget._usableEnergy));
    EXPECT_NE(CellEvent_Attacked, notTarget._event);
}

TEST_F(AttackerTests, sensorTargeting_nearbyCreaturesLimitedToNearest)
{
    auto data = createAttacker({100.5f, 100.5f});
    data.addCreature({ObjectDesc().id(100).pos({101.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(2));

    // Creatures sorted by their distance to the match
    data.addCreature({ObjectDesc().id(200).pos({102.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(3));
    data.addCreature({ObjectDesc().id(300).pos({99.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(4));
    data.addCreature({ObjectDesc().id(400).pos({99.5f, 101.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(5));
    data.addCreature({ObjectDesc().id(500).pos({98.5f, 100.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(6));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    for (auto id : {100, 200, 300, 400}) {
        EXPECT_TRUE(actualData.getObjectRef(id).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    }
    EXPECT_TRUE(approxCompare(100.0f, actualData.getObjectRef(500).getCellRef()._usableEnergy));
}

TEST_F(AttackerTests, sensorTargeting_nearbyCreaturesRestrictedToColors)
{
    auto data = createAttacker({100.5f, 100.5f}, 0.0f, 0, 1 << 1);
    data.addCreature({ObjectDesc().id(100).pos({101.5f, 102.5f}).color(1).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(2));
    data.addCreature({ObjectDesc().id(200).pos({102.5f, 102.5f}).color(2).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(3));
    data.addCreature({ObjectDesc().id(300).pos({99.5f, 102.5f}).color(1).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(4));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_TRUE(actualData.getObjectRef(100).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(100.0f, actualData.getObjectRef(200).getCellRef()._usableEnergy));
    EXPECT_TRUE(actualData.getObjectRef(300).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, sensorTargeting_relocation_tracksMatchAndDetectsNearbyCreaturesAgain)
{
    // Negative signal in channel #0 triggers the sensor with relocation
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.5f, 100.5f}).type(createAttackerCell()),
            ObjectDesc()
                .id(2)
                .pos({101.5f, 100.5f})
                .type(createSensorCell(SensorDesc().autoTrigger(false)).neuralNetwork(NeuralNetDesc().bias(0, -1.0f).connectionWeight(0, 0.0f))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addCreature({ObjectDesc().id(100).pos({101.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(2));
    data.addCreature({ObjectDesc().id(200).pos({102.5f, 102.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(3));
    data.addCreature({ObjectDesc().id(300).pos({100.5f, 97.5f}).type(CellDesc().usableEnergy(100.0f))}, CreatureDesc().id(4));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_TRUE(actualData.getObjectRef(100).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(actualData.getObjectRef(200).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(100.0f, actualData.getObjectRef(300).getCellRef()._usableEnergy));

    // Creature 3 is now closer to the sensor than creature 2
    actualData.getObjectRef(100)._pos = {101.5f, 97.5f};
    actualData.getObjectRef(1).getCellRef()._rawEnergy = 0.0f;
    for (auto id : {100, 200, 300}) {
        actualData.getObjectRef(id).getCellRef()._usableEnergy = 100.0f;
    }
    _simulationFacade->setSimulationData(actualData);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_TRUE(actualData.getObjectRef(100).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
    EXPECT_TRUE(approxCompare(100.0f, actualData.getObjectRef(200).getCellRef()._usableEnergy));
    EXPECT_TRUE(actualData.getObjectRef(300).getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, sensorTargeting_matchingColor)
{
    // Create attacker with sensor restricted to color 1
    auto data = createAttacker({100.0f, 100.0f}, 0.0f, 0, 1 << 1);
    data.add(createTargetCreature({100.0f, 103.0f}, 2, 1), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, sensorTargeting_mismatchingColorOfTargetCell)
{
    // Create attacker with sensor restricted to color 1
    auto data = createAttacker({100.0f, 100.0f}, 0.0f, 0, 1 << 1);

    // The sensor detects the target creature by its cell with color 1
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).color(1).type(CellDesc().usableEnergy(100.0f)),
            ObjectDesc().id(101).pos({101.0f, 103.0f}).color(0).type(CellDesc().usableEnergy(100.0f)),
        },
        CreatureDesc().id(2));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();

    // Only the cell with the color of the sensor restriction is attacked, the energy flow within the target drains the other cell as well
    EXPECT_EQ(CellEvent_Attacked, actualData.getObjectRef(100).getCellRef()._event);
    EXPECT_NE(CellEvent_Attacked, actualData.getObjectRef(101).getCellRef()._event);
}

TEST_F(AttackerTests, sensorTargeting_colorRestrictionsOfSensorsAreMerged)
{
    // Both sensors detect the target creature, each by the cell with its color
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(createSensorCell(SensorDesc().mode(DetectCreatureDesc().restrictToColors(1 << 1)))),
            ObjectDesc().id(3).pos({99.0f, 100.0f}).type(createSensorCell(SensorDesc().mode(DetectCreatureDesc().restrictToColors(1 << 0)))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addCreature(
        {
            ObjectDesc().id(100).pos({100.0f, 103.0f}).color(1).type(CellDesc().usableEnergy(100.0f)),
            ObjectDesc().id(101).pos({101.0f, 103.0f}).color(0).type(CellDesc().usableEnergy(100.0f)),
        },
        CreatureDesc().id(2));
    data.addConnection(100, 101);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 3));
    EXPECT_EQ(CellEvent_Attacked, actualData.getObjectRef(100).getCellRef()._event);
    EXPECT_EQ(CellEvent_Attacked, actualData.getObjectRef(101).getCellRef()._event);
}

/**
 * Test: AttackerMode_FreeCell tests
 * The attacker in FreeCell mode should only attack free cells (not part of a creature)
 */
TEST_F(AttackerTests, freeCellMode_attackFreeCell)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode (single cell is sufficient)
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc())).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell (not part of a creature) - using FreeCellDesc
    data.addObjects(
    {
        ObjectDesc().id(100).pos({100.0f, 103.0f}).type(FreeCellDesc()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should be attacked in FreeCell mode
    EXPECT_TRUE(actualTarget.getFreeCellRef()._energy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, freeCellMode_attackFreeCell_matchingColor)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode with color restriction to color 1 (single cell is sufficient)
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc()
            .id(1)
            .pos({100.0f, 100.0f})
            .color(0)
            .type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc().restrictToColors(1 << 1))).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell with matching color (color 1)
    data.addObjects(
    {
        ObjectDesc().id(100).pos({100.0f, 103.0f}).color(1).type(FreeCellDesc()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should be attacked because color matches restriction
    EXPECT_TRUE(actualTarget.getFreeCellRef()._energy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, freeCellMode_attackFreeCell_mismatchingColor)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode with color restriction to color 1 (single cell is sufficient)
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc()
            .id(1)
            .pos({100.0f, 100.0f})
            .color(0)
            .type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc().restrictToColors(1 << 1))).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell with non-matching color (color 0)
    data.addObjects(
    {
        ObjectDesc().id(100).pos({100.0f, 103.0f}).color(0).type(FreeCellDesc()),
    });

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should NOT be attacked because color does not match restriction
    EXPECT_TRUE(approxCompare(origTarget.getFreeCellRef()._energy, actualTarget.getFreeCellRef()._energy));
}

TEST_F(AttackerTests, freeCellMode_doesNotAttackCreature)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in FreeCell mode (single cell is sufficient)
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackFreeCellDesc())).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add target creature (cells that are part of a creature, not free cells)
    data.add(createTargetCreature({100.0f, 103.0f}), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Creature should NOT be attacked in FreeCell mode
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: tagForAttackers = true (default)
 * The attacker should attack the creatures detected by the sensor when tagForAttackers is true
 */
TEST_F(AttackerTests, sensorTargeting_tagForAttackers_true)
{
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(createSensorCell(SensorDesc().tagForAttackers(true))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);

    // Add target creature with matching creatureId
    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should be attacked because tagForAttackers is true
    EXPECT_TRUE(actualTarget.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

/**
 * Test: tagForAttackers = false
 * The attacker should NOT attack the creatures detected by the sensor when tagForAttackers is false
 */
TEST_F(AttackerTests, sensorTargeting_tagForAttackers_false)
{
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(createSensorCell(SensorDesc().tagForAttackers(false))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);

    data.add(createTargetCreature({100.0f, 103.0f}, 2), false);

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Target should NOT be attacked because tagForAttackers is false
    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_TRUE(approxCompare(origTarget.getCellRef()._usableEnergy, actualTarget.getCellRef()._usableEnergy));
}

/**
 * Test: Multiple sensors, only one tagged for attackers
 * When one sensor has tagForAttackers = true and another has tagForAttackers = false,
 * only the creatures detected by the tagged sensor should be attacked
 */
TEST_F(AttackerTests, sensorTargeting_tagForAttackers_mixedSensors)
{
    // Each sensor detects the creature nearest to it
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.5f, 100.5f}).type(createAttackerCell()),
            ObjectDesc().id(2).pos({101.5f, 100.5f}).type(createSensorCell(SensorDesc().tagForAttackers(false))),
            ObjectDesc().id(3).pos({99.5f, 100.5f}).type(createSensorCell(SensorDesc().tagForAttackers(true))),
        },
        CreatureDesc().id(1));
    data.addConnection(1, 2);
    data.addConnection(1, 3);

    // Add creature 2 (should NOT be attacked - its sensor is not tagged)
    data.add(createTargetCreature({101.5f, 103.5f}, 2), false);

    // Add creature 4 (should be attacked - its sensor IS tagged)
    data.addCreature(
        {
            ObjectDesc().id(200).pos({99.5f, 97.5f}).type(CellDesc().usableEnergy(100.0f)),
        },
        CreatureDesc().id(4));

    auto origTarget1 = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget1 = actualData.getObjectRef(100);
    auto actualTarget2 = actualData.getObjectRef(200);

    EXPECT_EQ(std::optional(2), getLastMatchedCreatureIdPart(actualData, 2));
    EXPECT_EQ(std::optional(4), getLastMatchedCreatureIdPart(actualData, 3));

    // Creature 2 should NOT be attacked (sensor for creature 2 has tagForAttackers = false)
    EXPECT_TRUE(approxCompare(origTarget1.getCellRef()._usableEnergy, actualTarget1.getCellRef()._usableEnergy));

    // Creature 4 should be attacked (sensor for creature 4 has tagForAttackers = true)
    EXPECT_TRUE(actualTarget2.getCellRef()._usableEnergy < 100.0f - NEAR_ZERO);
}

TEST_F(AttackerTests, creatureMode_doesNotAttackFreeCell)
{
    // Create a neural net with a bias on Channels::CellTypeActivation to trigger the attacker
    NeuralNetDesc nn;
    nn._biases[Channels::CellTypeActivation] = 1.0f;

    // Create attacker creature in Creature mode (single cell is sufficient)
    auto data = ContentDesc().addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().cellType(AttackerDesc().mode(AttackCreatureDesc())).neuralNetwork(nn)),
        },
        CreatureDesc().id(1));

    // Add a free cell (not part of a creature)
    data.addObjects(
    {
        ObjectDesc().id(100).pos({100.0f, 103.0f}).type(FreeCellDesc()),
    });

    auto origTarget = data.getObjectRef(100);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(TIMESTEPS_PER_CELL_FUNCTION);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualTarget = actualData.getObjectRef(100);

    // Free cell should NOT be attacked in Creature mode
    EXPECT_TRUE(approxCompare(origTarget.getFreeCellRef()._energy, actualTarget.getFreeCellRef()._energy));
}