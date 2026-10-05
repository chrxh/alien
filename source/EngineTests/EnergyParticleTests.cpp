#include <gtest/gtest.h>

#include <Data/DescEditService.h>
#include <Data/Descs.h>

#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class EnergyParticleTests : public IntegrationTestFramework
{
public:
    EnergyParticleTests()
        : IntegrationTestFramework()
    {
        _parameters.friction.baseValue = 0;
        _parameters.innerFriction.value = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~EnergyParticleTests() = default;
};

TEST_F(EnergyParticleTests, particleToFreeCell_transformationAllowed)
{
    // Enable particle transformation
    _parameters.particleTransformationAllowed.value = true;
    _simulationFacade->setSimulationParameters(_parameters);

    // Get the normal cell energy for color 0
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Create a particle with energy above normalCellEnergy
    ContentDesc data;
    data._energies.emplace_back(EnergyDesc().id(1).pos({100.0f, 100.0f}).vel({0.1f, 0.1f}).energy(normalCellEnergy + 10.0f).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for several timesteps to allow transformation
    _simulationFacade->calcTimesteps(10);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify that the particle was transformed into a free cell
    EXPECT_EQ(0, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    // Verify the free cell has approximately the same energy as the original particle
    if (!actualData._objects.empty()) {
        auto const& object = actualData._objects.at(0);
        EXPECT_EQ(ObjectType_FreeCell, object.getObjectType());
        EXPECT_TRUE(approxCompare(normalCellEnergy + 10.0f, object.getFreeCellRef()._energy, 1.0f));
        EXPECT_EQ(0, object._color);
    }
}

TEST_F(EnergyParticleTests, particleToCell_transformationDisabled)
{
    // Disable particle transformation
    _parameters.particleTransformationAllowed.value = false;
    _simulationFacade->setSimulationParameters(_parameters);

    // Get the normal cell energy for color 0
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Create a particle with energy above normalCellEnergy
    ContentDesc data;
    data._energies.emplace_back(EnergyDesc().id(1).pos({100.0f, 100.0f}).vel({0.1f, 0.1f}).energy(normalCellEnergy + 10.0f).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for several timesteps
    _simulationFacade->calcTimesteps(10);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify that the particle was NOT transformed (remains a particle)
    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(0, actualData._objects.size());
}

TEST_F(EnergyParticleTests, particleToCell_insufficientEnergy)
{
    // Enable particle transformation
    _parameters.particleTransformationAllowed.value = true;
    _simulationFacade->setSimulationParameters(_parameters);

    // Get the normal cell energy for color 0
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Create a particle with energy below normalCellEnergy
    ContentDesc data;
    data._energies.emplace_back(EnergyDesc().id(1).pos({100.0f, 100.0f}).vel({0.1f, 0.1f}).energy(normalCellEnergy - 1.0f).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for several timesteps
    _simulationFacade->calcTimesteps(10);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify that the particle was NOT transformed (insufficient energy)
    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(0, actualData._objects.size());
}

TEST_F(EnergyParticleTests, particleAbsorptionForCells)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];
    auto particleEnergy = 10.0f;

    auto data = ContentDesc()
                    .addCreature({ObjectDesc().pos({100.4f, 100.4f}).color(0).type(CellDesc().usableEnergy(cellEnergy))})
                    .energies({EnergyDesc().pos({100.4f, 100.4f}).energy(particleEnergy)});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    EXPECT_EQ(0, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    auto const& object = actualData._objects.at(0);
    EXPECT_TRUE(approxCompare(cellEnergy, object.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(particleEnergy, object.getCellRef()._rawEnergy));
}

TEST_F(EnergyParticleTests, noParticleAbsorptionForCellsUnderConstruction)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];
    auto particleEnergy = 10.0f;

    auto data = ContentDesc()
                    .addCreature({ObjectDesc().pos({100.4f, 100.4f}).color(0).type(CellDesc().usableEnergy(cellEnergy).cellState(CellState_UnderConstruction))})
                    .energies({EnergyDesc().pos({100.4f, 100.4f}).energy(particleEnergy)});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    auto const& object = actualData._objects.at(0);
    EXPECT_TRUE(approxCompare(0.0f, object.getCellRef()._rawEnergy));
}

TEST_F(EnergyParticleTests, particleAbsorptionForFreeCells)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];
    auto particleEnergy = 10.0f;

    auto data = ContentDesc()
                    .addObjects({ObjectDesc().pos({100.4f, 100.4f}).color(0).type(FreeCellDesc().energy(cellEnergy))})
                    .energies({EnergyDesc().pos({100.4f, 100.4f}).energy(particleEnergy)});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    EXPECT_EQ(0, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    auto const& object = actualData._objects.at(0);
    EXPECT_TRUE(approxCompare(cellEnergy + particleEnergy, object.getFreeCellRef()._energy));
}

TEST_F(EnergyParticleTests, particleAbsorptionForStaticDigestorCells)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];
    auto particleEnergy = 10.0f;

    auto digestor = DigestorDesc().rawEnergyConductivity(Const::DigestorRawEnergyConductivity_Max);
    auto data = ContentDesc()
                    .addCreature({ObjectDesc().pos({100.4f, 100.4f}).color(0).isStatic(true).type(CellDesc().usableEnergy(cellEnergy).cellType(digestor))})
                    .energies({EnergyDesc().pos({100.4f, 100.4f}).energy(particleEnergy)});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    EXPECT_EQ(0, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    auto const& object = actualData._objects.at(0);
    EXPECT_TRUE(approxCompare(cellEnergy, object.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(particleEnergy, object.getCellRef()._rawEnergy));
}

TEST_F(EnergyParticleTests, noParticleAbsorptionForStaticNonDigestorCells)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];
    auto particleEnergy = 10.0f;

    auto data = ContentDesc()
                    .addCreature({ObjectDesc().pos({100.4f, 100.4f}).color(0).isStatic(true).type(CellDesc().usableEnergy(cellEnergy))})
                    .energies({EnergyDesc().pos({100.4f, 100.4f}).energy(particleEnergy)});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    auto const& object = actualData._objects.at(0);
    EXPECT_TRUE(approxCompare(0.0f, object.getCellRef()._rawEnergy));
}

TEST_F(EnergyParticleTests, noParticleAbsorptionForStaticFreeCells)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];
    auto particleEnergy = 10.0f;

    auto data = ContentDesc()
                    .addObjects({ObjectDesc().pos({100.4f, 100.4f}).color(0).isStatic(true).type(FreeCellDesc().energy(cellEnergy))})
                    .energies({EnergyDesc().pos({100.4f, 100.4f}).energy(particleEnergy)});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    auto const& object = actualData._objects.at(0);
    EXPECT_TRUE(approxCompare(cellEnergy, object.getFreeCellRef()._energy));
}

TEST_F(EnergyParticleTests, particleBouncesOffConnectionBetweenSolids_fromLeft)
{
    auto data = ContentDesc()
                    .addObjects({ObjectDesc().id(1).pos({100.0f, 98.5f}).type(SolidDesc()), ObjectDesc().id(2).pos({100.0f, 101.5f}).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({98.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(99.49f, particle._pos.x, 0.01f);
    EXPECT_NEAR(100.0f, particle._pos.y, 0.01f);
    EXPECT_NEAR(-1.0f, particle._vel.x, 0.001f);
    EXPECT_NEAR(0.0f, particle._vel.y, 0.001f);
}

TEST_F(EnergyParticleTests, particleBouncesOffConnectionBetweenSolids_fromRight)
{
    auto data = ContentDesc()
                    .addObjects({ObjectDesc().id(1).pos({100.0f, 98.5f}).type(SolidDesc()), ObjectDesc().id(2).pos({100.0f, 101.5f}).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({101.5f, 100.0f}).vel({-1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(100.51f, particle._pos.x, 0.01f);
    EXPECT_NEAR(100.0f, particle._pos.y, 0.01f);
    EXPECT_NEAR(1.0f, particle._vel.x, 0.001f);
    EXPECT_NEAR(0.0f, particle._vel.y, 0.001f);
}

TEST_F(EnergyParticleTests, particleBouncesOffConnectionBetweenStaticSolids)
{
    auto data = ContentDesc()
                    .addObjects(
                        {ObjectDesc().id(1).pos({100.0f, 98.5f}).isStatic(true).type(SolidDesc()),
                         ObjectDesc().id(2).pos({100.0f, 101.5f}).isStatic(true).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({98.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(99.49f, particle._pos.x, 0.01f);
    EXPECT_NEAR(-1.0f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, particleBouncesOffConnectionAtWorldBoundary_fromLeft)
{
    auto data = ContentDesc()
                    .addObjects({ObjectDesc().id(1).pos({0.0f, 98.5f}).type(SolidDesc()), ObjectDesc().id(2).pos({0.0f, 101.5f}).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({998.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(999.49f, particle._pos.x, 0.01f);
    EXPECT_NEAR(100.0f, particle._pos.y, 0.01f);
    EXPECT_NEAR(-1.0f, particle._vel.x, 0.001f);
    EXPECT_NEAR(0.0f, particle._vel.y, 0.001f);
}

TEST_F(EnergyParticleTests, particleBouncesOffConnectionAtWorldBoundary_fromRight)
{
    auto data = ContentDesc()
                    .addObjects({ObjectDesc().id(1).pos({0.0f, 98.5f}).type(SolidDesc()), ObjectDesc().id(2).pos({0.0f, 101.5f}).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({1.5f, 100.0f}).vel({-1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(0.51f, particle._pos.x, 0.01f);
    EXPECT_NEAR(100.0f, particle._pos.y, 0.01f);
    EXPECT_NEAR(1.0f, particle._vel.x, 0.001f);
    EXPECT_NEAR(0.0f, particle._vel.y, 0.001f);
}

TEST_F(EnergyParticleTests, particleBouncesOffConnectionBetweenMovingNonStaticSolids)
{
    auto data = ContentDesc()
                    .addObjects(
                        {ObjectDesc().id(1).pos({100.0f, 98.5f}).vel({0.4f, 0.0f}).isStatic(false).type(SolidDesc()),
                         ObjectDesc().id(2).pos({100.0f, 101.5f}).vel({0.4f, 0.0f}).isStatic(false).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({98.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._objects.size());
    for (auto const& object : actualData._objects) {
        EXPECT_FALSE(object._isStatic);
        EXPECT_NEAR(101.2f, object._pos.x, 0.01f);
    }
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(100.89f, particle._pos.x, 0.01f);
    EXPECT_NEAR(-0.2f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, particleIsPushedByConnectionBetweenMovingNonStaticSolids)
{
    auto data = ContentDesc()
                    .addObjects(
                        {ObjectDesc().id(1).pos({99.8f, 98.5f}).vel({0.4f, 0.0f}).isStatic(false).type(SolidDesc()),
                         ObjectDesc().id(2).pos({99.8f, 101.5f}).vel({0.4f, 0.0f}).isStatic(false).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({100.5f, 100.0f}).vel({0.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(100.71f, particle._pos.x, 0.01f);
    EXPECT_NEAR(0.8f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, particleBouncesOffDiagonalConnection)
{
    auto data = ContentDesc()
                    .addObjects({ObjectDesc().id(1).pos({99.0f, 99.0f}).type(SolidDesc()), ObjectDesc().id(2).pos({101.0f, 101.0f}).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({98.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(99.993f, particle._pos.x, 0.01f);
    EXPECT_NEAR(100.507f, particle._pos.y, 0.01f);
    EXPECT_NEAR(0.0f, particle._vel.x, 0.001f);
    EXPECT_NEAR(1.0f, particle._vel.y, 0.001f);
}

TEST_F(EnergyParticleTests, fastParticleDoesNotTunnelThroughConnection)
{
    auto data = ContentDesc()
                    .addObjects({ObjectDesc().id(1).pos({100.0f, 98.5f}).type(SolidDesc()), ObjectDesc().id(2).pos({100.0f, 101.5f}).type(SolidDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({97.5f, 100.0f}).vel({3.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(99.49f, particle._pos.x, 0.01f);
    EXPECT_NEAR(-3.0f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, particlePassesUnconnectedSolid)
{
    auto data = ContentDesc()
                    .addObjects({ObjectDesc().id(1).pos({100.5f, 100.5f}).type(SolidDesc())})
                    .energies({EnergyDesc().pos({99.5f, 100.5f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(102.5f, particle._pos.x, 0.01f);
    EXPECT_NEAR(1.0f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, particlePassesConnectionBetweenStaticCells)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = ContentDesc()
                    .addCreature(
                        {ObjectDesc().id(1).pos({100.0f, 98.5f}).isStatic(true).type(CellDesc().usableEnergy(cellEnergy)),
                         ObjectDesc().id(2).pos({100.0f, 101.5f}).isStatic(true).type(CellDesc().usableEnergy(cellEnergy))})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({98.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(101.5f, particle._pos.x, 0.01f);
    EXPECT_NEAR(1.0f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, particlePassesConnectionBetweenSolidAndStaticFreeCell)
{
    auto data = ContentDesc()
                    .addObjects(
                        {ObjectDesc().id(1).pos({100.0f, 98.5f}).isStatic(true).type(SolidDesc()),
                         ObjectDesc().id(2).pos({100.0f, 101.5f}).isStatic(true).type(FreeCellDesc())})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({98.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(101.5f, particle._pos.x, 0.01f);
    EXPECT_NEAR(1.0f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, particlePassesConnectionBetweenNonStaticCells)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = ContentDesc()
                    .addCreature(
                        {ObjectDesc().id(1).pos({100.0f, 98.5f}).type(CellDesc().usableEnergy(cellEnergy)),
                         ObjectDesc().id(2).pos({100.0f, 101.5f}).type(CellDesc().usableEnergy(cellEnergy))})
                    .addConnection(1, 2)
                    .energies({EnergyDesc().pos({98.5f, 100.0f}).vel({1.0f, 0.0f}).energy(10.0f)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._energies.size());
    auto const& particle = actualData._energies.at(0);
    EXPECT_NEAR(101.5f, particle._pos.x, 0.01f);
    EXPECT_NEAR(1.0f, particle._vel.x, 0.001f);
}

TEST_F(EnergyParticleTests, cellToParticle_belowMinEnergy)
{
    _parameters.cellDeathProbability.baseValue[0] = 1.0f;  // Ensure cell will die instantly when below min energy
    _simulationFacade->setSimulationParameters(_parameters);

    auto cellEnergy = _parameters.minCellEnergy.baseValue[0] / 2;
    auto depotEnergy = 100.0f;

    auto data = ContentDesc().addCreature(
        {ObjectDesc().pos({100.4f, 100.4f}).color(0).type(CellDesc().usableEnergy(cellEnergy).cellType(DepotDesc().storedUsableEnergy(depotEnergy)))});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_EQ(0, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    _simulationFacade->calcTimesteps(1);
    actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(0, actualData._objects.size());
}

TEST_F(EnergyParticleTests, freeCellToParticle_belowMinEnergy)
{
    _parameters.cellDeathProbability.baseValue[0] = 1.0f;  // Ensure free cell will die instantly when below min energy
    _simulationFacade->setSimulationParameters(_parameters);

    auto freeCellEnergy = _parameters.minCellEnergy.baseValue[0] / 2;

    auto data = ContentDesc().addObjects({ObjectDesc().pos({100.4f, 100.4f}).color(0).type(FreeCellDesc().energy(freeCellEnergy))});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(2);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(0, actualData._objects.size());
}