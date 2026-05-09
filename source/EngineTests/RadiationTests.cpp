#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class RadiationTests : public IntegrationTestFramework
{
public:
    RadiationTests()
        : IntegrationTestFramework()
    {
        _parameters.friction.baseValue = 0;
        _parameters.innerFriction.value = 0;

        // Set a very high radiation rate for all colors as requested
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0.1f;  // Very high radiation rate
            _parameters.radiationType1_minimumAge.value[i] = 0;       // No minimum age requirement
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~RadiationTests() = default;
};

TEST_F(RadiationTests, fixedCells_shouldNotRadiate)
{
    auto initialEnergy = 200.0f;

    auto data =
        Desc().addCreature({ObjectDesc().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).fixed(true).color(0).type(CellDesc().usableEnergy(initialEnergy))});

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - fixed cells should not lose energy to radiation
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify no particles were created (no radiation emitted)
    EXPECT_EQ(0, actualData._energies.size());

    // Verify the fixed cell retained its energy
    EXPECT_EQ(1, actualData._objects.size());
    auto const& object = actualData._objects.at(0);
    EXPECT_TRUE(approxCompare(initialEnergy, object.getCellRef()._usableEnergy));
}

TEST_F(RadiationTests, solid_shouldNotRadiate)
{
    Desc data;
    data._objects.emplace_back(ObjectDesc().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).color(0).type(SolidDesc()));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - solid cells should not lose energy to radiation
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify no particles were created (no radiation emitted)
    EXPECT_EQ(0, actualData._energies.size());

    // Verify the solid cell is still present
    EXPECT_EQ(1, actualData._objects.size());
    auto const& object = actualData._objects.at(0);
    EXPECT_EQ(ObjectType_Solid, object.getObjectType());
}

TEST_F(RadiationTests, baseCells_shouldRadiate)
{
    auto initialEnergy = 200.0f;

    auto data = Desc().addCreature(
        {ObjectDesc().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).color(0).type(CellDesc().usableEnergy(initialEnergy).cellType(BaseDesc()))});

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - base cells should radiate
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // With high radiation rate, energy particles should have been created
    EXPECT_FALSE(actualData._energies.empty());

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(RadiationTests, freeCells_shouldRadiate)
{
    Desc data;
    data._objects.emplace_back(ObjectDesc().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).color(0).type(FreeCellDesc()));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - free cells should radiate
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // With high radiation rate, energy particles should have been created
    EXPECT_FALSE(actualData._energies.empty());

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(RadiationTests, constructorCells_shouldRadiate)
{
    auto initialEnergy = 200.0f;

    auto data = Desc().addCreature(
        {ObjectDesc().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).color(0).type(CellDesc().usableEnergy(initialEnergy).constructor(ConstructorDesc()))});

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - constructor cells should radiate
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // With high radiation rate, energy particles should have been created
    EXPECT_FALSE(actualData._energies.empty());

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
