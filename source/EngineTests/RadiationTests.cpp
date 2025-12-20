#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
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
            _parameters.radiationType1_strength.baseValue[i] = 0.01f;  // Very high radiation rate
            _parameters.radiationType1_minimumAge.value[i] = 0;        // No minimum age requirement
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~RadiationTests() = default;
};

TEST_F(RadiationTests, fixedCells_shouldNotRadiate)
{
    auto initialEnergy = 200.0f;

    Description data;
    data._cells.emplace_back(CellDescription().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).usableEnergy(initialEnergy).fixed(true).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - fixed cells should not lose energy to radiation
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify no particles were created (no radiation emitted)
    EXPECT_EQ(0, actualData._particles.size());

    // Verify the fixed cell retained its energy
    EXPECT_EQ(1, actualData._cells.size());
    auto const& cell = actualData._cells.at(0);
    EXPECT_TRUE(approxCompare(initialEnergy, cell._usableEnergy, 0.01f));
}

TEST_F(RadiationTests, structureCells_shouldNotRadiate)
{
    auto initialEnergy = 200.0f;

    Description data;
    data._cells.emplace_back(
        CellDescription().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).usableEnergy(initialEnergy).cellType(StructureCellDescription()).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - structure cells should not lose energy to radiation
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify no particles were created (no radiation emitted)
    EXPECT_EQ(0, actualData._particles.size());

    // Verify the structure cell retained its energy
    EXPECT_EQ(1, actualData._cells.size());
    auto const& cell = actualData._cells.at(0);
    EXPECT_TRUE(approxCompare(initialEnergy, cell._usableEnergy, 0.01f));
}

TEST_F(RadiationTests, baseCells_shouldRadiate)
{
    auto initialEnergy = 200.0f;

    Description data;
    data._cells.emplace_back(CellDescription().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).usableEnergy(initialEnergy).cellType(BaseDescription()).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - base cells should radiate
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // With high radiation rate, energy particles should have been created
    EXPECT_GT(actualData._particles.size(), 0);

    // Verify the base cell lost some energy due to radiation
    EXPECT_EQ(1, actualData._cells.size());
    auto const& cell = actualData._cells.at(0);
    EXPECT_LT(cell._usableEnergy, initialEnergy);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(RadiationTests, freeCells_shouldRadiate)
{
    auto initialEnergy = 200.0f;

    Description data;
    data._cells.emplace_back(
        CellDescription().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).usableEnergy(initialEnergy).cellType(FreeCellDescription()).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - free cells should radiate
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // With high radiation rate, energy particles should have been created
    EXPECT_GT(actualData._particles.size(), 0);

    // Verify the free cell lost some energy due to radiation
    EXPECT_EQ(1, actualData._cells.size());
    auto const& cell = actualData._cells.at(0);
    EXPECT_LT(cell._usableEnergy, initialEnergy);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(RadiationTests, constructorCells_shouldRadiate)
{
    auto initialEnergy = 200.0f;

    Description data;
    data._cells.emplace_back(
        CellDescription().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).usableEnergy(initialEnergy).cellType(ConstructorDescription()).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps - constructor cells should radiate
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // With high radiation rate, energy particles should have been created
    EXPECT_GT(actualData._particles.size(), 0);

    // Verify the constructor cell lost some energy due to radiation
    EXPECT_EQ(1, actualData._cells.size());
    auto const& cell = actualData._cells.at(0);
    EXPECT_LT(cell._usableEnergy, initialEnergy);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(RadiationTests, fixedStructureCells_shouldNotRadiate)
{
    // Test that a cell that is both fixed AND a structure cell does not radiate
    auto initialEnergy = 200.0f;

    Description data;
    data._cells.emplace_back(
        CellDescription().id(1).pos({100.0f, 100.0f}).vel({0.0f, 0.0f}).usableEnergy(initialEnergy).cellType(StructureCellDescription()).fixed(true).color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify no particles were created (no radiation emitted)
    EXPECT_EQ(0, actualData._particles.size());

    // Verify the cell retained its energy
    EXPECT_EQ(1, actualData._cells.size());
    auto const& cell = actualData._cells.at(0);
    EXPECT_TRUE(approxCompare(initialEnergy, cell._usableEnergy, 0.01f));
}

TEST_F(RadiationTests, rawEnergy_shouldAlsoRadiate)
{
    // Test that raw energy also radiates for non-fixed, non-structure cells
    auto initialUsableEnergy = 200.0f;
    auto initialRawEnergy = 100.0f;

    Description data;
    data._cells.emplace_back(CellDescription()
                                 .id(1)
                                 .pos({100.0f, 100.0f})
                                 .vel({0.0f, 0.0f})
                                 .usableEnergy(initialUsableEnergy)
                                 .rawEnergy(initialRawEnergy)
                                 .cellType(BaseDescription())
                                 .color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for many timesteps
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify particles were created
    EXPECT_GT(actualData._particles.size(), 0);

    // Verify the cell lost energy (both usable and raw)
    EXPECT_EQ(1, actualData._cells.size());
    auto const& cell = actualData._cells.at(0);
    EXPECT_LT(cell._usableEnergy + cell._rawEnergy, initialUsableEnergy + initialRawEnergy);

    // Total energy should be conserved
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
