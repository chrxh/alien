#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
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

TEST_F(EnergyParticleTests, particleToCell_transformationAllowed)
{
    // Enable particle transformation
    _parameters.particleTransformationAllowed.value = true;
    _simulationFacade->setSimulationParameters(_parameters);

    // Get the normal cell energy for color 0
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Create a particle with energy above normalCellEnergy
    Description data;
    data._particles.emplace_back(ParticleDescription()
                                      .id(1)
                                      .pos({100.0f, 100.0f})
                                      .vel({0.1f, 0.1f})
                                      .energy(normalCellEnergy + 10.0f)
                                      .color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for several timesteps to allow transformation
    _simulationFacade->calcTimesteps(10);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify that the particle was transformed into a cell
    EXPECT_EQ(0, actualData._particles.size());
    EXPECT_EQ(1, actualData._cells.size());

    // Verify the cell has approximately the same energy as the original particle
    if (!actualData._cells.empty()) {
        auto const& cell = actualData._cells.at(0);
        EXPECT_TRUE(approxCompare(normalCellEnergy + 10.0f, cell._energy, 1.0f));
        EXPECT_EQ(0, cell._color);
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
    Description data;
    data._particles.emplace_back(ParticleDescription()
                                      .id(1)
                                      .pos({100.0f, 100.0f})
                                      .vel({0.1f, 0.1f})
                                      .energy(normalCellEnergy + 10.0f)
                                      .color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for several timesteps
    _simulationFacade->calcTimesteps(10);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify that the particle was NOT transformed (remains a particle)
    EXPECT_EQ(1, actualData._particles.size());
    EXPECT_EQ(0, actualData._cells.size());
}

TEST_F(EnergyParticleTests, particleToCell_insufficientEnergy)
{
    // Enable particle transformation
    _parameters.particleTransformationAllowed.value = true;
    _simulationFacade->setSimulationParameters(_parameters);

    // Get the normal cell energy for color 0
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    // Create a particle with energy below normalCellEnergy
    Description data;
    data._particles.emplace_back(ParticleDescription()
                                      .id(1)
                                      .pos({100.0f, 100.0f})
                                      .vel({0.1f, 0.1f})
                                      .energy(normalCellEnergy - 1.0f)
                                      .color(0));

    _simulationFacade->setSimulationData(data);

    // Run simulation for several timesteps
    _simulationFacade->calcTimesteps(10);

    auto actualData = _simulationFacade->getSimulationData();

    // Verify that the particle was NOT transformed (insufficient energy)
    EXPECT_EQ(1, actualData._particles.size());
    EXPECT_EQ(0, actualData._cells.size());
}
