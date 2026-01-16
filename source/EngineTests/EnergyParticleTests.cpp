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
    data._energies.emplace_back(EnergyDescription()
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
    EXPECT_EQ(0, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    // Verify the cell has approximately the same energy as the original particle
    if (!actualData._objects.empty()) {
        auto const& object = actualData._objects.at(0);
        EXPECT_TRUE(approxCompare(normalCellEnergy + 10.0f, std::get<CellDescription>(object._type)._usableEnergy, 1.0f));
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
    Description data;
    data._energies.emplace_back(EnergyDescription()
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
    Description data;
    data._energies.emplace_back(EnergyDescription()
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
    EXPECT_EQ(1, actualData._energies.size());
    EXPECT_EQ(0, actualData._objects.size());
}

TEST_F(EnergyParticleTests, particleAbsorption)
{
    auto cellEnergy = _parameters.normalCellEnergy.value[0];
    auto particleEnergy = 10.0f;

    auto data = Description()
                    .objects({ObjectDescription().id(1).pos({100.4f, 100.4f}).color(0).type(CellDescription().usableEnergy(cellEnergy))})
                    .energies({EnergyDescription().pos({100.4f, 100.4f}).energy(particleEnergy)});

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    EXPECT_EQ(0, actualData._energies.size());
    EXPECT_EQ(1, actualData._objects.size());

    auto const& object = actualData.getObjectRef(1);
    EXPECT_TRUE(approxCompare(cellEnergy, std::get<CellDescription>(object._type)._usableEnergy));
    EXPECT_TRUE(approxCompare(particleEnergy, std::get<CellDescription>(object._type)._rawEnergy));
}

TEST_F(EnergyParticleTests, cellToParticle_belowMinEnergy)
{
    _parameters.cellDeathProbability.baseValue[0] = 1.0f;  // Ensure cell will die instantly when below min energy   
    _simulationFacade->setSimulationParameters(_parameters);

    auto cellEnergy = _parameters.minCellEnergy.baseValue[0] / 2;
    auto depotEnergy = 100.0f;

    auto data = Description().objects(
        {ObjectDescription().id(1).pos({100.4f, 100.4f}).color(0).type(CellDescription().usableEnergy(cellEnergy).cellType(DepotDescription().storedUsableEnergy(depotEnergy)))});

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
