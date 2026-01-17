#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SelectionShallowData.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class EditTests : public IntegrationTestFramework
{
public:
    EditTests()
        : IntegrationTestFramework({100, 100})
    {}
    virtual ~EditTests() = default;
};

TEST_F(EditTests, getSelectionShallowData_noSelection)
{
    auto data = Description().addCreature(
        {
            ObjectDescription().id(1).pos({50, 50}),
            ObjectDescription().id(2).pos({51, 50}),
        },
        CreatureDescription());
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->removeSelection();
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(0, selectionData.numObjects);
    EXPECT_EQ(0, selectionData.numCreatures);
    EXPECT_EQ(0, selectionData.numClusterCells);
    EXPECT_EQ(0, selectionData.numEnergyParticles);
}

TEST_F(EditTests, getSelectionShallowData_selectCells)
{
    auto data = Description().addCreature(
        {
            ObjectDescription().id(1).pos({50, 50}),
            ObjectDescription().id(2).pos({51, 50}),
            ObjectDescription().id(3).pos({52, 50}),
        },
        CreatureDescription());
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {51.5f, 51});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(2, selectionData.numObjects);
    EXPECT_EQ(1, selectionData.numCreatures);
    EXPECT_EQ(3, selectionData.numClusterCells);
    EXPECT_EQ(0, selectionData.numEnergyParticles);
}

TEST_F(EditTests, getSelectionShallowData_selectCreatures)
{
    auto data = Description()
                    .addCreature(
                        {
                            ObjectDescription().id(1).pos({50, 50}),
                            ObjectDescription().id(2).pos({51, 50}),
                        },
                        CreatureDescription())
                    .addCreature(
                        {
                            ObjectDescription().id(3).pos({60, 50}),
                        },
                        CreatureDescription());
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {52, 51});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(2, selectionData.numObjects);
    EXPECT_EQ(1, selectionData.numCreatures);
    EXPECT_EQ(2, selectionData.numClusterCells);
    EXPECT_EQ(0, selectionData.numEnergyParticles);
}

TEST_F(EditTests, getSelectionShallowData_selectParticles)
{
    auto data = Description().energies({
        EnergyDescription().id(1).pos({50, 50}).energy(10.0f),
        EnergyDescription().id(2).pos({51, 50}).energy(10.0f),
        EnergyDescription().id(3).pos({70, 50}).energy(10.0f),
    });
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {52, 51});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(0, selectionData.numObjects);
    EXPECT_EQ(0, selectionData.numCreatures);
    EXPECT_EQ(0, selectionData.numClusterCells);
    EXPECT_EQ(2, selectionData.numEnergyParticles);
}

TEST_F(EditTests, getSelectionShallowData_selectMixed)
{
    auto data = Description()
                    .addObjects({
                        ObjectDescription().id(1).pos({50, 50}).type(StructureDescription()),
                        ObjectDescription().id(2).pos({51, 50}).type(StructureDescription()),
                    })
                    .energies({
                        EnergyDescription().id(3).pos({52, 50}).energy(10.0f),
                    })
                    .addCreature(
                        {
                            ObjectDescription().id(4).pos({53, 50}),
                            ObjectDescription().id(5).pos({54, 50}),
                            ObjectDescription().id(6).pos({55, 50}),
                        },
                        CreatureDescription());
    data.addConnection(1, 2);
    data.addConnection(4, 5);
    data.addConnection(5, 6);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {56, 51});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(5, selectionData.numObjects);
    EXPECT_EQ(1, selectionData.numCreatures);
    EXPECT_EQ(5, selectionData.numClusterCells);
    EXPECT_EQ(1, selectionData.numEnergyParticles);
}

TEST_F(EditTests, getSelectionShallowData_selectMultipleCreatures)
{
    auto data = Description()
                    .addCreature(
                        {
                            ObjectDescription().id(1).pos({50, 50}),
                            ObjectDescription().id(2).pos({51, 50}),
                        },
                        CreatureDescription())
                    .addCreature(
                        {
                            ObjectDescription().id(3).pos({52, 50}),
                        },
                        CreatureDescription())
                    .addCreature(
                        {
                            ObjectDescription().id(4).pos({70, 70}),
                        },
                        CreatureDescription());
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {53, 51});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(3, selectionData.numObjects);
    EXPECT_EQ(2, selectionData.numCreatures);
    EXPECT_EQ(3, selectionData.numClusterCells);
    EXPECT_EQ(0, selectionData.numEnergyParticles);
}
