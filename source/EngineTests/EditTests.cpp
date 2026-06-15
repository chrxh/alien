#include <algorithm>

#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
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
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({50, 50}),
        ObjectDesc().id(2).pos({51, 50}),
    });
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
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({50, 50}),
        ObjectDesc().id(2).pos({51, 50}),
        ObjectDesc().id(3).pos({52, 50}),
    });
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
    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({50, 50}),
                        ObjectDesc().id(2).pos({51, 50}),
                    })
                    .addCreature({
                        ObjectDesc().id(3).pos({60, 50}),
                    });
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
    auto data = Desc().energies({
        EnergyDesc().id(1).pos({50, 50}).energy(10.0f),
        EnergyDesc().id(2).pos({51, 50}).energy(10.0f),
        EnergyDesc().id(3).pos({70, 50}).energy(10.0f),
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
    auto data = Desc()
                    .addObjects({
                        ObjectDesc().id(1).pos({50, 50}).type(SolidDesc()),
                        ObjectDesc().id(2).pos({51, 50}).type(SolidDesc()),
                    })
                    .energies({
                        EnergyDesc().id(3).pos({52, 50}).energy(10.0f),
                    })
                    .addCreature({
                        ObjectDesc().id(4).pos({53, 50}),
                        ObjectDesc().id(5).pos({54, 50}),
                        ObjectDesc().id(6).pos({55, 50}),
                    });
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
    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({50, 50}),
                        ObjectDesc().id(2).pos({51, 50}),
                    })
                    .addCreature({
                        ObjectDesc().id(3).pos({52, 50}),
                    })
                    .addCreature({
                        ObjectDesc().id(4).pos({70, 70}),
                    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {53, 51});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(3, selectionData.numObjects);
    EXPECT_EQ(2, selectionData.numCreatures);
    EXPECT_EQ(3, selectionData.numClusterCells);
    EXPECT_EQ(0, selectionData.numEnergyParticles);
}

TEST_F(EditTests, getSelectionShallowData_onlyDirectlySelectedCreatures)
{
    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({50, 50}),
                        ObjectDesc().id(2).pos({51, 50}),
                    })
                    .addCreature({
                        ObjectDesc().id(3).pos({52, 50}),
                        ObjectDesc().id(4).pos({53, 50}),
                    });
    data.addConnection(1, 2);
    data.addConnection(3, 4);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {51.5f, 51});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_EQ(2, selectionData.numObjects);
    EXPECT_EQ(1, selectionData.numCreatures);
    EXPECT_EQ(4, selectionData.numClusterCells);
    EXPECT_EQ(0, selectionData.numEnergyParticles);
}

TEST_F(EditTests, injectGenomeToSelectedCreatures_noSelection)
{
    auto data = Desc().addCreature({ObjectDesc().id(1).pos({50, 50})}, CreatureDesc(), GenomeDesc());
    _simulationFacade->setSimulationData(data);

    _simulationFacade->removeSelection();

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    auto result = _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    EXPECT_EQ(0, result);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(1, actualData._genomes.size());
    EXPECT_EQ(0, actualData._genomes.front()._genes.size());
}

TEST_F(EditTests, injectGenomeToSelectedCreatures_singleCreature)
{
    auto data = Desc().addCreature(
        {
            ObjectDesc().id(1).pos({50, 50}),
            ObjectDesc().id(2).pos({51, 50}),
            ObjectDesc().id(3).pos({52, 50}),
        },
        CreatureDesc(),
        GenomeDesc());
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {53, 51});

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    auto result = _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    EXPECT_EQ(1, result);
}

TEST_F(EditTests, injectGenomeToSelectedCreatures_multipleCreatures_onlySelectedAffected)
{
    auto data = Desc()
                    .addCreature({ObjectDesc().id(1).pos({50, 50})}, CreatureDesc(), GenomeDesc())
                    .addCreature({ObjectDesc().id(2).pos({80, 80})}, CreatureDesc(), GenomeDesc());
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {51, 51});

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    auto result = _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    EXPECT_EQ(1, result);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._creatures.size());

    auto selectedCreatureGenomeModified = false;
    auto unselectedCreatureGenomeUnchanged = false;
    for (auto const& creature : actualData._creatures) {
        auto genomeIt =
            std::find_if(actualData._genomes.begin(), actualData._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
        ASSERT_NE(genomeIt, actualData._genomes.end());

        if (genomeIt->_genes.size() == 1) {
            EXPECT_EQ(2, genomeIt->_genes.front()._nodes.size());
            selectedCreatureGenomeModified = true;
        } else {
            EXPECT_EQ(0, genomeIt->_genes.size());
            unselectedCreatureGenomeUnchanged = true;
        }
    }
    EXPECT_TRUE(selectedCreatureGenomeModified);
    EXPECT_TRUE(unselectedCreatureGenomeUnchanged);
}

TEST_F(EditTests, injectGenomeToSelectedCreatures_allSelected)
{
    auto data = Desc()
                    .addCreature({ObjectDesc().id(1).pos({50, 50})}, CreatureDesc(), GenomeDesc())
                    .addCreature({ObjectDesc().id(2).pos({60, 50})}, CreatureDesc(), GenomeDesc());
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {61, 51});

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    auto result = _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    EXPECT_EQ(2, result);
}
