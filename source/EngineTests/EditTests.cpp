#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include <Base/Math.h>

#include <Data/CreatorService.h>
#include <Data/Descs.h>

#include <EngineInterface/SelectionShallowData.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>
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
    auto data = ContentDesc().addCreature({
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
    auto data = ContentDesc().addCreature({
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
    auto data = ContentDesc()
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
    auto data = ContentDesc().energies({
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
    auto data = ContentDesc()
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
    auto data = ContentDesc()
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
    auto data = ContentDesc()
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
    auto data = ContentDesc().addCreature({ObjectDesc().id(1).pos({50, 50})}, CreatureDesc(), GenomeDesc());
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
    auto data = ContentDesc().addCreature(
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
    auto data = ContentDesc()
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
    auto data = ContentDesc()
                    .addCreature({ObjectDesc().id(1).pos({50, 50})}, CreatureDesc(), GenomeDesc())
                    .addCreature({ObjectDesc().id(2).pos({60, 50})}, CreatureDesc(), GenomeDesc());
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {61, 51});

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    auto result = _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    EXPECT_EQ(2, result);
}

TEST_F(EditTests, setStatic_releaseAfterTimesteps)
{
    auto const center = RealVector2D{50.0f, 50.0f};
    auto data = CreatorService::get().createRectangle(CreatorService::ObjectProperties().isStatic(true), center, {10, 10}, 1.0f);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(10);

    _simulationFacade->setSelection({40, 40}, {60, 60});
    _simulationFacade->setStatic(false, true);
    _simulationFacade->calcTimesteps(10);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(100, actualData._objects.size());
    for (auto const& object : actualData._objects) {
        EXPECT_FALSE(object._isStatic);
        ASSERT_TRUE(std::isfinite(object._pos.x) && std::isfinite(object._pos.y));
        EXPECT_LT(Math::length(object._pos - center), 10.0f);
    }
}

TEST_F(EditTests, getSelectionShallowData_bounds)
{
    auto data = ContentDesc().addCreature({
        ObjectDesc().id(1).pos({50, 50}),
        ObjectDesc().id(2).pos({51, 51}),
        ObjectDesc().id(3).pos({52, 52}),
    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({49, 49}, {51.5f, 51.5f});
    auto selectionData = _simulationFacade->getSelectionShallowData();

    EXPECT_TRUE(approxCompare(50.0f, selectionData.minPosX));
    EXPECT_TRUE(approxCompare(50.0f, selectionData.minPosY));
    EXPECT_TRUE(approxCompare(51.0f, selectionData.maxPosX));
    EXPECT_TRUE(approxCompare(51.0f, selectionData.maxPosY));
    EXPECT_TRUE(approxCompare(50.0f, selectionData.clusterMinPosX));
    EXPECT_TRUE(approxCompare(50.0f, selectionData.clusterMinPosY));
    EXPECT_TRUE(approxCompare(52.0f, selectionData.clusterMaxPosX));
    EXPECT_TRUE(approxCompare(52.0f, selectionData.clusterMaxPosY));
}

namespace
{
    ContentDesc createTwoConnectedAndOneFreeObject()
    {
        auto result = ContentDesc().addObjects({
            ObjectDesc().id(1).pos({50, 50}).type(SolidDesc()),
            ObjectDesc().id(2).pos({51, 50}).type(SolidDesc()),
            ObjectDesc().id(3).pos({60, 50}).type(SolidDesc()),
        });
        result.addConnection(1, 2);
        return result;
    }
}

TEST_F(EditTests, shallowUpdateSelectedObjects_selectedObjects_connectionsOnlyTear)
{
    _simulationFacade->setSimulationData(createTwoConnectedAndOneFreeObject());
    _simulationFacade->setSelection({50.5f, 49}, {51.5f, 51});

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = false;
    updateData.posDeltaX = 8.0f;
    _simulationFacade->shallowUpdateSelectedObjects(updateData);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(59.0f, actualData.getObjectRef(2)._pos.x));
    EXPECT_FALSE(actualData.hasConnection(1, 2));
    EXPECT_FALSE(actualData.hasConnection(2, 3));
}

TEST_F(EditTests, shallowUpdateSelectedObjects_selectedObjects_glueOnContact)
{
    _simulationFacade->setSimulationData(createTwoConnectedAndOneFreeObject());
    _simulationFacade->setSelection({50.5f, 49}, {51.5f, 51});

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = false;
    updateData.glueOnContact = true;
    updateData.posDeltaX = 8.0f;
    _simulationFacade->shallowUpdateSelectedObjects(updateData);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualData.hasConnection(2, 3));
    EXPECT_TRUE(actualData.hasConnection(3, 2));
}

TEST_F(EditTests, shallowUpdateSelectedObjects_entireNetworks_connectionsRemain)
{
    _simulationFacade->setSimulationData(createTwoConnectedAndOneFreeObject());
    _simulationFacade->setSelection({50.5f, 49}, {51.5f, 51});

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = true;
    updateData.posDeltaX = 3.0f;
    _simulationFacade->shallowUpdateSelectedObjects(updateData);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(approxCompare(53.0f, actualData.getObjectRef(1)._pos.x));
    EXPECT_TRUE(approxCompare(54.0f, actualData.getObjectRef(2)._pos.x));
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_FALSE(actualData.hasConnection(2, 3));
}

TEST_F(EditTests, glueSelectedObjects_onlyWithinSelection)
{
    auto data = ContentDesc().addObjects({
        ObjectDesc().id(1).pos({50, 50}).type(SolidDesc()),
        ObjectDesc().id(2).pos({51, 50}).type(SolidDesc()),
        ObjectDesc().id(3).pos({52, 50}).type(SolidDesc()),
    });
    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSelection({49, 49}, {51.5f, 51});

    _simulationFacade->glueSelectedObjects(false);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(actualData.hasConnection(2, 1));
    EXPECT_FALSE(actualData.hasConnection(2, 3));
}

namespace
{
    ContentDesc createChainOfThreeObjects()
    {
        auto result = ContentDesc().addObjects({
            ObjectDesc().id(1).pos({50, 50}).type(SolidDesc()),
            ObjectDesc().id(2).pos({51, 50}).type(SolidDesc()),
            ObjectDesc().id(3).pos({52, 50}).type(SolidDesc()),
        });
        result.addConnection(1, 2);
        result.addConnection(2, 3);
        return result;
    }
}

TEST_F(EditTests, cutConnections_crossedConnection)
{
    _simulationFacade->setSimulationData(createChainOfThreeObjects());

    _simulationFacade->cutConnections({50.5f, 49}, {50.5f, 51}, false, false);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(actualData.hasConnection(1, 2));
    EXPECT_FALSE(actualData.hasConnection(2, 1));
    EXPECT_TRUE(actualData.hasConnection(2, 3));
}

TEST_F(EditTests, cutConnections_onlyInSelection)
{
    _simulationFacade->setSimulationData(createChainOfThreeObjects());
    _simulationFacade->setSelection({50.5f, 49}, {52.5f, 51});

    _simulationFacade->cutConnections({50.5f, 49}, {50.5f, 51}, true, false);
    _simulationFacade->cutConnections({51.5f, 49}, {51.5f, 51}, true, false);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_FALSE(actualData.hasConnection(2, 3));
    EXPECT_FALSE(actualData.hasConnection(3, 2));
}
