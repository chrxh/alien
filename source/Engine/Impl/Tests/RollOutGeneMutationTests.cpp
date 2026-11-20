#include <limits>
#include <ranges>
#include <vector>

#include <gtest/gtest.h>

#include <Data/CellTypeConstants.h>
#include <Data/Descs.h>

#include <Engine/Interface/SimulationFacade.h>

#include "MutationTestsBase.h"

class RollOutGeneMutationTests : public MutationTestsBase
{
protected:
    GenomeDesc mutate(GenomeDesc genome, float rollOutProbability, float rollUpProbability)
    {
        genome._mutationRates._rollOutGeneMutation = RollOutGeneMutationDesc().geneProbability(rollOutProbability);
        genome._mutationRates._rollUpGeneMutation = RollUpGeneMutationDesc().geneProbability(rollUpProbability);
        _simulationFacade->setSimulationData(ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome));
        _simulationFacade->testOnly_mutate(1);
        return getMutatedGenome();
    }

    std::vector<NodeDesc> repeatNodes(std::vector<NodeDesc> const& nodes, int numCopies) const
    {
        std::vector<NodeDesc> result;
        for ([[maybe_unused]] auto copy : std::views::iota(0, numCopies)) {
            result.insert(result.end(), nodes.begin(), nodes.end());
            result.at(result.size() - nodes.size())._restartConstruction = true;
        }
        return result;
    }
};

TEST_F(RollOutGeneMutationTests, rollOutGeneMutation_repeatsNodesByCommonConcatenations)
{
    auto builtNodes = std::vector<NodeDesc>{NodeDesc().color(0).referenceAngle(10.0f), NodeDesc().color(1), NodeDesc().color(2).referenceAngle(-10.0f)};
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(4)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(6)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(std::numeric_limits<int>::max())),
        }),
        GeneDesc().nodes(builtNodes),
    });

    auto actualGenome = mutate(genome, 1.0f, 0.0f);

    auto expectedGenome = genome;
    expectedGenome._mutationRates = actualGenome._mutationRates;
    expectedGenome._genes.at(0)._nodes.at(0)._constructor->_numConcatenations = 2;
    expectedGenome._genes.at(0)._nodes.at(1)._constructor->_numConcatenations = 3;
    expectedGenome._genes.at(1)._nodes = repeatNodes(builtNodes, 2);
    EXPECT_EQ(expectedGenome, actualGenome);
}

TEST_F(RollOutGeneMutationTests, rollOutGeneMutation_rollsOutGeneBuildingItself)
{
    auto nodes =
        std::vector<NodeDesc>{NodeDesc().color(0), NodeDesc().color(1).constructor(ConstructorGenomeDesc().geneIndex(0).separation(true).numConcatenations(3))};

    auto actualGenome = mutate(GenomeDesc().genes({GeneDesc().nodes(nodes)}), 1.0f, 0.0f);

    nodes.at(1)._constructor->_numConcatenations = 1;
    EXPECT_EQ(repeatNodes(nodes, 3), actualGenome._genes.at(0)._nodes);
}

TEST_F(RollOutGeneMutationTests, rollOutGeneMutation_keepsGeneReferencedByInjector)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(3)),
            NodeDesc().cellType(InjectorGenomeDesc().geneIndex(1)),
        }),
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
    });

    auto actualGenome = mutate(genome, 1.0f, 0.0f);

    EXPECT_EQ(genome._genes, actualGenome._genes);
}

TEST_F(RollOutGeneMutationTests, rollOutGeneMutation_keepsGeneThatWouldExceedNodeLimit)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(2))}),
        GeneDesc().nodes(std::vector<NodeDesc>(1001)),
    });

    auto actualGenome = mutate(genome, 1.0f, 0.0f);

    EXPECT_EQ(genome._genes, actualGenome._genes);
}

TEST_F(RollOutGeneMutationTests, rollUpGeneMutation_cutsGeneBackToFirstCopy)
{
    std::vector<NodeDesc> builtNodes;
    for (auto color : std::views::iota(0, 6)) {
        builtNodes.emplace_back(NodeDesc().color(color));
    }
    builtNodes.at(2)._restartConstruction = true;
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(2)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(std::numeric_limits<int>::max())),
        }),
        GeneDesc().nodes(builtNodes),
    });

    auto actualGenome = mutate(genome, 0.0f, 1.0f);

    auto expectedGenome = genome;
    expectedGenome._mutationRates = actualGenome._mutationRates;
    expectedGenome._genes.at(0)._nodes.at(0)._constructor->_numConcatenations = 6;
    expectedGenome._genes.at(1)._nodes.resize(2);
    EXPECT_EQ(expectedGenome, actualGenome);
}

TEST_F(RollOutGeneMutationTests, rollUpGeneMutation_keepsGenesWithoutFittingCopy)
{
    std::vector<NodeDesc> nodesWithRemainder(5);
    nodesWithRemainder.at(2)._restartConstruction = true;
    std::vector<NodeDesc> nodesWithRestartAtStart(3);
    nodesWithRestartAtStart.at(0)._restartConstruction = true;
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(2)),
            NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(2).numConcatenations(2)),
        }),
        GeneDesc().nodes(nodesWithRemainder),
        GeneDesc().nodes(nodesWithRestartAtStart),
    });

    auto actualGenome = mutate(genome, 0.0f, 1.0f);

    EXPECT_EQ(genome._genes, actualGenome._genes);
}

TEST_F(RollOutGeneMutationTests, rollUpGeneMutation_replacesVoidAtNewLastNode)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().nodes({NodeDesc(), NodeDesc().cellType(VoidGenomeDesc()), NodeDesc().restartConstruction(true), NodeDesc()}),
    });

    auto actualGenome = mutate(genome, 0.0f, 1.0f);

    auto const& actualNodes = actualGenome._genes.at(1)._nodes;
    ASSERT_EQ(2, actualNodes.size());
    EXPECT_NE(CellType_Void, actualNodes.at(1).getCellType());
    EXPECT_EQ(2, actualGenome._genes.at(0)._nodes.at(0)._constructor->_numConcatenations);
}

TEST_F(RollOutGeneMutationTests, rollOutAndRollUpGeneMutations_restoreGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numConcatenations(3))}),
        GeneDesc().nodes({NodeDesc().color(0).restartConstruction(true), NodeDesc().color(1), NodeDesc().color(2)}),
    });

    auto actualGenome = mutate(genome, 1.0f, 1.0f);

    EXPECT_EQ(genome._genes, actualGenome._genes);
}
