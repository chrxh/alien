#include <random>
#include <ranges>

#include <gtest/gtest.h>

#include <Data/Descs.h>

#include <EngineInterface/GenomeValidationService.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

// GenomeValidationService replays the gene graph corrections on the CPU, so both have to arrive at the same genome
class GenomeValidationParityTests : public MutationTestsBase
{
protected:
    GenomeDesc createRandomGenome(std::mt19937& randomEngine) const
    {
        auto randomInt = [&](int min, int max) { return std::uniform_int_distribution(min, max)(randomEngine); };
        auto randomEvent = [&](double probability) { return std::bernoulli_distribution(probability)(randomEngine); };

        auto numGenes = randomInt(1, 6);
        std::vector<GeneDesc> genes;
        for ([[maybe_unused]] auto geneIndex : std::views::iota(0, numGenes)) {
            std::vector<NodeDesc> nodes;
            for ([[maybe_unused]] auto nodeIndex : std::views::iota(0, randomInt(1, 12))) {
                NodeDesc node;
                if (randomEvent(0.15)) {
                    node.cellType(VoidGenomeDesc());
                } else {
                    if (randomEvent(0.1)) {
                        node.cellType(InjectorGenomeDesc().geneIndex(randomInt(0, numGenes - 1)));
                    }
                    if (randomEvent(0.35)) {
                        node.constructor(ConstructorGenomeDesc().geneIndex(randomInt(0, numGenes - 1)).separation(randomEvent(0.3)));
                    }
                }
                nodes.emplace_back(node);
            }
            genes.emplace_back(GeneDesc().shape(randomInt(0, ConstructorShape_Count - 1)).homogeneousCellType(randomEvent(0.1)).nodes(nodes));
        }
        return GenomeDesc().genes(genes);
    }
};

TEST_F(GenomeValidationParityTests, correctionMatchesEngine)
{
    std::mt19937 randomEngine(1);
    for (auto run : std::views::iota(0, 200)) {
        auto genome = createRandomGenome(randomEngine);
        auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

        _simulationFacade->setSimulationData(data);
        _simulationFacade->testOnly_voidUnreachableNodes(1);
        _simulationFacade->testOnly_removeGeneCycles(1);
        _simulationFacade->testOnly_removeUnusedGenes(1);
        _simulationFacade->testOnly_limitGenesWithSeparation(1);

        auto expectedGenome = genome;
        GenomeValidationService::get().correct(expectedGenome);
        EXPECT_EQ(expectedGenome._genes, getMutatedGenome()._genes) << "run " << run;
    }
}
