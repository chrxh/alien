#include <algorithm>
#include <optional>

#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class ConnectionMutationTests : public MutationTestsBase
{
protected:
    bool compareAllExceptConnectionWeights(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    std::fill(node._neuralNetwork._connectionWeights.begin(), node._neuralNetwork._connectionWeights.end(), 0.0f);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }
};

TEST_F(ConnectionMutationTests, connectionWeightMutation_weightsActuallyChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().nodeProbability(0.0f).valueChangeSigma(0.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // At least some connection weights should have changed
    bool anyWeightChanged = false;
    for (size_t g = 0; g < genome._genes.size() && !anyWeightChanged; ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size() && !anyWeightChanged; ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            auto const& actualWeights = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            for (size_t w = 0; w < origWeights.size(); ++w) {
                if (origWeights.at(w) != actualWeights.at(w)) {
                    anyWeightChanged = true;
                    break;
                }
            }
        }
    }
    EXPECT_TRUE(anyWeightChanged);

    // All connection weights must be within [-2, 2]
    for (auto const& gene : actualGenome._genes) {
        for (auto const& node : gene._nodes) {
            for (auto const& w : node._neuralNetwork._connectionWeights) {
                EXPECT_GE(w, -2.0f - 0.1f);
                EXPECT_LE(w, 2.0f + 0.1f);
            }
        }
    }
}

TEST_F(ConnectionMutationTests, connectionWeightMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(0.0f).valueChangeSigma(1.0f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().nodeProbability(0.0f).valueChangeSigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    // No connection weights should have changed
    for (size_t g = 0; g < genome._genes.size(); ++g) {
        for (size_t n = 0; n < genome._genes.at(g)._nodes.size(); ++n) {
            auto const& origWeights = genome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            auto const& actualWeights = actualGenome._genes.at(g)._nodes.at(n)._neuralNetwork._connectionWeights;
            EXPECT_EQ(origWeights, actualWeights);
        }
    }
}

TEST_F(ConnectionMutationTests, connectionWeightMutation_keepOtherAttributesUnchanged)
{
    auto genome = createTestGenome();
    genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f);
    genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCell = actualData.getObjectRef(1).getCellRef();
    auto actualCreature = actualData.getCreatureRef(actualCell._creatureId);
    auto actualGenome = actualData.getGenomeRef(actualCreature._genomeId);

    EXPECT_TRUE(compareAllExceptConnectionWeights(genome, actualGenome));
}
