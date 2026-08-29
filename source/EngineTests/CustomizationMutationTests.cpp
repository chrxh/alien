#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <EngineInterface/Descs.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class CustomizationMutationTests : public MutationTestsBase
{
protected:
    void setTransitionMatrix(std::vector<std::pair<int, int>> const& allowedTransitions)
    {
        _parameters.customizationTransitionMatrix.value = ColorMatrix<bool>::uniform(false);
        for (auto const& [sourceColor, targetColor] : allowedTransitions) {
            _parameters.customizationTransitionMatrix.value[sourceColor][targetColor] = true;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }
};

TEST_F(CustomizationMutationTests, customizationMutation_allNodesGetAllowedCustomization)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().color(3),
        NodeDesc().color(3),
        NodeDesc().color(3),
    })});
    genome._mutationRates._customizationMutation = CustomizationMutationDesc().genomeProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    setTransitionMatrix({{3, 7}});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();
    for (auto const& node : actualGenome._genes.at(0)._nodes) {
        EXPECT_EQ(node._color, 7);
    }
}

TEST_F(CustomizationMutationTests, customizationMutation_targetIsChosenAmongAllowedCustomizations)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().color(0), NodeDesc().color(0)})});
    genome._mutationRates._customizationMutation = CustomizationMutationDesc().genomeProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    setTransitionMatrix({{0, 4}, {0, 9}, {4, 0}, {4, 9}, {9, 0}, {9, 4}});

    _simulationFacade->setSimulationData(data);

    std::set<int> obtainedColors;
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);

        auto actualGenome = getMutatedGenome();
        auto const& nodes = actualGenome._genes.at(0)._nodes;
        EXPECT_EQ(nodes.at(0)._color, nodes.at(1)._color);
        obtainedColors.insert(nodes.at(0)._color);
    }
    EXPECT_EQ(obtainedColors, (std::set<int>{0, 4, 9}));
}

TEST_F(CustomizationMutationTests, customizationMutation_otherCustomizationsRemainUnchanged)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({
        NodeDesc().color(3),
        NodeDesc().color(5),
        NodeDesc().color(3),
    })});
    genome._mutationRates._customizationMutation = CustomizationMutationDesc().genomeProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    setTransitionMatrix({{3, 7}});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 20; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    auto const& nodes = actualGenome._genes.at(0)._nodes;
    EXPECT_EQ(nodes.at(0)._color, 7);
    EXPECT_EQ(nodes.at(1)._color, 5);
    EXPECT_EQ(nodes.at(2)._color, 7);
}

TEST_F(CustomizationMutationTests, customizationMutation_noAllowedTransitionNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().color(3), NodeDesc().color(3)})});
    genome._mutationRates._customizationMutation = CustomizationMutationDesc().genomeProbability(1.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    setTransitionMatrix({});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome, genome);
}

TEST_F(CustomizationMutationTests, customizationMutation_zeroProbabilityNoChange)
{
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc().color(3), NodeDesc().color(3)})});
    genome._mutationRates._customizationMutation = CustomizationMutationDesc().genomeProbability(0.0f);

    auto data = ContentDesc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    setTransitionMatrix({{3, 7}});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();
    EXPECT_EQ(actualGenome, genome);
}
