#include <ranges>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class GeometryMutationTests : public MutationTestsBase
{
};

TEST_F(GeometryMutationTests, geometryMutation_changesShapeStiffness)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    for (auto const& [expectedGene, actualGene] : std::views::zip(genome._genes, actualGenome._genes)) {
        EXPECT_NE(expectedGene._shape, actualGene._shape);
        EXPECT_NE(expectedGene._stiffness, actualGene._stiffness);

        EXPECT_GE(actualGene._stiffness, Const::GeneStiffness_Min);
        EXPECT_LE(actualGene._stiffness, Const::GeneStiffness_Max);
        EXPECT_GE(actualGene._shape, 0);
        EXPECT_LT(actualGene._shape, ConstructorShape_Count);
    }
}

TEST_F(GeometryMutationTests, geometryMutation_zeroProbabilityNoChange)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(0.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
    genome._mutationRates._geometryMutations[1] = GeometryMutationDesc().geneProbability(0.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    for (auto const& [expectedGene, actualGene] : std::views::zip(genome._genes, actualGenome._genes)) {
        EXPECT_EQ(expectedGene._shape, actualGene._shape);
        EXPECT_EQ(expectedGene._stiffness, actualGene._stiffness);
    }
}

TEST_F(GeometryMutationTests, geometryMutation_doesNotChangeNodes)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().geneProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
    genome._mutationRates._geometryMutations[1] = GeometryMutationDesc().geneProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    for (auto const& [expectedGene, actualGene] : std::views::zip(genome._genes, actualGenome._genes)) {
        EXPECT_EQ(expectedGene._nodes, actualGene._nodes);
        EXPECT_EQ(expectedGene._name, actualGene._name);
        EXPECT_EQ(expectedGene._homogeneousCellType, actualGene._homogeneousCellType);
    }
}
