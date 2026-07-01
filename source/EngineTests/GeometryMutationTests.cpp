#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class GeometryMutationTests : public MutationTestsBase
{
};

TEST_F(GeometryMutationTests, geometryMutation_changesShapeStiffnessAndConnectionDistance)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 20; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    bool shapeChanged = false;
    bool stiffnessChanged = false;
    bool connectionDistanceChanged = false;
    for (size_t i = 0; i < genome._genes.size(); ++i) {
        if (genome._genes[i]._shape != actualGenome._genes[i]._shape) {
            shapeChanged = true;
        }
        if (genome._genes[i]._stiffness != actualGenome._genes[i]._stiffness) {
            stiffnessChanged = true;
        }
        if (genome._genes[i]._connectionDistance != actualGenome._genes[i]._connectionDistance) {
            connectionDistanceChanged = true;
        }
    }
    EXPECT_TRUE(shapeChanged);
    EXPECT_TRUE(stiffnessChanged);
    EXPECT_TRUE(connectionDistanceChanged);
}

TEST_F(GeometryMutationTests, geometryMutation_respectsBounds)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
    genome._mutationRates._geometryMutations[1] = GeometryMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    for (auto const& gene : actualGenome._genes) {
        EXPECT_GE(gene._stiffness, Const::GeneStiffness_Min);
        EXPECT_LE(gene._stiffness, Const::GeneStiffness_Max);
        EXPECT_GE(gene._connectionDistance, Const::GeneConnectionDistance_Min);
        EXPECT_LE(gene._connectionDistance, Const::GeneConnectionDistance_Max);
        EXPECT_GE(gene._shape, 0);
        EXPECT_LT(gene._shape, ConstructorShape_Count);
    }
}

TEST_F(GeometryMutationTests, geometryMutation_doesNotChangeNodes)
{
    auto genome = createTestGenome();
    genome._mutationRates._geometryMutations[0] = GeometryMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);
    genome._mutationRates._geometryMutations[1] = GeometryMutationDesc().nodeProbability(1.0f).valueChangeSigma(1.0f).enumChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    ASSERT_EQ(genome._genes.size(), actualGenome._genes.size());
    for (size_t i = 0; i < genome._genes.size(); ++i) {
        EXPECT_EQ(genome._genes[i]._nodes, actualGenome._genes[i]._nodes);
        EXPECT_EQ(genome._genes[i]._name, actualGenome._genes[i]._name);
        EXPECT_EQ(genome._genes[i]._homogeneousCellType, actualGenome._genes[i]._homogeneousCellType);
    }
}
