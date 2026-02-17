#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include <Base/Definitions.h>
#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/PreviewDescriptionConverterService.h>
#include <EngineInterface/SpaceCalculator.h>

#include <EngineTestData/TestHelper.h>

class PreviewDescConverterServiceTests : public ::testing::Test
{
public:
    PreviewDescConverterServiceTests() = default;
    virtual ~PreviewDescConverterServiceTests() = default;

    CellPreviewDesc const& getPreviewCell(PreviewDesc const& preview, int geneIndex, int nodeIndex)
    {
        for (auto const& object : preview._objects) {
            if (object._geneIndex == geneIndex && object._nodeIndex == nodeIndex) {
                return object;
            }
        }
        CHECK(false);
    }

    struct ConnectionCheckDescription
    {
        RealVector2D object1;
        RealVector2D object2;
        std::optional<float> connectionWeightToObject1 = std::nullopt;
        std::optional<float> connectionWeightToObject2 = std::nullopt;
    };
    void checkConnections(PreviewDesc const& preview, std::vector<ConnectionCheckDescription> const& expectedConnections)
    {
        for (auto const& expectedConnection : expectedConnections) {
            auto found = false;
            for (auto const& connection : preview._connections) {
                if (TestHelper::TestHelper::approxCompare(expectedConnection.object1, connection._object1)
                    && TestHelper::TestHelper::approxCompare(expectedConnection.object2, connection._object2)
                    && (!expectedConnection.connectionWeightToObject1.has_value()
                        || expectedConnection.connectionWeightToObject1.value() == connection._connectionWeightToObject1)
                    && (!expectedConnection.connectionWeightToObject2.has_value()
                        || expectedConnection.connectionWeightToObject2.value() == connection._connectionWeightToObject2)) {
                    found = true;
                    break;
                }
                if (TestHelper::TestHelper::approxCompare(expectedConnection.object2, connection._object1)
                    && TestHelper::TestHelper::approxCompare(expectedConnection.object1, connection._object2)
                    && (!expectedConnection.connectionWeightToObject2.has_value()
                        || expectedConnection.connectionWeightToObject2.value() == connection._connectionWeightToObject1)
                    && (!expectedConnection.connectionWeightToObject1.has_value()
                        || expectedConnection.connectionWeightToObject1.value() == connection._connectionWeightToObject2)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                EXPECT_TRUE(false);
            }
        }
    }
};

TEST_F(PreviewDescConverterServiceTests, convertEmptyCollection)
{
    Desc input;

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(GenomeDesc(), 0, std::move(input), std::nullopt);

    EXPECT_TRUE(result.description._objects.empty());
    EXPECT_TRUE(result.description._connections.empty());
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_withSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc().color(2), NodeDesc().color(3)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
        },
        CreatureDesc(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    EXPECT_EQ(2, object1._color);
    EXPECT_EQ(3, object2._color);

    auto expectedCell1_pos = RealVector2D{0, 0.5f};
    auto expectedCell2_pos = RealVector2D{0, -0.5f};
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell1_pos, object1._pos));
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell2_pos, object2._pos));
    checkConnections(result.description, {{object1._pos, object2._pos, 1.0f, 1.0f}});
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_withSeparation_nonRootGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc().color(1)}),
        GeneDesc().separation(true).nodes({NodeDesc().color(2), NodeDesc().color(3)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(1)),
        },
        CreatureDesc(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 1, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 1, 0);
    auto object2 = getPreviewCell(result.description, 1, 1);
    EXPECT_EQ(2, object1._color);
    EXPECT_EQ(3, object2._color);

    auto expectedCell1_pos = RealVector2D{0, 0.5f};
    auto expectedCell2_pos = RealVector2D{0, -0.5f};
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell1_pos, object1._pos));
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell2_pos, object2._pos));
    checkConnections(result.description, {{object1._pos, object2._pos, 1.0f, 1.0f}});
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_withoutSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({NodeDesc().color(2), NodeDesc().color(3)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
        },
        CreatureDesc(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    EXPECT_EQ(2, object1._color);
    EXPECT_EQ(3, object2._color);

    auto expectedCell1_pos = RealVector2D{0, 0.5f};
    auto expectedCell2_pos = RealVector2D{0, -0.5f};
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell1_pos, object1._pos));
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell2_pos, object2._pos));
    checkConnections(result.description, {{object1._pos, object2._pos, 1.0f, 1.0f}});
}

TEST_F(PreviewDescConverterServiceTests, convertThreeCellCreature)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc().color(2), NodeDesc().color(3), NodeDesc().color(4)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 9.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(2).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
            ObjectDesc().id(3).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(2)),
        },
        CreatureDesc(),
        genome);
    input.addConnection(1, 2);
    input.addConnection(2, 3);
    input.addConnection(3, 1);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(3, result.description._objects.size());
    ASSERT_EQ(3, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    auto cell3 = getPreviewCell(result.description, 0, 2);
    EXPECT_EQ(2, object1._color);
    EXPECT_EQ(3, object2._color);
    EXPECT_EQ(4, cell3._color);

    auto oneThird = 0.333333f;
    auto expectedCell1_pos = RealVector2D{oneThird, oneThird * 2};
    auto expectedCell2_pos = RealVector2D{oneThird, -oneThird};
    auto expectedCell3_pos = RealVector2D{-oneThird * 2, -oneThird};
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell1_pos, object1._pos));
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell2_pos, object2._pos));
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell3_pos, cell3._pos));
    checkConnections(result.description, {{object1._pos, object2._pos}, {object2._pos, cell3._pos}, {cell3._pos, object1._pos}});
}

TEST_F(PreviewDescConverterServiceTests, convertCreature_oneGene_multipleNodes_multipleConcatenations)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numConcatenations(4).nodes({NodeDesc().color(2), NodeDesc().color(3)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 4.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(2).pos({10.0f, 5.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
            ObjectDesc().id(3).pos({10.0f, 6.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(4).pos({10.0f, 7.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
            ObjectDesc().id(5).pos({10.0f, 8.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(6).pos({10.0f, 9.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
            ObjectDesc().id(7).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(8).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
        },
        CreatureDesc(),
        genome);
    for (int i = 0; i < 7; ++i) {
        input.addConnection(i + 1, i + 2);
    }

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._objects.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& object : result.description._objects) {
        actualPositions.insert(object._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {-0.875000238, -2.62500000},
        {0.124999769, -2.62500000},
        {0.124999858, -1.62500000},
        {0.124999948, -0.625000000},
        {0.125000030, 0.375000000},
        {0.125000119, 1.37500000},
        {0.125000209, 2.37500000},
        {0.125000298, 3.37500000},
    };
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(TestHelper::approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescConverterServiceTests, convertCreature_oneGene_oneNode_multipleConcatenations)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numConcatenations(8).nodes({NodeDesc().color(2)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 4.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(2).pos({10.0f, 5.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(3).pos({10.0f, 6.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(4).pos({10.0f, 7.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(5).pos({10.0f, 8.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(6).pos({10.0f, 9.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(7).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(8).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
        },
        CreatureDesc(),
        genome);
    for (int i = 0; i < 7; ++i) {
        input.addConnection(i + 1, i + 2);
    }

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._objects.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& object : result.description._objects) {
        actualPositions.insert(object._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {-0.875000238, -2.62500000},
        {0.124999769, -2.62500000},
        {0.124999858, -1.62500000},
        {0.124999948, -0.625000000},
        {0.125000030, 0.375000000},
        {0.125000119, 1.37500000},
        {0.125000209, 2.37500000},
        {0.125000298, 3.37500000},
    };
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(TestHelper::approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescConverterServiceTests, convertCreature_twoGenes_oneNode_multipleConcatenations)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({NodeDesc().color(2).constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().separation(false).numConcatenations(7).nodes({NodeDesc().color(3)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(2).pos({10.0f, 4.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(3).pos({10.0f, 5.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(4).pos({10.0f, 6.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(5).pos({10.0f, 7.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(6).pos({10.0f, 8.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(7).pos({10.0f, 9.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(8).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(1).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0).constructor(ConstructorDesc().geneIndex(1))),
        },
        CreatureDesc(),
        genome);
    input.addConnection(2, 3);
    input.addConnection(3, 4);
    input.addConnection(4, 5);
    input.addConnection(5, 6);
    input.addConnection(6, 7);
    input.addConnection(7, 8);
    input.addConnection(8, 1);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._objects.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& object : result.description._objects) {
        actualPositions.insert(object._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {3.375f, -0.125f}, {2.375f, -0.125f}, {1.375f, -0.125f}, {0.375f, -0.125f}, {-0.625f, -0.125}, {-1.625f, -0.125}, {-2.625f, -0.125f}, {-2.625f, 0.875f}};
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(TestHelper::approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescConverterServiceTests, convertCreature_oneGenes_twoNode_multipleConcatenations)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).numConcatenations(4).nodes({NodeDesc(), NodeDesc()}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 4.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(2).pos({10.0f, 5.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
            ObjectDesc().id(3).pos({10.0f, 6.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(4).pos({10.0f, 7.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
            ObjectDesc().id(5).pos({10.0f, 8.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(6).pos({10.0f, 9.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
            ObjectDesc().id(7).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(8).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
        },
        CreatureDesc(),
        genome);
    input.addConnection(1, 2);
    input.addConnection(2, 3);
    input.addConnection(3, 4);
    input.addConnection(4, 5);
    input.addConnection(5, 6);
    input.addConnection(6, 7);
    input.addConnection(7, 8);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(8, result.description._objects.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& object : result.description._objects) {
        actualPositions.insert(object._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {-0.875000238, -2.62500000},
        {0.124999769, -2.62500000},
        {0.124999858, -1.62500000},
        {0.124999948, -0.625000000},
        {0.125000030, 0.375000000},
        {0.125000119, 1.37500000},
        {0.125000209, 2.37500000},
        {0.125000298, 3.37500000},
    };
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(TestHelper::approxCompare(expectedPosition, actualPosition));
    }
}

TEST_F(PreviewDescConverterServiceTests, convertCastratedCreature_withSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().separation(true).nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))}),
    });

    Desc inputCreature1;
    inputCreature1.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0).constructor(ConstructorDesc().geneIndex(2))),
        },
        CreatureDesc(),
        genome);

    Desc inputCreature2;
    inputCreature2.addCreature(
        {
            ObjectDesc().id(2).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(0).constructor(ConstructorDesc().geneIndex(2))),
        },
        CreatureDesc(),
        genome);
    {
        auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(inputCreature1), std::nullopt);

        ASSERT_EQ(1, result.description._objects.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto object1 = getPreviewCell(result.description, 0, 0);
        EXPECT_EQ(1, object1._constructorGeneIndex);
    }
    {
        auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 1, std::move(inputCreature2), std::nullopt);

        ASSERT_EQ(1, result.description._objects.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto object1 = getPreviewCell(result.description, 1, 0);
        EXPECT_EQ(0, object1._constructorGeneIndex);
    }
}

TEST_F(PreviewDescConverterServiceTests, convertCastratedCreature_withoutSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(false).nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1))}),
        GeneDesc().separation(false).nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0))}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(0).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0).constructor(ConstructorDesc().geneIndex(1))),
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(0).constructor(ConstructorDesc().geneIndex(0))),
        },
        CreatureDesc(),
        genome);
    input.addConnection(0, 1);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    EXPECT_EQ(1, object1._constructorGeneIndex.value());

    auto object2 = getPreviewCell(result.description, 1, 0);
    EXPECT_EQ(0, object2._constructorGeneIndex.value());
}

TEST_F(PreviewDescConverterServiceTests, convertCreatureWithSignals)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()}),
    });

    std::vector<float> signal{0.2f, 0.2f, 0.2f, 0.8f, 0.2f, -1.2f, 0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0).signal(signal)),
            ObjectDesc().id(2).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
        },
        CreatureDesc(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);

    EXPECT_EQ(signal, object1._signal._channels);
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_connectionWeightsFromGenome)
{
    auto nn1 = NeuralNetGenomeDesc();
    nn1._connectionWeights.at(0) = 0.5f;
    auto nn2 = NeuralNetGenomeDesc();
    nn2._connectionWeights.at(0) = 0.3f;
    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc().color(2).neuralNetwork(nn1), NodeDesc().color(3).neuralNetwork(nn2)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0)),
            ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(1)),
        },
        CreatureDesc(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    checkConnections(result.description, {{object1._pos, object2._pos, 0.5f, 0.3f}});
}
