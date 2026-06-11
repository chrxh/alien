#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include <Base/Definitions.h>
#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/PreviewDescConverterService.h>
#include <EngineInterface/SpaceCalculator.h>

#include <EngineTestData/TestHelper.h>

class PreviewDescConverterServiceTests : public ::testing::Test
{
public:
    PreviewDescConverterServiceTests() = default;
    virtual ~PreviewDescConverterServiceTests() = default;

    CellPreviewDesc const& getPreviewCell(PreviewDesc const& preview, int geneIndex, int nodeIndex)
    {
        for (auto const& object : preview._cells) {
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
                if (TestHelper::TestHelper::approxCompare(expectedConnection.object1, connection._cell1)
                    && TestHelper::TestHelper::approxCompare(expectedConnection.object2, connection._cell2)
                    && (!expectedConnection.connectionWeightToObject1.has_value()
                        || expectedConnection.connectionWeightToObject1.value() == connection._connectionWeightToObject1)
                    && (!expectedConnection.connectionWeightToObject2.has_value()
                        || expectedConnection.connectionWeightToObject2.value() == connection._connectionWeightToObject2)) {
                    found = true;
                    break;
                }
                if (TestHelper::TestHelper::approxCompare(expectedConnection.object2, connection._cell1)
                    && TestHelper::TestHelper::approxCompare(expectedConnection.object1, connection._cell2)
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(GenomeDesc(), 0, std::move(input));

    EXPECT_TRUE(result.description._cells.empty());
    EXPECT_TRUE(result.description._connections.empty());
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(3)}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    EXPECT_EQ(2, object1._color);
    EXPECT_EQ(3, object2._color);
    EXPECT_FALSE(object1._inactive);
    EXPECT_FALSE(object2._inactive);

    auto expectedCell1_pos = RealVector2D{0, 0.5f};
    auto expectedCell2_pos = RealVector2D{0, -0.5f};
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell1_pos, object1._pos));
    EXPECT_TRUE(TestHelper::approxCompare(expectedCell2_pos, object2._pos));
    checkConnections(result.description, {{object1._pos, object2._pos, 1.0f, 1.0f}});
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_marksConnectionInactiveForVoidCell)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().cellType(VoidGenomeDesc())}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    EXPECT_EQ(2, object1._color);
    EXPECT_FALSE(object1._inactive);
    EXPECT_TRUE(object2._inactive);
    EXPECT_TRUE(result.description._connections.at(0)._inactive);
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_homogeneCellTypeKeepsVoidCellActive)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().homogeneCellType(true).nodes({NodeDesc().color(2).cellType(MuscleGenomeDesc()), NodeDesc().cellType(VoidGenomeDesc())}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object2 = getPreviewCell(result.description, 0, 1);

    // With homogeneous cell type the void node inherits the first node's cell type and must not be grayed out.
    EXPECT_EQ(CellType_Muscle, object2._cellType);
    EXPECT_FALSE(object2._inactive);
    EXPECT_FALSE(result.description._connections.at(0)._inactive);
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_usesGenomeForCellTypes)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(3).cellType(MuscleGenomeDesc())}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    EXPECT_EQ(CellType_Base, object1._cellType);
    EXPECT_EQ(CellType_Muscle, object2._cellType);
    EXPECT_EQ(3, object2._color);
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_setsVisualFrontAngle)
{
    auto genome = GenomeDesc().frontAngle(30.0f).genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_TRUE(result.visualFrontAngle.has_value());
    EXPECT_TRUE(TestHelper::approxCompareAngles(120.0f, result.visualFrontAngle.value()));
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_smoothsVisualFrontAngle)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input), 0.0f);

    ASSERT_TRUE(result.visualFrontAngle.has_value());
    EXPECT_TRUE(TestHelper::approxCompareAngles(18.0f, result.visualFrontAngle.value()));
}

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_nonRootGene)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(1)}),
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(3)}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 1, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
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

TEST_F(PreviewDescConverterServiceTests, convertTwoCellCreature_defaultGenome)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(3)}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
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
        GeneDesc().nodes({NodeDesc().color(2), NodeDesc().color(3), NodeDesc().color(4)}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(3, result.description._cells.size());
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

TEST_F(PreviewDescConverterServiceTests, convertCreature_twoGenes_oneNode_multipleConcatenations)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2).constructor(ConstructorGenomeDesc().geneIndex(1).separation(false).numConcatenations(7))}),
        GeneDesc().nodes({NodeDesc().color(3)}),
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
            ObjectDesc()
                .id(1)
                .pos({11.0f, 10.0f})
                .type(CellDesc().geneIndex(0).nodeIndex(0).constructor(ConstructorDesc().geneIndex(1).separation(false).numConcatenations(7))),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(8, result.description._cells.size());
    ASSERT_EQ(7, result.description._connections.size());

    std::set<RealVector2D> actualPositions;
    for (const auto& object : result.description._cells) {
        actualPositions.insert(object._pos);
    }
    std::set<RealVector2D> expectedPositions = {
        {3.375f, -0.125f}, {2.375f, -0.125f}, {1.375f, -0.125f}, {0.375f, -0.125f}, {-0.625f, -0.125}, {-1.625f, -0.125}, {-2.625f, -0.125f}, {-2.625f, 0.875f}};
    ASSERT_EQ(expectedPositions.size(), actualPositions.size());
    for (auto const& [expectedPosition, actualPosition] : boost::combine(expectedPositions, actualPositions)) {
        EXPECT_TRUE(TestHelper::approxCompare(expectedPosition, actualPosition));
    }
}


TEST_F(PreviewDescConverterServiceTests, convertCreature_twoGenes_multipleNodes_multipleConcatenations)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().color(2).constructor(ConstructorGenomeDesc().geneIndex(1).separation(false).numConcatenations(4))}),
        GeneDesc().nodes({NodeDesc().color(3), NodeDesc().color(4)}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(2).pos({10.0f, 3.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(3).pos({10.0f, 4.0f}).type(CellDesc().geneIndex(1).nodeIndex(1)),
            ObjectDesc().id(4).pos({10.0f, 5.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(5).pos({10.0f, 6.0f}).type(CellDesc().geneIndex(1).nodeIndex(1)),
            ObjectDesc().id(6).pos({10.0f, 7.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(7).pos({10.0f, 8.0f}).type(CellDesc().geneIndex(1).nodeIndex(1)),
            ObjectDesc().id(8).pos({10.0f, 9.0f}).type(CellDesc().geneIndex(1).nodeIndex(0)),
            ObjectDesc().id(9).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(1)),
            ObjectDesc()
                .id(1)
                .pos({11.0f, 10.0f})
                .type(CellDesc().geneIndex(0).nodeIndex(0).constructor(ConstructorDesc().geneIndex(1).separation(false).numConcatenations(4))),
        },
        CreatureDesc(),
        genome);
    for (int i = 2; i < 9; ++i) {
        input.addConnection(i, i + 1);
    }
    input.addConnection(9, 1);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(9, result.description._cells.size());
    ASSERT_EQ(8, result.description._connections.size());

    auto constructor = getPreviewCell(result.description, 0, 0);
    EXPECT_EQ(2, constructor._color);
    EXPECT_EQ(1, constructor._constructorGeneIndex);

    int gene1Node0Count = 0;
    int gene1Node1Count = 0;
    for (auto const& object : result.description._cells) {
        if (object._geneIndex == 1 && object._nodeIndex == 0) {
            ++gene1Node0Count;
            EXPECT_EQ(3, object._color);
        }
        if (object._geneIndex == 1 && object._nodeIndex == 1) {
            ++gene1Node1Count;
            EXPECT_EQ(4, object._color);
        }
    }
    EXPECT_EQ(4, gene1Node0Count);
    EXPECT_EQ(4, gene1Node1Count);
}

TEST_F(PreviewDescConverterServiceTests, convertCastratedCreature_withSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(true))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(true))}),
    });

    Desc inputCreature1;
    inputCreature1.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0).constructor(ConstructorDesc().geneIndex(2).separation(true))),
        },
        CreatureDesc(),
        genome);

    Desc inputCreature2;
    inputCreature2.addCreature(
        {
            ObjectDesc().id(2).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(0).constructor(ConstructorDesc().geneIndex(2).separation(true))),
        },
        CreatureDesc(),
        genome);
    {
        auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(inputCreature1));

        ASSERT_EQ(1, result.description._cells.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto object1 = getPreviewCell(result.description, 0, 0);
        EXPECT_EQ(1, object1._constructorGeneIndex);
    }
    {
        auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 1, std::move(inputCreature2));

        ASSERT_EQ(1, result.description._cells.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto object1 = getPreviewCell(result.description, 1, 0);
        EXPECT_EQ(0, object1._constructorGeneIndex);
    }
}

TEST_F(PreviewDescConverterServiceTests, convertCastratedCreature_withoutSeparation)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).separation(false))}),
        GeneDesc().nodes({NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(0).separation(false))}),
    });

    Desc input;
    input.addCreature(
        {
            ObjectDesc().id(0).pos({11.0f, 10.0f}).type(CellDesc().geneIndex(0).nodeIndex(0).constructor(ConstructorDesc().geneIndex(1).separation(false))),
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().geneIndex(1).nodeIndex(0).constructor(ConstructorDesc().geneIndex(0).separation(false))),
        },
        CreatureDesc(),
        genome);
    input.addConnection(0, 1);

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    EXPECT_EQ(1, object1._constructorGeneIndex.value());

    auto object2 = getPreviewCell(result.description, 1, 0);
    EXPECT_EQ(0, object2._constructorGeneIndex.value());
}

TEST_F(PreviewDescConverterServiceTests, convertCreatureWithSignals)
{
    auto genome = GenomeDesc().genes({
        GeneDesc().nodes({NodeDesc(), NodeDesc()}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
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
        GeneDesc().nodes({NodeDesc().color(2).neuralNetwork(nn1), NodeDesc().color(3).neuralNetwork(nn2)}),
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

    auto result = PreviewDescConverterService::get().convertToPreviewDesc(genome, 0, std::move(input));

    ASSERT_EQ(2, result.description._cells.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    checkConnections(result.description, {{object1._pos, object2._pos, 0.5f, 0.3f}});
}
