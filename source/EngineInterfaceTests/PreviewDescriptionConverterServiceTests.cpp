#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include <Base/Definitions.h>
#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/PreviewDescriptionConverterService.h>
#include <EngineInterface/SpaceCalculator.h>

#include <EngineTestData/TestHelper.h>

class PreviewDescriptionConverterServiceTests : public ::testing::Test
{
public:
    PreviewDescriptionConverterServiceTests() = default;
    virtual ~PreviewDescriptionConverterServiceTests() = default;

    CellPreviewDescription const& getPreviewCell(PreviewDescription const& preview, int geneIndex, int nodeIndex)
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
        bool arrowToCell1 = true;
        bool arrowToCell2 = true;
    };
    void checkConnections(PreviewDescription const& preview, std::vector<ConnectionCheckDescription> const& expectedConnections)
    {
        for (auto const& expectedConnection : expectedConnections) {
            auto found = false;
            for (auto const& connection : preview._connections) {
                if (TestHelper::TestHelper::approxCompare(expectedConnection.object1, connection._cell1)
                    && TestHelper::TestHelper::approxCompare(expectedConnection.object2, connection._cell2)
                    && expectedConnection.arrowToCell1 == connection._arrowToCell1 && expectedConnection.arrowToCell2 == connection._arrowToCell2) {
                    found = true;
                    break;
                }
                if (TestHelper::TestHelper::approxCompare(expectedConnection.object2, connection._cell1)
                    && TestHelper::TestHelper::approxCompare(expectedConnection.object1, connection._cell2)
                    && expectedConnection.arrowToCell2 == connection._arrowToCell1 && expectedConnection.arrowToCell1 == connection._arrowToCell2) {
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

TEST_F(PreviewDescriptionConverterServiceTests, convertEmptyCollection)
{
    Description input;

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(GenomeDescription(), 0, std::move(input), std::nullopt);

    EXPECT_TRUE(result.description._objects.empty());
    EXPECT_TRUE(result.description._connections.empty());
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_withSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        },
        CreatureDescription(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

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
    checkConnections(result.description, {{object1._pos, object2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_withSeparation_nonRootGene)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().color(1)}),
        GeneDescription().separation(true).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).geneIndex(1).nodeIndex(1),
        },
        CreatureDescription(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 1, std::move(input), std::nullopt);

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
    checkConnections(result.description, {{object1._pos, object2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_withoutSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        },
        CreatureDescription(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

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
    checkConnections(result.description, {{object1._pos, object2._pos}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertTwoCellCreature_separated_withSignalRestrictions)
{
    auto constexpr BaseAngle = 10.0f;
    auto constexpr OpeningAngle = 90.0f;

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({
            NodeDescription().color(2).signalRestriction(SignalRestrictionGenomeDescription().mode(SignalRestrictionMode_Active).baseAngle(BaseAngle).openingAngle(OpeningAngle)),
            NodeDescription().color(3),
        }),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        },
        CreatureDescription(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);
    ASSERT_TRUE(object1._signalRestriction.has_value());
    EXPECT_TRUE(
        std::abs(Math::getNormalizedAngle(Math::subtractAngle(180.0f + BaseAngle - OpeningAngle / 2, object1._signalRestriction->_startAngle), -180.0f))
        < NEAR_ZERO);
    EXPECT_TRUE(
        std::abs(Math::getNormalizedAngle(Math::subtractAngle(180.0f + BaseAngle + OpeningAngle / 2, object1._signalRestriction->_endAngle), -180.0f))
        < NEAR_ZERO);
    checkConnections(result.description, {{object1._pos, object2._pos, true, false}});
}

TEST_F(PreviewDescriptionConverterServiceTests, convertThreeCellCreature)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().color(2), NodeDescription().color(3), NodeDescription().color(4)}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(2).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        ObjectDescription().id(3).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(2),
        },
        CreatureDescription(),
        genome);
    input.addConnection(1, 2);
    input.addConnection(2, 3);
    input.addConnection(3, 1);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

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

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_oneGene_multipleNodes_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(4).nodes({NodeDescription().color(2), NodeDescription().color(3)}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 4.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(2).pos({10.0f, 5.0f}).geneIndex(0).nodeIndex(1),
        ObjectDescription().id(3).pos({10.0f, 6.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(4).pos({10.0f, 7.0f}).geneIndex(0).nodeIndex(1),
        ObjectDescription().id(5).pos({10.0f, 8.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(6).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(1),
        ObjectDescription().id(7).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(8).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        },
        CreatureDescription(),
        genome);
    for (int i = 0; i < 7; ++i) {
        input.addConnection(i + 1, i + 2);
    }

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

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

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_oneGene_oneNode_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(8).nodes({NodeDescription().color(2)}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 4.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(2).pos({10.0f, 5.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(3).pos({10.0f, 6.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(4).pos({10.0f, 7.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(5).pos({10.0f, 8.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(6).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(7).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(8).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        },
        CreatureDescription(),
        genome);
    for (int i = 0; i < 7; ++i) {
        input.addConnection(i + 1, i + 2);
    }

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

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

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_twoGenes_oneNode_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription().color(2).cellType(ConstructorGenomeDescription().geneIndex(1))}),
        GeneDescription().separation(false).numConcatenations(7).nodes({NodeDescription().color(3)}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(2).pos({10.0f, 4.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(3).pos({10.0f, 5.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(4).pos({10.0f, 6.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(5).pos({10.0f, 7.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(6).pos({10.0f, 8.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(7).pos({10.0f, 9.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(8).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0),
        ObjectDescription().id(1).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(0).cellType(ConstructorDescription().geneIndex(1)),
        },
        CreatureDescription(),
        genome);
    input.addConnection(2, 3);
    input.addConnection(3, 4);
    input.addConnection(4, 5);
    input.addConnection(5, 6);
    input.addConnection(6, 7);
    input.addConnection(7, 8);
    input.addConnection(8, 1);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

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

TEST_F(PreviewDescriptionConverterServiceTests, convertCreature_oneGenes_twoNode_multipleConcatenations)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).numConcatenations(4).nodes({NodeDescription(), NodeDescription()}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 4.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(2).pos({10.0f, 5.0f}).geneIndex(0).nodeIndex(1),
        ObjectDescription().id(3).pos({10.0f, 6.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(4).pos({10.0f, 7.0f}).geneIndex(0).nodeIndex(1),
        ObjectDescription().id(5).pos({10.0f, 8.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(6).pos({10.0f, 9.0f}).geneIndex(0).nodeIndex(1),
        ObjectDescription().id(7).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0),
        ObjectDescription().id(8).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(1),
        },
        CreatureDescription(),
        genome);
    input.addConnection(1, 2);
    input.addConnection(2, 3);
    input.addConnection(3, 4);
    input.addConnection(4, 5);
    input.addConnection(5, 6);
    input.addConnection(6, 7);
    input.addConnection(7, 8);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

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

TEST_F(PreviewDescriptionConverterServiceTests, convertCastratedCreature_withSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1))}),
        GeneDescription().separation(true).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0))}),
    });

    Description inputCreature1;
    inputCreature1.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0).cellType(ConstructorDescription().geneIndex(2)),
        },
        CreatureDescription(),
        genome);

    Description inputCreature2;
    inputCreature2.addCreature({
        ObjectDescription().id(2).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0).cellType(ConstructorDescription().geneIndex(2)),
        },
        CreatureDescription(),
        genome);
    {
        auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(inputCreature1), std::nullopt);

        ASSERT_EQ(1, result.description._objects.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto object1 = getPreviewCell(result.description, 0, 0);
        EXPECT_EQ(1, object1._constructorGeneIndex);
    }
    {
        auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 1, std::move(inputCreature2), std::nullopt);

        ASSERT_EQ(1, result.description._objects.size());
        ASSERT_EQ(0, result.description._connections.size());

        auto object1 = getPreviewCell(result.description, 1, 0);
        EXPECT_EQ(0, object1._constructorGeneIndex);
    }
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCastratedCreature_withoutSeparation)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(false).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1))}),
        GeneDescription().separation(false).nodes({NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(0))}),
    });

    Description input;
    input.addCreature({
        ObjectDescription().id(0).pos({11.0f, 10.0f}).geneIndex(0).nodeIndex(0).cellType(ConstructorDescription().geneIndex(1)),
        ObjectDescription().id(1).pos({10.0f, 10.0f}).geneIndex(1).nodeIndex(0).cellType(ConstructorDescription().geneIndex(0)),
        },
        CreatureDescription(),
        genome);
    input.addConnection(0, 1);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    EXPECT_EQ(1, object1._constructorGeneIndex.value());

    auto object2 = getPreviewCell(result.description, 1, 0);
    EXPECT_EQ(0, object2._constructorGeneIndex.value());
}

TEST_F(PreviewDescriptionConverterServiceTests, convertCreatureWithSignals)
{
    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription(), NodeDescription()}),
    });

    std::vector<float> signal{0.2f, 0.2f, 0.2f, 0.8f, 0.2f, -1.2f, 0.2f, -0.2f};

    Description input;
    input.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(0).signalAndState(signal),
        ObjectDescription().id(2).pos({10.0f, 10.0f}).geneIndex(0).nodeIndex(1).signalState(SignalState_Fading),
        },
        CreatureDescription(),
        genome);
    input.addConnection(1, 2);

    auto result = PreviewDescriptionConverterService::get().convertToPreviewDescription(genome, 0, std::move(input), std::nullopt);

    ASSERT_EQ(2, result.description._objects.size());
    ASSERT_EQ(1, result.description._connections.size());

    auto object1 = getPreviewCell(result.description, 0, 0);
    auto object2 = getPreviewCell(result.description, 0, 1);

    EXPECT_EQ(SignalState_Active, object1._signalState);
    EXPECT_TRUE(object1._signalState == SignalState_Active);
    EXPECT_EQ(signal, object1._signal->_channels);
    EXPECT_EQ(SignalState_Fading, object2._signalState);
}
