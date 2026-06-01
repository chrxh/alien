#include <algorithm>
#include <cmath>
#include <ranges>

#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

#include "PersisterInterface/DeserializedSimulation.h"
#include "PersisterInterface/SerializerService.h"

enum class Direction
{
    Forward,
    Backward
};

class CreatureTests : public IntegrationTestFramework
{
public:
    CreatureTests()
        : IntegrationTestFramework({1000, 1000})
    {
        _parameters.friction.baseValue = 0.01f;
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~CreatureTests() = default;

protected:
    GenomeDesc createGenomeForCreatureWithTwoLegs(MuscleMode const& muscleMode, Direction direction) const
    {
        auto muscleDesc = [&muscleMode, &direction] {
            if (muscleMode == MuscleMode_AutoBending) {
                return MuscleGenomeDesc().mode(AutoBendingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_ManualBending) {
                return MuscleGenomeDesc().mode(ManualBendingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_AngleBending) {
                return MuscleGenomeDesc().mode(AngleBendingGenomeDesc().attractionRepulsionRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else {
                CHECK(false);
            }
        }();
        auto generator = muscleMode == MuscleMode_AutoBending ? GeneratorGenomeDesc().mode(SquareSignalGenomeDesc().period(1)).minValue(1.0f).maxValue(1.0f)
                                                              : GeneratorGenomeDesc().mode(SquareSignalGenomeDesc().period(30 * 20)).minValue(-1.0f).maxValue(1.0f);
        return GenomeDesc().genes({
            GeneDesc().nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).autoTriggerInterval(10).numBranches(2).numConcatenations(4)),
                NodeDesc().cellType(generator),
            }),
            GeneDesc().nodes({NodeDesc().cellType(muscleDesc)}),
        });
    }

    GenomeDesc createGenomeForCreatureWithOneLegAndSpikes(MuscleMode const& muscleMode, Direction direction) const
    {
        auto muscleDesc = [&muscleMode, &direction] {
            if (muscleMode == MuscleMode_AutoBending) {
                return MuscleGenomeDesc().mode(AutoBendingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_ManualBending) {
                return MuscleGenomeDesc().mode(ManualBendingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_AngleBending) {
                return MuscleGenomeDesc().mode(AngleBendingGenomeDesc().attractionRepulsionRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else {
                CHECK(false);
            }
        }();
        auto generator = muscleMode == MuscleMode_AutoBending ? GeneratorGenomeDesc().mode(SquareSignalGenomeDesc().period(1)).minValue(1.0f).maxValue(1.0f)
                                                              : GeneratorGenomeDesc().mode(SquareSignalGenomeDesc().period(30 * 20)).minValue(-1.0f).maxValue(1.0f);
        return GenomeDesc().genes({
            GeneDesc().nodes({
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1).numBranches(1).numConcatenations(2)),
                NodeDesc().cellType(generator),
            }),
            GeneDesc().nodes(
                {NodeDesc().color(1).cellType(muscleDesc),
                 NodeDesc().color(1).constructor(ConstructorGenomeDesc().geneIndex(2).numBranches(2).numConcatenations(1))}),
            GeneDesc().nodes({NodeDesc().color(2)}),
        });
    }

    GenomeDesc createGenomeForCrawlingCreature(MuscleMode const& muscleMode, Direction direction, float frontAngle) const
    {
        auto muscleDesc = muscleMode == MuscleMode_AutoCrawling
            ? MuscleGenomeDesc().mode(AutoCrawlingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.9f : 0.1f))
            : MuscleGenomeDesc().mode(ManualCrawlingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.9f : 0.1f));
        auto generator = muscleMode == MuscleMode_AutoBending
            ? GeneratorGenomeDesc().mode(SquareSignalGenomeDesc().period(TIMESTEPS_PER_CELL_FUNCTION)).minValue(1.0f).maxValue(1.0f)
            : GeneratorGenomeDesc().mode(SquareSignalGenomeDesc().period(30 * 20 * TIMESTEPS_PER_CELL_FUNCTION)).minValue(-1.0f).maxValue(1.0f);
        return GenomeDesc()
            .frontAngle(frontAngle)
            .genes({
                GeneDesc().nodes({
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc().cellType(muscleDesc),
                    NodeDesc().cellType(muscleDesc),
                    NodeDesc(),
                    NodeDesc().cellType(generator),
                }),
            });
    }
};

class CreatureTests_BendingMuscles
    : public CreatureTests
    , public testing::WithParamInterface<MuscleMode>
{};

INSTANTIATE_TEST_SUITE_P(
    CreatureTests_BendingMuscles,
    CreatureTests_BendingMuscles,
    ::testing::Values(MuscleMode_AutoBending, MuscleMode_ManualBending, MuscleMode_AngleBending));

TEST_P(CreatureTests_BendingMuscles, constructCreatureWithTwoLegs)
{
    auto muscleMode = GetParam();

    auto genome = createGenomeForCreatureWithTwoLegs(muscleMode, Direction::Forward);
    auto data = Desc().addCreature(
        {ObjectDesc()
             .id(0)
             .pos({200.0f, 200.0f})
             .type(CellDesc().headCell(true).constructor(ConstructorDesc().provideEnergy(ProvideEnergy_Free).geneIndex(0).separation(true)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    // Check that the seed provides no energy anymore after the creature is constructed
    auto constructor = actualData.getObjectRef(0).getCellRef()._constructor.value();
    EXPECT_EQ(ProvideEnergy_FromConstructor, constructor._provideEnergy);

    DescEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(6 + 4 + 4, actualData.getObjectsForCreature(creature._id).size());

    auto cells = actualData.getObjectsForCreature(creature._id);
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    std::vector<ObjectDesc> body;
    for (int i = 0; i < 6; ++i) {
        body.emplace_back(cells.at(i));
    }
    std::vector<ObjectDesc> leg1;
    for (int i = 6; i < 6 + 4; ++i) {
        leg1.emplace_back(cells.at(i));
    }
    std::vector<ObjectDesc> leg2;
    for (int i = 6 + 4; i < 6 + 4 + 4; ++i) {
        leg2.emplace_back(cells.at(i));
    }

    // Check front angles
    if (muscleMode != MuscleMode_AngleBending) {
        for (int i = 0; i < 5; ++i) {
            EXPECT_TRUE(approxCompareAngles(0.0f, body.at(i).getCellRef()._frontAngle.value()));
        }
        EXPECT_TRUE(approxCompareAngles(-180.0f, body.at(5).getCellRef()._frontAngle.value()));

        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(approxCompareAngles(90.0f, leg1.at(i).getCellRef()._frontAngle.value()));
        }

        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(approxCompareAngles(-90.0f, leg2.at(i).getCellRef()._frontAngle.value()));
        }
    }

    // Check angles without muscle distortions
    auto getInitialAngle = [&muscleMode](ObjectDesc const& object) {
        if (muscleMode == MuscleMode_AutoBending) {
            return std::get<AutoBendingDesc>(std::get<MuscleDesc>(object.getCellRef()._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_ManualBending) {
            return std::get<ManualBendingDesc>(std::get<MuscleDesc>(object.getCellRef()._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_AngleBending) {
            return std::get<AngleBendingDesc>(std::get<MuscleDesc>(object.getCellRef()._cellType)._mode)._initialAngle.value();
        } else {
            CHECK(false);
        }
    };
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(approxCompareAngles(180.0f, getInitialAngle(leg1.at(i))));
    }
    EXPECT_TRUE(approxCompareAngles(90.0f, getInitialAngle(leg1.at(3))));

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(approxCompareAngles(180.0f, getInitialAngle(leg2.at(i))));
    }
    EXPECT_TRUE(approxCompareAngles(90.0f, getInitialAngle(leg2.at(3))));
}

TEST_P(CreatureTests_BendingMuscles, constructCreatureWithOneLegAndSpikes)
{
    auto muscleMode = GetParam();

    auto genome = createGenomeForCreatureWithOneLegAndSpikes(muscleMode, Direction::Forward);
    auto data = Desc().addCreature(
        {ObjectDesc()
             .id(0)
             .pos({200.0f, 200.0f})
             .type(CellDesc().headCell(true).constructor(ConstructorDesc().provideEnergy(ProvideEnergy_Free).geneIndex(0).separation(true)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2300);

    auto actualData = _simulationFacade->getSimulationData();

    // Check that the seed provides no energy anymore after the creature is constructed
    auto constructor = actualData.getObjectRef(0).getCellRef()._constructor.value();
    if (constructor._separation) {
        EXPECT_EQ(ProvideEnergy_FromConstructor, constructor._provideEnergy);
    }
    DescEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(6 + 4 + 4, actualData.getObjectsForCreature(creature._id).size());

    auto cells = actualData.getObjectsForCreature(creature._id);
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    std::vector<ObjectDesc> body;
    std::vector<ObjectDesc> leg;
    std::vector<ObjectDesc> spikes1;
    std::vector<ObjectDesc> spikes2;
    for (auto const& object : cells) {
        if (object._color == 0) {
            body.emplace_back(object);
        } else if (object._color == 1) {
            leg.emplace_back(object);
        }
    }
    spikes1.emplace_back(actualData.getObjectRef(leg.at(1)._connections.at(3)._objectId));
    spikes1.emplace_back(actualData.getObjectRef(leg.at(3)._connections.at(3)._objectId));
    spikes2.emplace_back(actualData.getObjectRef(leg.at(1)._connections.at(1)._objectId));
    spikes2.emplace_back(actualData.getObjectRef(leg.at(3)._connections.at(1)._objectId));


    // Check front angles
    if (muscleMode != MuscleMode_AngleBending) {
        for (int i = 0; i < 5; ++i) {
            EXPECT_TRUE(approxCompareAngles(0.0f, body.at(i).getCellRef()._frontAngle.value()));
        }
        EXPECT_TRUE(approxCompareAngles(-180.0f, body.at(5).getCellRef()._frontAngle.value()));

        for (int i = 0; i < 4; ++i) {
            EXPECT_TRUE(approxCompareAngles(90.0f, leg.at(i).getCellRef()._frontAngle.value()));
        }
        EXPECT_TRUE(approxCompareAngles(0.0f, spikes1.at(0).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(0.0f, spikes1.at(1).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(180.0f, spikes2.at(0).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(180.0f, spikes2.at(1).getCellRef()._frontAngle.value()));
    }

    // Check angles without muscle distortions
    auto getInitialAngle = [&muscleMode](ObjectDesc const& object) {
        if (muscleMode == MuscleMode_AutoBending) {
            return std::get<AutoBendingDesc>(std::get<MuscleDesc>(object.getCellRef()._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_ManualBending) {
            return std::get<ManualBendingDesc>(std::get<MuscleDesc>(object.getCellRef()._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_AngleBending) {
            return std::get<AngleBendingDesc>(std::get<MuscleDesc>(object.getCellRef()._cellType)._mode)._initialAngle.value();
        } else {
            CHECK(false);
        }
    };

    // Check angles for first cell leg
    ASSERT_EQ(1, leg.at(0)._connections.size());

    // Check angles for second cell leg
    ASSERT_EQ(4, leg.at(1)._connections.size());
    EXPECT_TRUE(approxCompareAngles(90.0f, getInitialAngle(leg.at(0))));  // initial angle connection is stored in connected muscle
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(1)._connections.at(0)._angleFromPrevious));
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(1)._connections.at(1)._angleFromPrevious));

    // Check angles for third cell leg
    ASSERT_EQ(2, leg.at(2)._connections.size());
    EXPECT_TRUE(approxCompareAngles(180.0, leg.at(2)._connections.at(0)._angleFromPrevious));

    // Check angles for forth cell leg
    ASSERT_EQ(4, leg.at(3)._connections.size());
    EXPECT_TRUE(approxCompareAngles(90.0f, getInitialAngle(leg.at(2))));  // initial angle connection is stored in connected muscle
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(3)._connections.at(0)._angleFromPrevious));
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(3)._connections.at(1)._angleFromPrevious));
}

// Regression test for issue where muscles connected to void were not constructed with the correct angles
TEST_F(CreatureTests, constructMusclesVoidsSmallCreature)
{
    auto const AutoTriggerInterval = 100;

    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().headCell(true).constructor(
                    ConstructorDesc().autoTriggerInterval(AutoTriggerInterval).provideEnergy(ProvideEnergy_Free).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc()
                .shape(ConstructorShape_Hexagon)
                .nodes({
                    NodeDesc(),
                    NodeDesc().cellType(VoidGenomeDesc()),
                    NodeDesc()
                        .cellType(MuscleGenomeDesc().mode(AutoBendingGenomeDesc()))
                        .constructor(ConstructorGenomeDesc().autoTriggerInterval(AutoTriggerInterval).geneIndex(1).separation(false)),
                    NodeDesc().cellType(VoidGenomeDesc()),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                }),
            GeneDesc().nodes({
                NodeDesc().cellType(MuscleGenomeDesc().mode(AutoBendingGenomeDesc())),
                NodeDesc().cellType(MuscleGenomeDesc().mode(AutoBendingGenomeDesc())),
                NodeDesc().cellType(MuscleGenomeDesc().mode(AutoBendingGenomeDesc())),
            }),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1500);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(9, actualData._objects.size());

    auto cellIt = std::ranges::find_if(
        actualData._objects, [](ObjectDesc const& object) { return object.getCellRef()._nodeIndex == 1 && object.getCellRef()._parentNodeIndex == 2; });
    ASSERT_NE(cellIt, actualData._objects.end());
    auto cell1 = *cellIt;
    EXPECT_TRUE(approxCompare(180.0f, cell1._connections.at(0)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(180.0f, cell1._connections.at(1)._angleFromPrevious));
}

// Regression test for issue where muscles connected to void were not constructed with the correct angles
TEST_F(CreatureTests, constructMusclesVoidsLargeCreature)
{
    auto const AutoTriggerInterval = 20;

    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().headCell(true).constructor(
                    ConstructorDesc().autoTriggerInterval(AutoTriggerInterval).provideEnergy(ProvideEnergy_Free).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc()
                .shape(ConstructorShape_Hexagon)
                .nodes({
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc().cellType(VoidGenomeDesc()),
                    NodeDesc(),
                    NodeDesc()
                        .cellType(MuscleGenomeDesc().mode(AutoBendingGenomeDesc()))
                        .constructor(ConstructorGenomeDesc().autoTriggerInterval(AutoTriggerInterval).geneIndex(1).separation(false)),
                    NodeDesc().cellType(VoidGenomeDesc()),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                }),
            GeneDesc().nodes({
                NodeDesc().cellType(MuscleGenomeDesc().mode(AutoBendingGenomeDesc())),
            }),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(500);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(19, actualData._objects.size());

    auto cellIt = std::ranges::find_if(
        actualData._objects, [](ObjectDesc const& object) { return object.getCellRef()._nodeIndex == 12 && object.getCellRef()._parentNodeIndex == 0; });
    ASSERT_NE(cellIt, actualData._objects.end());
    auto cell1 = *cellIt;
    EXPECT_TRUE(approxCompare(120.0f, cell1._connections.at(0)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, cell1._connections.at(1)._angleFromPrevious));
}

class CreatureTests_BendingMuscles_TwoDirections
    : public CreatureTests
    , public testing::WithParamInterface<std::pair<MuscleMode, Direction>>
{};

INSTANTIATE_TEST_SUITE_P(
    CreatureTests_BendingMuscles_TwoDirections,
    CreatureTests_BendingMuscles_TwoDirections,
    ::testing::Values(
        std::make_pair(MuscleMode_AutoBending, Direction::Forward),
        std::make_pair(MuscleMode_ManualBending, Direction::Forward),
        std::make_pair(MuscleMode_AutoBending, Direction::Backward),
        std::make_pair(MuscleMode_ManualBending, Direction::Backward)));

TEST_P(CreatureTests_BendingMuscles_TwoDirections, moveCreatureWithTwoLegs)
{
    auto [muscleMode, direction] = GetParam();

    auto genome = createGenomeForCreatureWithTwoLegs(muscleMode, direction);
    auto data = Desc().addCreature(
        {ObjectDesc()
             .id(0)
             .pos({500.0f, 500.0f})
             .type(
            CellDesc().headCell(true).constructor(ConstructorDesc().provideEnergy(ProvideEnergy_Free).geneIndex(0).separation(true)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->calcTimesteps(1500);

    RealVector2D movementDirection;
    RealVector2D lastPos;
    {
        auto actualData = _simulationFacade->getSimulationData();
        for (auto& object : actualData._objects) {
            object._vel = {0, 0};
        }
        _simulationFacade->setSimulationData(actualData);

        DescEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6 + 4 + 4, actualData.getObjectsForCreature(creature._id).size());

        auto cells = actualData.getObjectsForCreature(creature._id);

        auto backCellIt = std::ranges::find_if(
            actualData._objects, [](ObjectDesc const& object) { return object.getCellRef()._nodeIndex == 0 && object.getCellRef()._parentNodeIndex== 0; });

        auto frontCellIt = std::ranges::find_if(
            actualData._objects, [](ObjectDesc const& object) { return object.getCellRef()._nodeIndex == 5 && object.getCellRef()._parentNodeIndex == 0; });

        movementDirection = Math::getNormalized(frontCellIt->_pos - backCellIt->_pos);
        if (direction == Direction::Backward) {
            movementDirection = -movementDirection;
        }
        lastPos = backCellIt->_pos;
    }

    _simulationFacade->calcTimesteps(2000);
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6 + 4 + 4, actualData.getObjectsForCreature(creature._id).size());

        auto cells = actualData.getObjectsForCreature(creature._id);

        auto backCellIt = std::ranges::find_if(
            actualData._objects, [](ObjectDesc const& object) { return object.getCellRef()._nodeIndex == 0 && object.getCellRef()._parentNodeIndex == 0; });

        auto movedRefPoint = lastPos + movementDirection * 5.0f;
        EXPECT_LT(0.0, Math::dot(backCellIt->_pos - movedRefPoint, movementDirection));
    }
}

class CreatureTests_CrawlingMuscles
    : public CreatureTests
    , public testing::WithParamInterface<MuscleMode>
{};

INSTANTIATE_TEST_SUITE_P(CreatureTests_CrawlingMuscles, CreatureTests_CrawlingMuscles, ::testing::Values(MuscleMode_AutoCrawling, MuscleMode_ManualCrawling));

TEST_P(CreatureTests_CrawlingMuscles, constructCrawlingCreature)
{
    auto muscleMode = GetParam();

    auto genome = createGenomeForCrawlingCreature(muscleMode, Direction::Forward, 0.0f);
    auto data = Desc().addCreature(
        {ObjectDesc()
             .id(0)
             .pos({200.0f, 200.0f})
             .type(CellDesc().headCell(true).constructor(
                 ConstructorDesc().autoTriggerInterval(10).provideEnergy(ProvideEnergy_Free).geneIndex(0).separation(true)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(400);

    auto actualData = _simulationFacade->getSimulationData();
    DescEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(6, actualData.getObjectsForCreature(creature._id).size());

    auto cells = actualData.getObjectsForCreature(creature._id);
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    // Check front angles
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(approxCompareAngles(0.0f, cells[i].getCellRef()._frontAngle.value()));
    }
    EXPECT_TRUE(approxCompareAngles(-180.0f, cells[5].getCellRef()._frontAngle.value()));
}

class CreatureTests_CrawlingMuscles_TwoDirections_DifferentFrontAngles
    : public CreatureTests
    , public testing::WithParamInterface<std::tuple<MuscleMode, Direction, float>>
{};

INSTANTIATE_TEST_SUITE_P(
    CreatureTests_CrawlingMuscles_TwoDirections_DifferentFrontAngles,
    CreatureTests_CrawlingMuscles_TwoDirections_DifferentFrontAngles,
    ::testing::Values(
        std::make_tuple(MuscleMode_AutoCrawling, Direction::Forward, 0.0f),
        std::make_tuple(MuscleMode_ManualCrawling, Direction::Forward, 0.0f),
        std::make_tuple(MuscleMode_AutoCrawling, Direction::Backward, 0.0f),
        std::make_tuple(MuscleMode_ManualCrawling, Direction::Backward, 0.0f),
        std::make_tuple(MuscleMode_AutoCrawling, Direction::Forward, 180.0f),
        std::make_tuple(MuscleMode_ManualCrawling, Direction::Forward, 180.0f),
        std::make_tuple(MuscleMode_AutoCrawling, Direction::Backward, 180.0f),
        std::make_tuple(MuscleMode_ManualCrawling, Direction::Backward, 180.0f)));

TEST_P(CreatureTests_CrawlingMuscles_TwoDirections_DifferentFrontAngles, moveCrawlingCreature)
{
    auto [muscleMode, direction, frontAngle] = GetParam();
    RealVector2D refPoint{500.0f, 500.0f};

    auto genome = createGenomeForCrawlingCreature(muscleMode, direction, frontAngle);
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos(refPoint).type(CellDesc().headCell(true).constructor(
            ConstructorDesc().autoTriggerInterval(10).provideEnergy(ProvideEnergy_Free).geneIndex(0).separation(true)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(400);

    RealVector2D movementDirection;
    float startPos_projected = 0;
    {
        auto actualData = _simulationFacade->getSimulationData();
        for (auto& object : actualData._objects) {
            object._vel = {0, 0};
        }
        _simulationFacade->setSimulationData(actualData);

        DescEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6, actualData.getObjectsForCreature(creature._id).size());

        auto cells = actualData.getObjectsForCreature(creature._id);
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        movementDirection = Math::getNormalized(cells.back()._pos - cells.front()._pos);
        if (direction == Direction::Backward) {
            movementDirection = -movementDirection;
        }
        if (abs(frontAngle) > 90.0f) {
            movementDirection = -movementDirection;
        }

        auto movedRefPoint = refPoint + movementDirection * 10.0f;
        startPos_projected = Math::dot(cells.front()._pos - movedRefPoint, movementDirection);
    }

    _simulationFacade->calcTimesteps(1700);
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6, actualData.getObjectsForCreature(creature._id).size());

        auto cells = actualData.getObjectsForCreature(creature._id);
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        auto movedRefPoint = refPoint + movementDirection * 10.0f;
        auto endPos_projected = Math::dot(cells.front()._pos - movedRefPoint, movementDirection);
        EXPECT_TRUE(startPos_projected + NEAR_ZERO < endPos_projected);
    }
}

class CreatureTests_NumCells : public IntegrationTestFramework
{
public:
    CreatureTests_NumCells()
        : IntegrationTestFramework({1000, 1000})
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.cellDeathProbability.baseValue[i] = 1.0f;
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~CreatureTests_NumCells() = default;
};

TEST_F(CreatureTests_NumCells, numCellsUpdatedWhenOneCellDies)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().headCell(true)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().lastUpdate(2 * CELL_UPDATE_INTERVAL + 2)),
            ObjectDesc().id(3).pos({102.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc());
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    auto creature = actualData.getCreatureRef(0);
    auto actualObjects = actualData.getObjectsForCreature(creature._id);

    EXPECT_EQ(2, toInt(actualObjects.size()));
    EXPECT_EQ(2, creature._numCells);
}

TEST_F(CreatureTests_NumCells, numCellsUpdatedWhenMultipleCellsDie)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().headCell(true)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}).type(CellDesc().lastUpdate(2 * CELL_UPDATE_INTERVAL + 2)),
            ObjectDesc().id(3).pos({100.0f, 101.0f}).type(CellDesc().lastUpdate(2 * CELL_UPDATE_INTERVAL + 1)),
            ObjectDesc().id(4).pos({99.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc());
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    auto creature = actualData.getCreatureRef(0);
    auto actualObjects = actualData.getObjectsForCreature(creature._id);

    EXPECT_EQ(2, toInt(actualObjects.size()));
    EXPECT_EQ(2, creature._numCells);
}

TEST_F(CreatureTests_NumCells, numCellsUnchangedWhenNoCellDies)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc().id(1).pos({100.0f, 100.0f}).type(CellDesc().headCell(true)),
            ObjectDesc().id(2).pos({101.0f, 100.0f}),
            ObjectDesc().id(3).pos({102.0f, 100.0f}),
        },
        CreatureDesc().id(0),
        GenomeDesc());
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3);

    auto actualData = _simulationFacade->getSimulationData();
    auto creature = actualData.getCreatureRef(0);
    auto actualObjects = actualData.getObjectsForCreature(creature._id);

    EXPECT_EQ(3, toInt(actualObjects.size()));
    EXPECT_EQ(3, creature._numCells);
}

TEST_F(CreatureTests, muscleSeed)
{
    Desc data;
    data.addCreature(
        {
            ObjectDesc()
                .id(1)
                .pos({100.0f, 100.0f})
                .type(CellDesc().headCell(true).constructor(ConstructorDesc().provideEnergy(ProvideEnergy_Free).separation(true))),
        },
        CreatureDesc().id(0),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc().cellType(MuscleGenomeDesc()).constructor(ConstructorGenomeDesc().geneIndex(1).separation(false))}),
            GeneDesc().shape(ConstructorShape_Hexagon).nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(401);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(5, actualData._objects.size());

    auto cellIt = std::ranges::find_if(
        actualData._objects, [](ObjectDesc const& object) { return object.getObjectType() == ObjectType_Cell && object.getCellRef()._nodeIndex == 1; });

    ASSERT_NE(cellIt, actualData._objects.end());
    auto cell1 = *cellIt;
    EXPECT_TRUE(approxCompare(240.0f, cell1._connections.at(0)._angleFromPrevious));
    EXPECT_TRUE(approxCompare(120.0f, cell1._connections.at(1)._angleFromPrevious));
}
