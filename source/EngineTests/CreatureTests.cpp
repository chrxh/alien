#include <algorithm>
#include <cmath>
#include <ranges>

#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

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
        auto generator = muscleMode == MuscleMode_AutoBending
            ? GeneratorGenomeDesc().autoTriggerInterval(15)
            : GeneratorGenomeDesc().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDesc().genes({
            GeneDesc().separation(true).nodes({
                NodeDesc().cellType(generator),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
                NodeDesc(),
            }),
            GeneDesc().numConcatenations(4).numBranches(2).nodes({NodeDesc().cellType(muscleDesc)}),
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
        auto generator = muscleMode == MuscleMode_AutoBending
            ? GeneratorGenomeDesc().autoTriggerInterval(15)
            : GeneratorGenomeDesc().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDesc().genes({
            GeneDesc().separation(true).nodes({
                NodeDesc().cellType(generator),
                NodeDesc(),
                NodeDesc(),
                NodeDesc(),
                NodeDesc().constructor(ConstructorGenomeDesc().geneIndex(1)),
                NodeDesc(),
            }),
            GeneDesc().numConcatenations(2).numBranches(1).nodes(
                {NodeDesc().color(1).cellType(muscleDesc), NodeDesc().color(1).constructor(ConstructorGenomeDesc().geneIndex(2))}),
            GeneDesc().numConcatenations(1).numBranches(2).nodes({NodeDesc().color(2)}),
        });
    }

    GenomeDesc createGenomeForCrawlingCreature(MuscleMode const& muscleMode, Direction direction, float frontAngle) const
    {
        auto muscleDesc = muscleMode == MuscleMode_AutoCrawling
            ? MuscleGenomeDesc().mode(AutoCrawlingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.9f : 0.1f))
            : MuscleGenomeDesc().mode(ManualCrawlingGenomeDesc().forwardBackwardRatio(direction == Direction::Forward ? 0.9f : 0.1f));
        auto generator = muscleMode == MuscleMode_AutoCrawling
            ? GeneratorGenomeDesc()
            : GeneratorGenomeDesc().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDesc()
            .frontAngle(frontAngle)
            .genes({
                GeneDesc().separation(false).nodes({
                    NodeDesc().cellType(generator),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc().cellType(muscleDesc),
                    NodeDesc().cellType(muscleDesc),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
                    NodeDesc(),
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
        {ObjectDesc().id(0).pos({200.0f, 200.0f}).type(CellDesc().constructor(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    // Check that the seed provides no energy anymore after the creature is constructed
    auto constructor = actualData.getObjectRef(0).getCellRef()._constructor.value();
    EXPECT_EQ(ProvideEnergy_CellOnly, constructor._provideEnergy);

    DescriptionEditService::get().removeCell(actualData, 0);
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
        EXPECT_TRUE(approxCompareAngles(0.0f, body.at(0).getCellRef()._frontAngle.value()));
        for (int i = 1; i < 6; ++i) {
            EXPECT_TRUE(approxCompareAngles(180.0f, body.at(i).getCellRef()._frontAngle.value()));
        }

        EXPECT_TRUE(approxCompareAngles(-90.0f, leg1.at(0).getCellRef()._frontAngle.value()));
        for (int i = 1; i < 4; ++i) {
            EXPECT_TRUE(approxCompareAngles(90.0f, leg1.at(i).getCellRef()._frontAngle.value()));
        }

        EXPECT_TRUE(approxCompareAngles(90.0f, leg2.at(0).getCellRef()._frontAngle.value()));
        for (int i = 1; i < 4; ++i) {
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
        {ObjectDesc().id(0).pos({200.0f, 200.0f}).type(CellDesc().constructor(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2300);

    auto actualData = _simulationFacade->getSimulationData();

    // Check that the seed provides no energy anymore after the creature is constructed
    auto constructor = actualData.getObjectRef(0).getCellRef()._constructor.value();
    if (genome._genes[constructor._geneIndex]._separation) {
        EXPECT_EQ(ProvideEnergy_CellOnly, constructor._provideEnergy);
    }
    DescriptionEditService::get().removeCell(actualData, 0);
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
        EXPECT_TRUE(approxCompareAngles(0.0f, body.at(0).getCellRef()._frontAngle.value()));
        for (int i = 1; i < 6; ++i) {
            EXPECT_TRUE(approxCompareAngles(180.0f, body.at(i).getCellRef()._frontAngle.value()));
        }

        EXPECT_TRUE(approxCompareAngles(-90.0f, leg.at(0).getCellRef()._frontAngle.value()));
        for (int i = 1; i < 4; ++i) {
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
    EXPECT_TRUE(approxCompareAngles(90.0f, getInitialAngle(leg.at(0))));    // initial angle connection is stored in connected muscle
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(1)._connections.at(2)._angleFromPrevious));
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(1)._connections.at(3)._angleFromPrevious));

    // Check angles for third cell leg
    ASSERT_EQ(2, leg.at(2)._connections.size());
    EXPECT_TRUE(approxCompareAngles(180.0, leg.at(2)._connections.at(0)._angleFromPrevious));

    // Check angles for forth cell leg
    ASSERT_EQ(4, leg.at(3)._connections.size());
    EXPECT_TRUE(approxCompareAngles(90.0f, getInitialAngle(leg.at(2))));  // initial angle connection is stored in connected muscle
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(3)._connections.at(2)._angleFromPrevious));
    EXPECT_TRUE(approxCompareAngles(90.0, leg.at(3)._connections.at(3)._angleFromPrevious));
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
    RealVector2D refPoint{500.0f, 500.0f};

    auto genome = createGenomeForCreatureWithTwoLegs(muscleMode, direction);
    auto data = Desc().addCreature(
        {ObjectDesc().id(0).pos(refPoint).type(CellDesc().constructor(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3000);

    RealVector2D movementDirection;
    {
        auto actualData = _simulationFacade->getSimulationData();
        for (auto& object : actualData._objects) { object._vel = {0, 0}; }
        _simulationFacade->setSimulationData(actualData);

        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6 + 4 + 4, actualData.getObjectsForCreature(creature._id).size());

        auto cells = actualData.getObjectsForCreature(creature._id);
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        movementDirection = Math::getNormalized(cells.at(5)._pos - cells.at(0)._pos);
        if (direction == Direction::Backward) {
            movementDirection = -movementDirection;
        }
    }

    _simulationFacade->calcTimesteps(4000);
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6 + 4 + 4, actualData.getObjectsForCreature(creature._id).size());

        auto cells = actualData.getObjectsForCreature(creature._id);
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        auto movedRefPoint = refPoint + movementDirection * 30.0f;
        EXPECT_LT(0.0, Math::dot(cells.front()._pos - movedRefPoint, movementDirection));
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
        {ObjectDesc().id(0).pos({200.0f, 200.0f}).type(CellDesc().constructor(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1300);

    auto actualData = _simulationFacade->getSimulationData();
    DescriptionEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(10, actualData.getObjectsForCreature(creature._id).size());

    auto cells = actualData.getObjectsForCreature(creature._id);
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    // Check front angles
    auto first = true;
    for (auto const& object : cells) {
        if (first) {
            EXPECT_TRUE(approxCompareAngles(0.0f, object.getCellRef()._frontAngle.value()));
        } else {
            EXPECT_TRUE(approxCompareAngles(180.0f, object.getCellRef()._frontAngle.value()));
        }
        first = false;
    }
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
        {ObjectDesc().id(0).pos(refPoint).type(CellDesc().constructor(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0)))},
        CreatureDesc(),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1300);

    RealVector2D movementDirection;
    float startPos_projected = 0;
    {
        auto actualData = _simulationFacade->getSimulationData();
        for (auto& object : actualData._objects) { object._vel = {0, 0}; }
        _simulationFacade->setSimulationData(actualData);

        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(10, actualData.getObjectsForCreature(creature._id).size());

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

    _simulationFacade->calcTimesteps(1000);
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(10, actualData.getObjectsForCreature(creature._id).size());

        auto cells = actualData.getObjectsForCreature(creature._id);
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        auto movedRefPoint = refPoint + movementDirection * 10.0f;
        auto endPos_projected = Math::dot(cells.front()._pos - movedRefPoint, movementDirection);
        EXPECT_TRUE(startPos_projected + NEAR_ZERO < endPos_projected);
    }
}
