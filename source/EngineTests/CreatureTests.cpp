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
    GenomeDescription createGenomeForCreatureWithTwoLegs(MuscleMode const& muscleMode, Direction direction) const
    {
        auto muscleDesc = [&muscleMode, &direction] {
            if (muscleMode == MuscleMode_AutoBending) {
                return MuscleGenomeDescription().mode(AutoBendingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_ManualBending) {
                return MuscleGenomeDescription().mode(ManualBendingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_AngleBending) {
                return MuscleGenomeDescription().mode(AngleBendingGenomeDescription().attractionRepulsionRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else {
                CHECK(false);
            }
        }();
        auto generator = muscleMode == MuscleMode_AutoBending
            ? GeneratorGenomeDescription().autoTriggerInterval(15)
            : GeneratorGenomeDescription().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellType(generator),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription(),
            }),
            GeneDescription().numConcatenations(4).numBranches(2).nodes({NodeDescription().cellType(muscleDesc)}),
        });
    }

    GenomeDescription createGenomeForCreatureWithOneLegAndSpikes(MuscleMode const& muscleMode, Direction direction) const
    {
        auto muscleDesc = [&muscleMode, &direction] {
            if (muscleMode == MuscleMode_AutoBending) {
                return MuscleGenomeDescription().mode(AutoBendingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_ManualBending) {
                return MuscleGenomeDescription().mode(ManualBendingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else if (muscleMode == MuscleMode_AngleBending) {
                return MuscleGenomeDescription().mode(AngleBendingGenomeDescription().attractionRepulsionRatio(direction == Direction::Forward ? 0.8f : 0.2f));
            } else {
                CHECK(false);
            }
        }();
        auto generator = muscleMode == MuscleMode_AutoBending
            ? GeneratorGenomeDescription().autoTriggerInterval(15)
            : GeneratorGenomeDescription().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDescription().genes({
            GeneDescription().separation(true).nodes({
                NodeDescription().cellType(generator),
                NodeDescription(),
                NodeDescription(),
                NodeDescription(),
                NodeDescription().cellType(ConstructorGenomeDescription().geneIndex(1)),
                NodeDescription(),
            }),
            GeneDescription().numConcatenations(2).numBranches(1).nodes(
                {NodeDescription().color(1).cellType(muscleDesc), NodeDescription().color(1).cellType(ConstructorGenomeDescription().geneIndex(2))}),
            GeneDescription().numConcatenations(1).numBranches(2).nodes({NodeDescription().color(2)}),
        });
    }

    GenomeDescription createGenomeForCrawlingCreature(MuscleMode const& muscleMode, Direction direction, float frontAngle) const
    {
        auto muscleDesc = muscleMode == MuscleMode_AutoCrawling
            ? MuscleGenomeDescription().mode(AutoCrawlingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.9f : 0.1f))
            : MuscleGenomeDescription().mode(ManualCrawlingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.9f : 0.1f));
        auto generator = muscleMode == MuscleMode_AutoCrawling
            ? GeneratorGenomeDescription()
            : GeneratorGenomeDescription().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDescription()
            .frontAngle(frontAngle)
            .genes({
                GeneDescription().separation(false).nodes({
                    NodeDescription().cellType(generator),
                    NodeDescription(),
                    NodeDescription(),
                    NodeDescription(),
                    NodeDescription().cellType(muscleDesc),
                    NodeDescription().cellType(muscleDesc),
                    NodeDescription(),
                    NodeDescription(),
                    NodeDescription(),
                    NodeDescription(),
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
    auto data = Description().addCreature(
        CreatureDescription().cells(
            {CellDescription().id(0).pos({200.0f, 200.0f}).cellType(ConstructorDescription().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0))}),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();

    // Check that the seed provides no energy anymore after the creature is constructed
    auto constructor = std::get<ConstructorDescription>(actualData.getCellRef(0)._cellType);
    if (genome._genes[constructor._geneIndex]._separation) {
        EXPECT_EQ(ProvideEnergy_CellOnly, constructor._provideEnergy);
    }
    DescriptionEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(6 + 4 + 4, creature._cells.size());

    auto& cells = creature._cells;
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    std::vector<CellDescription> body;
    for (int i = 0; i < 6; ++i) {
        body.emplace_back(cells.at(i));
    }
    std::vector<CellDescription> leg1;
    for (int i = 6; i < 6 + 4; ++i) {
        leg1.emplace_back(cells.at(i));
    }
    std::vector<CellDescription> leg2;
    for (int i = 6 + 4; i < 6 + 4 + 4; ++i) {
        leg2.emplace_back(cells.at(i));
    }

    // Check front angles
    if (muscleMode != MuscleMode_AngleBending) {
        EXPECT_TRUE(approxCompareAngles(0.0f, body.at(0)._frontAngle.value()));
        for (int i = 1; i < 6; ++i) {
            EXPECT_TRUE(approxCompareAngles(180.0f, body.at(i)._frontAngle.value()));
        }

        EXPECT_TRUE(approxCompareAngles(-90.0f, leg1.at(0)._frontAngle.value()));
        for (int i = 1; i < 4; ++i) {
            EXPECT_TRUE(approxCompareAngles(90.0f, leg1.at(i)._frontAngle.value()));
        }

        EXPECT_TRUE(approxCompareAngles(90.0f, leg2.at(0)._frontAngle.value()));
        for (int i = 1; i < 4; ++i) {
            EXPECT_TRUE(approxCompareAngles(-90.0f, leg2.at(i)._frontAngle.value()));
        }
    }

    // Check angles without muscle distortions
    auto getInitialAngle = [&muscleMode](CellDescription const& cell) {
        if (muscleMode == MuscleMode_AutoBending) {
            return std::get<AutoBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_ManualBending) {
            return std::get<ManualBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_AngleBending) {
            return std::get<AngleBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
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
    auto data = Description().addCreature(
        CreatureDescription().cells(
            {CellDescription().id(0).pos({200.0f, 200.0f}).cellType(ConstructorDescription().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0))}),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2300);

    auto actualData = _simulationFacade->getSimulationData();

    // Check that the seed provides no energy anymore after the creature is constructed
    auto constructor = std::get<ConstructorDescription>(actualData.getCellRef(0)._cellType);
    if (genome._genes[constructor._geneIndex]._separation) {
        EXPECT_EQ(ProvideEnergy_CellOnly, constructor._provideEnergy);
    }
    DescriptionEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(6 + 4 + 4, creature._cells.size());

    auto& cells = creature._cells;
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    std::vector<CellDescription> body;
    std::vector<CellDescription> leg;
    std::vector<CellDescription> spikes1;
    std::vector<CellDescription> spikes2;
    for (auto const& cell : cells) {
        if (cell._color == 0) {
            body.emplace_back(cell);
        } else if (cell._color == 1) {
            leg.emplace_back(cell);
        }
    }
    spikes1.emplace_back(actualData.getCellRef(leg.at(1)._connections.at(3)._cellId));
    spikes1.emplace_back(actualData.getCellRef(leg.at(3)._connections.at(3)._cellId));
    spikes2.emplace_back(actualData.getCellRef(leg.at(1)._connections.at(1)._cellId));
    spikes2.emplace_back(actualData.getCellRef(leg.at(3)._connections.at(1)._cellId));


    // Check front angles
    if (muscleMode != MuscleMode_AngleBending) {
        EXPECT_TRUE(approxCompareAngles(0.0f, body.at(0)._frontAngle.value()));
        for (int i = 1; i < 6; ++i) {
            EXPECT_TRUE(approxCompareAngles(180.0f, body.at(i)._frontAngle.value()));
        }

        EXPECT_TRUE(approxCompareAngles(-90.0f, leg.at(0)._frontAngle.value()));
        for (int i = 1; i < 4; ++i) {
            EXPECT_TRUE(approxCompareAngles(90.0f, leg.at(i)._frontAngle.value()));
        }
        EXPECT_TRUE(approxCompareAngles(0.0f, spikes1.at(0)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(0.0f, spikes1.at(1)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(180.0f, spikes2.at(0)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(180.0f, spikes2.at(1)._frontAngle.value()));
    }

    // Check angles without muscle distortions
    auto getInitialAngle = [&muscleMode](CellDescription const& cell) {
        if (muscleMode == MuscleMode_AutoBending) {
            return std::get<AutoBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_ManualBending) {
            return std::get<ManualBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
        } else if (muscleMode == MuscleMode_AngleBending) {
            return std::get<AngleBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
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
    auto data = Description().addCreature(

        CreatureDescription().cells(
            {CellDescription().id(0).pos(refPoint).cellType(ConstructorDescription().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0))}),

        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(3000);

    RealVector2D movementDirection;
    {
        auto actualData = _simulationFacade->getSimulationData();
        actualData.forEachCell([](CellDescription& cell) { cell._vel = {0, 0}; });
        _simulationFacade->setSimulationData(actualData);

        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6 + 4 + 4, creature._cells.size());

        auto& cells = creature._cells;
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
        ASSERT_EQ(6 + 4 + 4, creature._cells.size());

        auto& cells = creature._cells;
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
    auto data = Description().addCreature(
        CreatureDescription().cells(
            {CellDescription().id(0).pos({200.0f, 200.0f}).cellType(ConstructorDescription().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0))}),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1300);

    auto actualData = _simulationFacade->getSimulationData();
    DescriptionEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(10, creature._cells.size());

    auto& cells = creature._cells;
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    // Check front angles
    auto first = true;
    for (auto const& cell : cells) {
        if (first) {
            EXPECT_TRUE(approxCompareAngles(0.0f, cell._frontAngle.value()));
        } else {
            EXPECT_TRUE(approxCompareAngles(180.0f, cell._frontAngle.value()));
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
    auto data = Description().addCreature(
        CreatureDescription().cells(
            {CellDescription().id(0).pos(refPoint).cellType(ConstructorDescription().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(0))}),
        genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1300);

    RealVector2D movementDirection;
    {
        auto actualData = _simulationFacade->getSimulationData();
        actualData.forEachCell([](CellDescription& cell) { cell._vel = {0, 0}; });
        _simulationFacade->setSimulationData(actualData);

        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(10, creature._cells.size());

        auto& cells = creature._cells;
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        movementDirection = Math::getNormalized(cells.back()._pos - cells.front()._pos);
        if (direction == Direction::Backward) {
            movementDirection = -movementDirection;
        }
        if (abs(frontAngle) > 90.0f) {
            movementDirection = -movementDirection;
        }
    }

    _simulationFacade->calcTimesteps(4000);
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(10, creature._cells.size());

        auto& cells = creature._cells;
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        auto movedRefPoint = refPoint + movementDirection * 10.0f;
        EXPECT_LT(0.0, Math::dot(cells.front()._pos - movedRefPoint, movementDirection));
    }
}
