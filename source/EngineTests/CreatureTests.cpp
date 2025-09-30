#include <algorithm>
#include <cmath>
#include <ranges>

#include <gtest/gtest.h>

#include "Base/Math.h"

#include "EngineInterface/Description.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

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
    GenomeDescription createGenomeForCreatureWithLegs(MuscleMode const& muscleMode, Direction direction) const
    {
        auto muscleDesc = muscleMode == MuscleMode_AutoBending
            ? MuscleGenomeDescription().mode(AutoBendingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f))
            : MuscleGenomeDescription().mode(ManualBendingGenomeDescription().forwardBackwardRatio(direction == Direction::Forward ? 0.8f : 0.2f));
        auto generator = muscleMode == MuscleMode_AutoBending
            ? GeneratorGenomeDescription()
            : GeneratorGenomeDescription().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDescription().genes({
            GeneDescription().separation(false).nodes({
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

    GenomeDescription createGenomeForCrawlingCreature(MuscleMode const& muscleMode, Direction direction) const
    {
        auto muscleDesc = muscleMode == MuscleMode_AutoBending
            ? MuscleGenomeDescription().mode(AutoCrawlingGenomeDescription().frontBackVelRatio(direction == Direction::Forward ? 0.8f : 0.2f))
            : MuscleGenomeDescription().mode(ManualCrawlingGenomeDescription().frontBackVelRatio(direction == Direction::Forward ? 0.8f : 0.2f));
        auto generator = muscleMode == MuscleMode_AutoBending
            ? GeneratorGenomeDescription()
            : GeneratorGenomeDescription().pulseType(GeneratorPulseType_Alternation).autoTriggerInterval(15).alternationInterval(20);
        return GenomeDescription().genes({
            GeneDescription().separation(false).nodes({
                NodeDescription().cellType(generator),
                NodeDescription().cellType(muscleDesc),
                NodeDescription().cellType(muscleDesc),
                NodeDescription().cellType(muscleDesc),
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
    ::testing::Values(MuscleMode_AutoBending, MuscleMode_ManualBending));

TEST_P(CreatureTests_BendingMuscles, constructCreatureWithLegs)
{
    auto muscleMode = GetParam();

    auto genome = createGenomeForCreatureWithLegs(muscleMode, Direction::Forward);
    auto data = Description().creatures({
        CreatureDescription().genome(genome).cells({CellDescription().id(0).pos({200.0f, 200.0f}).cellType(ConstructorDescription().geneIndex(0))}),
    });

    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = (4 + 4 + 6) * _parameters.normalCellEnergy.value[0] + 10.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    auto actualData = _simulationFacade->getSimulationData();
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

    // Check angles without muscle distortions
    auto getInitialAngle = [](CellDescription const& cell) {
        auto muscle = std::get_if<MuscleDescription>(&cell._cellType);
        if (muscle->getMode() == MuscleMode_AutoBending) {
            return std::get<AutoBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
        } else if (muscle->getMode() == MuscleMode_ManualBending) {
            return std::get<ManualBendingDescription>(std::get<MuscleDescription>(cell._cellType)._mode)._initialAngle.value();
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

TEST_P(CreatureTests_BendingMuscles_TwoDirections, moveCreatureWithLegs)
{
    auto [muscleMode, direction] = GetParam();
    RealVector2D refPoint{500.0f, 500.0f};

    auto genome = createGenomeForCreatureWithLegs(muscleMode, direction);
    auto data = Description().creatures({
        CreatureDescription().genome(genome).cells({CellDescription().id(0).pos(refPoint).cellType(ConstructorDescription().geneIndex(0))}),
    });

    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = (4 + 4 + 6) * _parameters.normalCellEnergy.value[0] + 10.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(2000);

    RealVector2D moveAxis;
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(6 + 4 + 4, creature._cells.size());

        auto& cells = creature._cells;
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        moveAxis = Math::getNormalized(cells.at(5)._pos - cells.at(0)._pos);
        if (direction == Direction::Backward) {
            moveAxis = -moveAxis;
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

        auto movedRefPoint =  refPoint + moveAxis * 50.0f;
        EXPECT_LT(0.0, Math::dot(cells.front()._pos - movedRefPoint, moveAxis));
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

    auto genome = createGenomeForCrawlingCreature(muscleMode, Direction::Forward);
    auto data = Description().creatures({
        CreatureDescription().genome(genome).cells({CellDescription().id(0).pos({200.0f, 200.0f}).cellType(ConstructorDescription().geneIndex(0))}),
    });

    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = 4 * _parameters.normalCellEnergy.value[0] + 10.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(500);

    auto actualData = _simulationFacade->getSimulationData();
    DescriptionEditService::get().removeCell(actualData, 0);
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(4, creature._cells.size());

    auto& cells = creature._cells;
    std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

    // Check front angles
    EXPECT_TRUE(approxCompareAngles(0.0f, cells.at(0)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(180.0f, cells.at(1)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(180.0f, cells.at(2)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(180.0f, cells.at(3)._frontAngle.value()));
}

class CreatureTests_Crawlinguscles_TwoDirections
    : public CreatureTests
    , public testing::WithParamInterface<std::pair<MuscleMode, Direction>>
{};

INSTANTIATE_TEST_SUITE_P(
    CreatureTests_Crawlinguscles_TwoDirections,
    CreatureTests_Crawlinguscles_TwoDirections,
    ::testing::Values(
        std::make_pair(MuscleMode_AutoCrawling, Direction::Forward),
        std::make_pair(MuscleMode_ManualCrawling, Direction::Forward),
        std::make_pair(MuscleMode_AutoCrawling, Direction::Backward),
        std::make_pair(MuscleMode_ManualCrawling, Direction::Backward)));

TEST_P(CreatureTests_Crawlinguscles_TwoDirections, moveCrawlingCreature)
{
    auto [muscleMode, direction] = GetParam();
    RealVector2D refPoint{500.0f, 500.0f};

    auto genome = createGenomeForCrawlingCreature(muscleMode, direction);
    auto data = Description().creatures({
        CreatureDescription().genome(genome).cells({CellDescription().id(0).pos(refPoint).cellType(ConstructorDescription().geneIndex(0))}),
    });

    _parameters.externalEnergyControlToggle.value = true;
    _parameters.externalEnergy.value = 4 * _parameters.normalCellEnergy.value[0] + 10.0f;
    _simulationFacade->setSimulationParameters(_parameters);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(500);

    RealVector2D moveAxis;
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(4, creature._cells.size());

        auto& cells = creature._cells;
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        moveAxis = Math::getNormalized(cells.at(3)._pos - cells.at(0)._pos);
        if (direction == Direction::Backward) {
            moveAxis = -moveAxis;
        }
    }

    _simulationFacade->calcTimesteps(4000);
    {
        auto actualData = _simulationFacade->getSimulationData();
        DescriptionEditService::get().removeCell(actualData, 0);
        ASSERT_EQ(1, actualData._creatures.size());

        auto creature = actualData._creatures.front();
        ASSERT_EQ(4, creature._cells.size());

        auto& cells = creature._cells;
        std::ranges::sort(cells, [](auto const& left, auto const& right) { return left._id < right._id; });

        auto movedRefPoint = refPoint + moveAxis * 50.0f;
        EXPECT_LT(0.0, Math::dot(cells.front()._pos - movedRefPoint, moveAxis));
    }
}
