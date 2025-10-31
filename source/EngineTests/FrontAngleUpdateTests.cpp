#include <gtest/gtest.h>

#include "EngineInterface/Description.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/NumberGenerator.h"
#include "EngineInterface/ShapeGenerator.h"
#include "EngineInterface/SimulationFacade.h"

#include "EngineTestData/DescriptionTestDataFactory.h"

#include "IntegrationTestFramework.h"

class FrontAngleUpdateTests : public IntegrationTestFramework
{
public:
    FrontAngleUpdateTests()
        : IntegrationTestFramework()
    {
    }

    ~FrontAngleUpdateTests() = default;

};

TEST_F(FrontAngleUpdateTests, noUpdate_noFrontAngleRefCell)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1)
            
            .cells({
                CellDescription().id(1).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(2).pos({10.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(3).pos({10.0f, 12.0f}).frontAngleId(InitialFrontAngleId),
            }),
    GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, creature._cells.size());

    EXPECT_FALSE(actualData.getCellRef(1)._frontAngle.has_value());
    EXPECT_FALSE(actualData.getCellRef(2)._frontAngle.has_value());
    EXPECT_FALSE(actualData.getCellRef(3)._frontAngle.has_value());
}

TEST_F(FrontAngleUpdateTests, noUpdate_equalFrontAngleId)
{
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId)
            
            .cells({
                CellDescription().id(1).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId).isFrontAngleRefCell(true),
                CellDescription().id(2).pos({10.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
            }),
    GenomeDescription().frontAngle(45.0f));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(2, creature._cells.size());

    EXPECT_FALSE(actualData.getCellRef(1)._frontAngle.has_value());
    EXPECT_FALSE(actualData.getCellRef(2)._frontAngle.has_value());
}


TEST_F(FrontAngleUpdateTests, higherFrontAngleIdLeadsToUpdate)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1)
            
            .cells({
                CellDescription().id(1).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId).isFrontAngleRefCell(true),
                CellDescription().id(2).pos({10.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(3).pos({10.0f, 12.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(4).pos({9.0f, 10.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(5).pos({8.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(6).pos({9.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(7).pos({12.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(8).pos({11.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(9).pos({11.0f, 12.0f}).frontAngleId(InitialFrontAngleId),
            }),
    GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 9);
    data.addConnection(4, 1);
    data.addConnection(5, 6);
    data.addConnection(6, 2);
    data.addConnection(7, 8);
    data.addConnection(8, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(9, creature._cells.size());

    EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(1)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(2)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(3)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getCellRef(4)._frontAngle.value()));

    EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getCellRef(5)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getCellRef(6)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getCellRef(7)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getCellRef(8)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getCellRef(9)._frontAngle.value()));
}

TEST_F(FrontAngleUpdateTests, frontAngleUpdate)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1)
            
            .cells({
                CellDescription().id(1).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId).frontAngle(7.0f).isFrontAngleRefCell(true),
                CellDescription().id(2).pos({10.0f, 11.0f}).frontAngleId(InitialFrontAngleId).frontAngle(42.0f),
                CellDescription().id(3).pos({10.0f, 12.0f}).frontAngleId(InitialFrontAngleId).frontAngle(23.0f),
            }),
    GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, creature._cells.size());

    EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(1)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(2)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(3)._frontAngle.value()));
}

TEST_F(FrontAngleUpdateTests, updateRestrictedToSameCreature)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    Description data;

    data.addCreature(CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1)
            
            .cells({
                CellDescription().id(1).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId).isFrontAngleRefCell(true),
                CellDescription().id(2).pos({10.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
            }), GenomeDescription().frontAngle(FrontAngle));

    data.addCreature(CreatureDescription()
            .id(2)
            .cells({
                CellDescription().id(3).pos({10.0f, 12.0f}).frontAngleId(InitialFrontAngleId),
            }), GenomeDescription().frontAngle(FrontAngle));

    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(2, actualData._creatures.size());

    {
        auto creature = actualData.getCreatureRef(1);
        ASSERT_EQ(2, creature._cells.size());

        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(1)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(2)._frontAngle.value()));
    }
    {
        auto creature = actualData.getCreatureRef(2);
        ASSERT_EQ(1, creature._cells.size());

        EXPECT_FALSE(actualData.getCellRef(3)._frontAngle.has_value());
    }
}

class FrontAngleUpdateTests_BendingMuscles
    : public FrontAngleUpdateTests
    , public testing::WithParamInterface<MuscleMode>
{};

INSTANTIATE_TEST_SUITE_P(
    FrontAngleUpdateTests_BendingMuscles,
    FrontAngleUpdateTests_BendingMuscles,
    ::testing::Values(MuscleMode_AutoBending, MuscleMode_ManualBending, MuscleMode_AngleBending));

TEST_P(FrontAngleUpdateTests_BendingMuscles, useInitialAngleForBendingMuscles_twoConnections)
{
    auto muscleModeType = GetParam();
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto muscleMode = [&muscleModeType] -> MuscleModeDescription {
        if (muscleModeType == MuscleMode_AutoBending)
            return AutoBendingDescription().initialAngle(180.0f);
        else if (muscleModeType == MuscleMode_ManualBending)
            return ManualBendingDescription().initialAngle(180.0f);
        else
            return AngleBendingDescription().initialAngle(180.0f);
    }();
    auto data = Description().addCreature(
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1)
            
            .cells({
                CellDescription().id(1).pos({11.0f, 10.0f}).frontAngleId(InitialFrontAngleId).isFrontAngleRefCell(true),
                CellDescription().id(2).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId).cellType(MuscleDescription().mode(muscleMode)),
                CellDescription().id(3).pos({9.0f, 10.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(4).pos({9.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
            }),
    GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(4, creature._cells.size());

    if (muscleModeType == MuscleMode_AutoBending || muscleModeType == MuscleMode_ManualBending) {
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(1)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(2)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(3)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(4)._frontAngle.value()));
    } else {
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(1)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(2)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(3)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getCellRef(4)._frontAngle.value()));
    }
}

TEST_P(FrontAngleUpdateTests_BendingMuscles, useInitialAngleForBendingMuscles_oneConnections)
{
    auto muscleModeType = GetParam();
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto muscleMode = [&muscleModeType] -> MuscleModeDescription {
        if (muscleModeType == MuscleMode_AutoBending)
            return AutoBendingDescription().initialAngle(180.0f);
        else if (muscleModeType == MuscleMode_ManualBending)
            return ManualBendingDescription().initialAngle(180.0f);
        else
            return AngleBendingDescription().initialAngle(180.0f);
    }();
    auto data = Description().addCreature(
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1)
            
            .cells({
                CellDescription().id(1).pos({11.0f, 10.0f}).frontAngleId(InitialFrontAngleId).cellType(MuscleDescription().mode(muscleMode)),
                CellDescription().id(2).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId),
                CellDescription().id(3).pos({10.0f, 11.0f}).frontAngleId(InitialFrontAngleId).isFrontAngleRefCell(true),
            }),
    GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, creature._cells.size());

    if (muscleModeType == MuscleMode_AutoBending || muscleModeType == MuscleMode_ManualBending) {
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(1)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(2)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(3)._frontAngle.value()));
    } else {
        EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getCellRef(1)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getCellRef(2)._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(3)._frontAngle.value()));
    }
}

TEST_P(FrontAngleUpdateTests_BendingMuscles, useInitialAngleForBendingMuscles_initialAngleInvalid)
{
    auto muscleModeType = GetParam();
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto muscleMode = [&muscleModeType] -> MuscleModeDescription {
        if (muscleModeType == MuscleMode_AutoBending)
            return AutoBendingDescription();
        else if (muscleModeType == MuscleMode_ManualBending)
            return ManualBendingDescription();
        else
            return AngleBendingDescription();
    }();
    auto data = Description().addCreature(
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1)
            
            .cells({
                CellDescription().id(1).pos({11.0f, 10.0f}).frontAngleId(InitialFrontAngleId).isFrontAngleRefCell(true),
                CellDescription().id(2).pos({10.0f, 10.0f}).frontAngleId(InitialFrontAngleId).cellType(MuscleDescription().mode(muscleMode)),
                CellDescription().id(3).pos({10.0f, 11.0f}).frontAngleId(InitialFrontAngleId),
            }),
    GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData._cells.size());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, creature._cells.size());

    EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getCellRef(1)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getCellRef(2)._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getCellRef(3)._frontAngle.value()));
}
