#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/ShapeGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include "IntegrationTestFramework.h"

class FrontAngleUpdateTests : public IntegrationTestFramework
{
public:
    FrontAngleUpdateTests()
        : IntegrationTestFramework()
    {}

    ~FrontAngleUpdateTests() = default;
};

TEST_F(FrontAngleUpdateTests, noUpdate_noFrontAngleRefCell)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
            {
                ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(2).pos({10.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(3).pos({10.0f, 12.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1),
        GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    EXPECT_FALSE(actualData.getObjectRef(1).getCellRef()._frontAngle.has_value());
    EXPECT_FALSE(actualData.getObjectRef(2).getCellRef()._frontAngle.has_value());
    EXPECT_FALSE(actualData.getObjectRef(3).getCellRef()._frontAngle.has_value());
}

TEST_F(FrontAngleUpdateTests, noUpdate_equalFrontAngleId)
{
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
            {
                ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).headCell(true)),
                ObjectDescription().id(2).pos({10.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId),
        GenomeDescription().frontAngle(45.0f));
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());

    EXPECT_FALSE(actualData.getObjectRef(1).getCellRef()._frontAngle.has_value());
    EXPECT_FALSE(actualData.getObjectRef(2).getCellRef()._frontAngle.has_value());
}


TEST_F(FrontAngleUpdateTests, higherFrontAngleIdLeadsToUpdate)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
            {
                ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).headCell(true)),
                ObjectDescription().id(2).pos({10.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(3).pos({10.0f, 12.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(4).pos({9.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(5).pos({8.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(6).pos({9.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(7).pos({12.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(8).pos({11.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(9).pos({11.0f, 12.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1),
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

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(9, actualData.getObjectsForCreature(creature._id).size());

    EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(3).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getObjectRef(4).getCellRef()._frontAngle.value()));

    EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getObjectRef(5).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getObjectRef(6).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getObjectRef(7).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getObjectRef(8).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getObjectRef(9).getCellRef()._frontAngle.value()));
}

TEST_F(FrontAngleUpdateTests, frontAngleUpdate)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    auto data = Description().addCreature(
            {
                ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).frontAngle(7.0f).headCell(true)),
                ObjectDescription().id(2).pos({10.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).frontAngle(42.0f)),
                ObjectDescription().id(3).pos({10.0f, 12.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).frontAngle(23.0f)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1),
        GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(3).getCellRef()._frontAngle.value()));
}

TEST_F(FrontAngleUpdateTests, updateRestrictedToSameCreature)
{
    auto const FrontAngle = 45.0f;
    auto const InitialFrontAngleId = 4;

    Description data;

    data.addCreature(
            {
                ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).headCell(true)),
                ObjectDescription().id(2).pos({10.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1),
        GenomeDescription().frontAngle(FrontAngle));

    data.addCreature(
            {
            ObjectDescription().id(3).pos({10.0f, 12.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
        },
        CreatureDescription().id(2),
        GenomeDescription().frontAngle(FrontAngle));

    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(2, actualData._creatures.size());

    {
        auto creature = actualData.getCreatureRef(1);
        ASSERT_EQ(2, actualData.getObjectsForCreature(creature._id).size());

        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
    }
    {
        auto creature = actualData.getCreatureRef(2);
        ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());

        EXPECT_FALSE(actualData.getObjectRef(3).getCellRef()._frontAngle.has_value());
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
            {
                ObjectDescription().id(1).pos({11.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).headCell(true)),
                ObjectDescription().id(2).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).cellType(MuscleDescription().mode(muscleMode))),
                ObjectDescription().id(3).pos({9.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(4).pos({9.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1),
        GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(4, actualData.getObjectsForCreature(creature._id).size());

    if (muscleModeType == MuscleMode_AutoBending || muscleModeType == MuscleMode_ManualBending) {
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(3).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(4).getCellRef()._frontAngle.value()));
    } else {
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(3).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getObjectRef(4).getCellRef()._frontAngle.value()));
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
            {
                ObjectDescription().id(1).pos({11.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).cellType(MuscleDescription().mode(muscleMode))),
                ObjectDescription().id(2).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
                ObjectDescription().id(3).pos({10.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).headCell(true)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1),
        GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    if (muscleModeType == MuscleMode_AutoBending || muscleModeType == MuscleMode_ManualBending) {
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(3).getCellRef()._frontAngle.value()));
    } else {
        EXPECT_TRUE(approxCompareAngles(FrontAngle + 90.0f, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
        EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(3).getCellRef()._frontAngle.value()));
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
            {
                ObjectDescription().id(1).pos({11.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).headCell(true)),
                ObjectDescription().id(2).pos({10.0f, 10.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId).cellType(MuscleDescription().mode(muscleMode))),
                ObjectDescription().id(3).pos({10.0f, 11.0f}).type(CellDescription().frontAngleId(InitialFrontAngleId)),
            },
        CreatureDescription()
            .id(1)
            .frontAngleId(InitialFrontAngleId + 1),
        GenomeDescription().frontAngle(FrontAngle));
    data.addConnection(1, 2);
    data.addConnection(2, 3);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(5);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData.getCreatureRef(1);
    ASSERT_EQ(3, actualData.getObjectsForCreature(creature._id).size());

    EXPECT_TRUE(approxCompareAngles(FrontAngle, actualData.getObjectRef(1).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 180.0f, actualData.getObjectRef(2).getCellRef()._frontAngle.value()));
    EXPECT_TRUE(approxCompareAngles(FrontAngle - 90.0f, actualData.getObjectRef(3).getCellRef()._frontAngle.value()));
}
