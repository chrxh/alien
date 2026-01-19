#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class CellStateTransitionTests
    : public IntegrationTestFramework
    , public testing::WithParamInterface<CellDeathConsequences>
{
public:
    CellStateTransitionTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.cellDeathProbability.baseValue[i] = 0;
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ObjectTypeDescription getObjectTypeDescription(ObjectType objectType)
    {
        if (objectType == ObjectType_Structure) {
            return StructureDescription();
        } else if (objectType == ObjectType_FreeCell) {
            return FreeCellDescription();
        } else {
            return CellDescription().cellType(BaseDescription());
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    CellStateTransitionTests,
    CellStateTransitionTests,
    ::testing::Values(
        CellDeathConsequences_None,
        CellDeathConsequences_DetachedPartsDie,
        CellDeathConsequences_CreatureDies));

TEST_P(CellStateTransitionTests, ready_ready)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).fixed(true).type(CellDescription().cellState(CellState_Ready)),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).fixed(true).type(CellDescription().cellState(CellState_Ready)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests, ready_dying)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.addCreature(
        {
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready)),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Dying)),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests, ready_detaching)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.addCreature({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready)),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else {
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_onHeadCell)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription()}),
    });

    Description data;
    data.addCreature(
        {
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready)),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching)),
        },
        CreatureDescription(),
        genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        // Both cells go to Detaching (not Reviving for cell 1)
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_onNonHeadCell)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription()}),
    });

    Description data;
    data.addCreature(
        {
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready)),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching)),
        },
        CreatureDescription(),
        genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_differentCreature)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.addCreature({ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready))});
    data.addCreature({ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching))});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, detaching_reviving)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.addCreature(
        {
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching)),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Reviving)),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Reviving, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, underConstruction_activating)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.addCreature(
        {
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Constructing)),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Activating)),
        },
        CreatureDescription());
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // Object 1 stays at Ready (Activating state doesn't spread to connected cells)
    EXPECT_EQ(CellState_Activating, actualData.getObjectRef(1).getCellRef()._cellState);
    // Object 2 should transition to Ready state after activating
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests, noDyingForFixedCells)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().addCreature({ObjectDescription().id(1).fixed(true).pos({10.0f, 10.0f}).type(CellDescription())}, CreatureDescription());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}
