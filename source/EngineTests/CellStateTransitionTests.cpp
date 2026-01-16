#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

struct TransitionTestParameters
{
    CellDeathConsequences cellDeathConsequences = CellDeathConsequences_None;
    ObjectType objectType = ObjectType_Cell;
    CellType cellType = CellType_Base;
};

class CellStateTransitionTests
    : public IntegrationTestFramework
    , public testing::WithParamInterface<TransitionTestParameters>
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

    ObjectTypeDescription getObjectTypeDescription(ObjectType objectType, CellType cellType)
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
        TransitionTestParameters{CellDeathConsequences_None, ObjectType_Structure, CellType_Base},
        TransitionTestParameters{CellDeathConsequences_DetachedPartsDie, ObjectType_Structure, CellType_Base},
        TransitionTestParameters{CellDeathConsequences_CreatureDies, ObjectType_Structure, CellType_Base},
        
        TransitionTestParameters{CellDeathConsequences_None, ObjectType_FreeCell, CellType_Base},
        TransitionTestParameters{CellDeathConsequences_DetachedPartsDie, ObjectType_FreeCell, CellType_Base},
        TransitionTestParameters{CellDeathConsequences_CreatureDies, ObjectType_FreeCell, CellType_Base},

        TransitionTestParameters{CellDeathConsequences_None, ObjectType_Cell, CellType_Base},
        TransitionTestParameters{CellDeathConsequences_DetachedPartsDie, ObjectType_Cell, CellType_Base},
        TransitionTestParameters{CellDeathConsequences_CreatureDies, ObjectType_Cell, CellType_Base}
    ));

TEST_P(CellStateTransitionTests, ready_ready)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().objects({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).type(getObjectTypeDescription(objectType, cellType)),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    if (objectType == ObjectType_Cell) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    }
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests, ready_dying)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().objects({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).type(getObjectTypeDescription(objectType, cellType)),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Dying)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    if (objectType == ObjectType_Cell) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    }
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests, ready_detaching)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().objects({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready).cellType(getObjectTypeDescription(objectType, cellType))),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_onHeadCell)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription()}),
    });

    Description data;
    data.addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready).headCell(true).cellType(getObjectTypeDescription(objectType, cellType))),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching)),
        }, CreatureDescription(), genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Reviving, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_onNonHeadCell)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescription().genes({
        GeneDescription().separation(true).nodes({NodeDescription()}),
    });

    Description data;
    data.addCreature({
            ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready).headCell(false).cellType(getObjectTypeDescription(objectType, cellType))),
            ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching)),
        }, CreatureDescription(), genome);
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_differentCreature)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.addCreature({ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Ready).headCell(true).cellType(getObjectTypeDescription(objectType, cellType)))}, CreatureDescription(), GenomeDescription());
    data.addCreature({ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching))}, CreatureDescription(), GenomeDescription());
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
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().objects({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Detaching).cellType(getObjectTypeDescription(objectType, cellType))),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Reviving)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (deathConsequences == CellDeathConsequences_None) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (deathConsequences == CellDeathConsequences_DetachedPartsDie) {
        if (objectType == ObjectType_Cell) {
            EXPECT_EQ(CellState_Reviving, actualData.getObjectRef(1).getCellRef()._cellState);
        } else {
            EXPECT_EQ(CellState_Detaching, actualData.getObjectRef(1).getCellRef()._cellState);
        }
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests, underConstruction_activating)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().objects({
        ObjectDescription().id(1).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Constructing).cellType(getObjectTypeDescription(objectType, cellType))),
        ObjectDescription().id(2).pos({11.0f, 10.0f}).type(CellDescription().cellState(CellState_Activating)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (objectType == ObjectType_Cell) {
        EXPECT_EQ(CellState_Activating, actualData.getObjectRef(1).getCellRef()._cellState);
    } else {
        EXPECT_EQ(CellState_Constructing, actualData.getObjectRef(1).getCellRef()._cellState);
    }
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests, noDyingForBarrierCells)
{
    auto [deathConsequences, objectType, cellType] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().objects({
        ObjectDescription().id(1).fixed(true).pos({10.0f, 10.0f}).type(CellDescription().cellState(CellState_Dying).cellType(getObjectTypeDescription(objectType, cellType))),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}
