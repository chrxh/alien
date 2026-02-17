#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class CellStateTransitionTests
    : public IntegrationTestFramework
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

    ObjectTypeDesc getObjectTypeDesc(ObjectType objectType)
    {
        if (objectType == ObjectType_Structure) {
            return StructureDesc();
        } else if (objectType == ObjectType_FreeCell) {
            return FreeCellDesc();
        } else {
            return CellDesc().cellType(BaseDesc());
        }
    }
};

class CellStateTransitionTests_CellDeathConsequences
    : public CellStateTransitionTests
    , public testing::WithParamInterface<CellDeathConsequences>
{};


INSTANTIATE_TEST_SUITE_P(
    CellStateTransitionTests_CellDeathConsequences,
    CellStateTransitionTests_CellDeathConsequences,
    ::testing::Values(CellDeathConsequences_None, CellDeathConsequences_DetachedPartsDie, CellDeathConsequences_CreatureDies));

TEST_P(CellStateTransitionTests_CellDeathConsequences, ready_ready)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).fixed(true).type(CellDesc().cellState(CellState_Ready)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).fixed(true).type(CellDesc().cellState(CellState_Ready)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests_CellDeathConsequences, ready_dying)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Dying)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
    EXPECT_EQ(CellState_Dying, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests_CellDeathConsequences, ready_detaching)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Detaching)),
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

TEST_P(CellStateTransitionTests_CellDeathConsequences, ready_detaching_onHeadCell)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc()}),
    });

    Desc data;
    data.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready)),
            ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Detaching)),
        },
        CreatureDesc(),
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

TEST_P(CellStateTransitionTests_CellDeathConsequences, ready_detaching_onNonHeadCell)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDesc().genes({
        GeneDesc().separation(true).nodes({NodeDesc()}),
    });

    Desc data;
    data.addCreature(
        {
            ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready)),
            ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Detaching)),
        },
        CreatureDesc(),
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

TEST_P(CellStateTransitionTests_CellDeathConsequences, ready_detaching_differentCreature)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    data.addCreature({ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Ready))});
    data.addCreature({ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Detaching))});
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

TEST_P(CellStateTransitionTests_CellDeathConsequences, detaching_reviving)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Detaching)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Reviving)),
    });
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

TEST_P(CellStateTransitionTests_CellDeathConsequences, underConstruction_activating)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    Desc data;
    data.addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(CellDesc().cellState(CellState_Constructing)),
        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(CellState_Activating)),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    // Object 1 stays at Ready (Activating state doesn't spread to connected cells)
    EXPECT_EQ(CellState_Activating, actualData.getObjectRef(1).getCellRef()._cellState);
    // Object 2 should transition to Ready state after activating
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
}

TEST_P(CellStateTransitionTests_CellDeathConsequences, noDyingForFixedCells)
{
    auto deathConsequences = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Desc().addCreature({ObjectDesc().id(1).fixed(true).pos({10.0f, 10.0f}).type(CellDesc())});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getObjectRef(1).getCellRef()._cellState);
}

class CellStateTransitionTests_CellDeathConsequences_AllStates
    : public CellStateTransitionTests
    , public testing::WithParamInterface<std::pair<CellDeathConsequences, CellState>>
{};


INSTANTIATE_TEST_SUITE_P(
    CellStateTransitionTests_CellDeathConsequences_AllStates,
    CellStateTransitionTests_CellDeathConsequences_AllStates,
    ::testing::Values(
        std::make_pair(CellDeathConsequences_None, CellState_Ready),
        std::make_pair(CellDeathConsequences_None, CellState_Constructing),
        std::make_pair(CellDeathConsequences_None, CellState_Activating),
        std::make_pair(CellDeathConsequences_None, CellState_Reviving),
        std::make_pair(CellDeathConsequences_None, CellState_Dying)));

TEST_P(CellStateTransitionTests_CellDeathConsequences_AllStates, structure_cell)
{
    auto [deathConsequences, cellState] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Desc()
                    .addObjects({
                        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(StructureDesc()),
                    })
                    .addCreature({
                        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(cellState)),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (cellState == CellState_Activating) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (cellState == CellState_Reviving) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else {
        EXPECT_EQ(cellState, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}

TEST_P(CellStateTransitionTests_CellDeathConsequences_AllStates, freeCell_cell)
{
    auto [deathConsequences, cellState] = GetParam();
    _parameters.cellDeathConsequences.value = deathConsequences;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Desc()
                    .addObjects({
                        ObjectDesc().id(1).pos({10.0f, 10.0f}).type(FreeCellDesc()),
                    })
                    .addCreature({
                        ObjectDesc().id(2).pos({11.0f, 10.0f}).type(CellDesc().cellState(cellState)),
                    })
                    .addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (cellState == CellState_Activating) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else if (cellState == CellState_Reviving) {
        EXPECT_EQ(CellState_Ready, actualData.getObjectRef(2).getCellRef()._cellState);
    } else {
        EXPECT_EQ(cellState, actualData.getObjectRef(2).getCellRef()._cellState);
    }
}
