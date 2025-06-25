#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class CellStateTransitionTests_New
    : public IntegrationTestFramework
    , public testing::WithParamInterface<CellDeathConsquences>
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction.value = 0;
        result.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.radiationType1_strength.baseValue[i] = 0;
        }
        return result;
    }

    CellStateTransitionTests_New()
        : IntegrationTestFramework(getParameters())
    {}

    ~CellStateTransitionTests_New() = default;
};

INSTANTIATE_TEST_SUITE_P(
    CellStateTransitionTests,
    CellStateTransitionTests_New,
    ::testing::Values(CellDeathConsquences_None, CellDeathConsquences_DetachedPartsDie, CellDeathConsquences_CreatureDies));

TEST_P(CellStateTransitionTests_New, ready_ready)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Ready),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Ready),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
    EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
}

TEST_P(CellStateTransitionTests_New, ready_dying)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Ready),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Dying),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
    EXPECT_EQ(CellState_Dying, getCell(actualData, 2)._cellState);
}

TEST_P(CellStateTransitionTests_New, ready_detaching)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Ready),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Detaching),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 2)._cellState);
    }
}

TEST_P(CellStateTransitionTests_New, ready_detaching_onSelfReplicator)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    // TODO
    //auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(
    //    GenomeDescription()
    //        .header(GenomeHeaderDescription())
    //        .cells({CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy())}));

    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .cellTypeData(ConstructorDescription())
            .pos({10.0f, 10.0f})
            .cellState(CellState_Ready),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellState(CellState_Detaching),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Reviving, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 2)._cellState);
    }
}

TEST_P(CellStateTransitionTests_New, ready_detaching_differentCreature)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).creatureId(1).cellState(CellState_Ready),
        CellDescription().id(2).pos({11.0f, 10.0f}).creatureId(2).cellState(CellState_Detaching),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 2)._cellState);
    }
}

TEST_P(CellStateTransitionTests_New, detaching_reviving)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Detaching),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Reviving),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Reviving, getCell(actualData, 1)._cellState);
        EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
    }
}

TEST_P(CellStateTransitionTests_New, underConstruction_activating)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_UnderConstruction),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Activating),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Activating, getCell(actualData, 1)._cellState);
    EXPECT_EQ(CellState_Ready, getCell(actualData, 2)._cellState);
}

TEST_P(CellStateTransitionTests_New, noDyingForBarrierCells)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    CollectionDescription data;
    data.addCells({
        CellDescription().id(1).barrier(true).pos({10.0f, 10.0f}).cellState(CellState_Dying),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, getCell(actualData, 1)._cellState);
}
