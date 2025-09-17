#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class CellStateTransitionTests
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
            result.cellDeathProbability.baseValue[i] = 0;
            result.radiationType1_strength.baseValue[i] = 0;
        }
        return result;
    }

    CellStateTransitionTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~CellStateTransitionTests() = default;
};

INSTANTIATE_TEST_SUITE_P(
    CellStateTransitionTests,
    CellStateTransitionTests,
    ::testing::Values(CellDeathConsquences_None, CellDeathConsquences_DetachedPartsDie, CellDeathConsquences_CreatureDies));

TEST_P(CellStateTransitionTests, ready_ready)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().cells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Ready),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Ready),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
}

TEST_P(CellStateTransitionTests, ready_dying)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().cells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Ready),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Dying),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
    EXPECT_EQ(CellState_Dying, actualData.getCellRef(2)._cellState);
}

TEST_P(CellStateTransitionTests, ready_detaching)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().cells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Ready),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Detaching),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(2)._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_onSelfReplicator)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().creatures(
        {CreatureDescription()
             .genome(GenomeDescription().genes({
                 GeneDescription().separation(true).nodes({NodeDescription()}),
             }))
             .cells({
                 CellDescription().id(1).cellTypeData(ConstructorDescription().geneIndex(0)).pos({10.0f, 10.0f}).cellState(CellState_Ready),
                 CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Detaching),
             })});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Reviving, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(2)._cellState);
    }
}

TEST_P(CellStateTransitionTests, ready_detaching_differentCreature)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    Description data;
    data.creatures({
        CreatureDescription().cells({CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Ready)}),
        CreatureDescription().cells({CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Detaching)}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(2)._cellState);
    }
}

TEST_P(CellStateTransitionTests, detaching_reviving)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().cells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Detaching),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Reviving),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(CellState_Detaching, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(CellState_Reviving, actualData.getCellRef(1)._cellState);
        EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
    }
}

TEST_P(CellStateTransitionTests, underConstruction_activating)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().cells({
        CellDescription().id(1).pos({10.0f, 10.0f}).cellState(CellState_Constructing),
        CellDescription().id(2).pos({11.0f, 10.0f}).cellState(CellState_Activating),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Activating, actualData.getCellRef(1)._cellState);
    EXPECT_EQ(CellState_Ready, actualData.getCellRef(2)._cellState);
}

TEST_P(CellStateTransitionTests, noDyingForBarrierCells)
{
    _parameters.cellDeathConsequences.value = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Description().cells({
        CellDescription().id(1).barrier(true).pos({10.0f, 10.0f}).cellState(CellState_Dying),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(CellState_Ready, actualData.getCellRef(1)._cellState);
}
