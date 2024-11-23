#include <gtest/gtest.h>

#include "Base/Math.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class LivingStateTransitionTests
    : public IntegrationTestFramework
    , public testing::WithParamInterface<CellDeathConsquences>
{
public:
    static SimulationParameters getParameters()
    {
        SimulationParameters result;
        result.innerFriction = 0;
        result.baseValues.friction = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.baseValues.radiationCellAgeStrength[i] = 0;
        }
        return result;
    }

    LivingStateTransitionTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~LivingStateTransitionTests() = default;
};

INSTANTIATE_TEST_SUITE_P(
    LivingStateTransitionTests,
    LivingStateTransitionTests,
    ::testing::Values(CellDeathConsquences_None, CellDeathConsquences_DetachedPartsDie, CellDeathConsquences_CreatureDies));

TEST_P(LivingStateTransitionTests, ready_ready)
{
    _parameters.cellDeathConsequences = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Ready),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
}

TEST_P(LivingStateTransitionTests, ready_dying)
{
    _parameters.cellDeathConsequences = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Dying),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 2).livingState);
}

TEST_P(LivingStateTransitionTests, ready_detaching)
{
    _parameters.cellDeathConsequences = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Detaching),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 2).livingState);
    }
}

TEST_P(LivingStateTransitionTests, ready_detaching_onSelfReplicator)
{
    _parameters.cellDeathConsequences = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription())
            .setCells({CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy())}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setCellFunction(ConstructorDescription().setGenome(genome))
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setLivingState(LivingState_Ready),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setMaxConnections(1)
            .setLivingState(LivingState_Detaching),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(LivingState_Reviving, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 2).livingState);
    }
}

TEST_P(LivingStateTransitionTests, ready_detaching_differentCreature)
{
    _parameters.cellDeathConsequences = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setCreatureId(1).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setCreatureId(2).setLivingState(LivingState_Detaching),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 2).livingState);
    }
}

TEST_P(LivingStateTransitionTests, detaching_reviving)
{
    _parameters.cellDeathConsequences = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Detaching),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Reviving),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();

    if (GetParam() == CellDeathConsquences_None) {
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_CreatureDies) {
        EXPECT_EQ(LivingState_Detaching, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
    } else if (GetParam() == CellDeathConsquences_DetachedPartsDie) {
        EXPECT_EQ(LivingState_Reviving, getCell(actualData, 1).livingState);
        EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
    }
}

TEST_P(LivingStateTransitionTests, underConstruction_activating)
{
    _parameters.cellDeathConsequences = GetParam();
    _simulationFacade->setSimulationParameters(_parameters);

    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Activating),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);
    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(LivingState_Activating, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
}
