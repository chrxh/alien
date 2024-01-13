#include <gtest/gtest.h>

#include "Base/Math.h"
#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeConstants.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/SimulationController.h"
#include "IntegrationTestFramework.h"

class LivingStateTransitionTests : public IntegrationTestFramework
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

protected:
};

TEST_F(LivingStateTransitionTests, staysReady)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Ready),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
}

TEST_F(LivingStateTransitionTests, dyingIfAdjacentDying)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Dying),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 2).livingState);
}

TEST_F(LivingStateTransitionTests, activatingUnderConstruction)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_UnderConstruction),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Activating),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    EXPECT_EQ(LivingState_Activating, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 2).livingState);
}

TEST_F(LivingStateTransitionTests, staysReadyIfAdjacentDying_differentCreatureId)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setCreatureId(1).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setCreatureId(2).setLivingState(LivingState_Dying),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    EXPECT_EQ(LivingState_Ready, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 2).livingState);
}

TEST_F(LivingStateTransitionTests, noSelfReplicatingConstructorIsDyingIfAdjacentDying)
{
    DataDescription data;
    data.addCells({
        CellDescription().setId(1).setPos({10.0f, 10.0f}).setMaxConnections(1).setCellFunction(ConstructorDescription()).setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Dying),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 2).livingState);
}

TEST_F(LivingStateTransitionTests, separatingSelfReplicatorIsDyingIfAdjacentDying)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription().setHeader(GenomeHeaderDescription().setSeparateConstruction(true)).setCells({CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy())}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setCellFunction(ConstructorDescription().setGenome(genome))
            .setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Dying),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 1).livingState);
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 2).livingState);
}

TEST_F(LivingStateTransitionTests, noSeparatingSelfReplicatorStaysReadyIfAdjacentDying)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(
        GenomeDescription()
            .setHeader(GenomeHeaderDescription().setSeparateConstruction(false))
            .setCells({CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()), CellGenomeDescription()}));

    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMaxConnections(1)
            .setCellFunction(ConstructorDescription().setGenome(genome))
            .setLivingState(LivingState_Ready),
        CellDescription().setId(2).setPos({11.0f, 10.0f}).setMaxConnections(1).setLivingState(LivingState_Dying),
    });
    data.addConnection(1, 2);

    _simController->setSimulationData(data);
    _simController->calcTimesteps(1);
    auto actualData = _simController->getSimulationData();
    auto actualCell1 = getCell(actualData, 1);
    auto actualConstructor = std::get<ConstructorDescription>(*actualCell1.cellFunction);
    EXPECT_TRUE(actualConstructor.isConstructionBuilt());
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
    EXPECT_EQ(LivingState_Ready, actualCell1.livingState);
    EXPECT_EQ(LivingState_Dying, getCell(actualData, 2).livingState);
}
