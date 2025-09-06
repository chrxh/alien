#include <gtest/gtest.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescription.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class ReconnectorTests : public IntegrationTestFramework
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
    ReconnectorTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~ReconnectorTests() = default;
};

TEST_F(ReconnectorTests, establishConnection_noRestriction_nothingFound)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription()),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
    };
    data.addConnection(1, 2);
    
    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnectorCell = actualData.getCellRef(1);

    EXPECT_EQ(2, actualData._cells.size());
    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnectorCell._signal->_channels[0]));
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_noRestriction_success)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription()),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = actualData.getCellRef(1);

    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToColor_failed)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription().restrictToColor(1)),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToColor_success)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription().restrictToColor(1)),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).color(1),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = actualData.getCellRef(1);

    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToSameMutants_success)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).id(0).lineageId(5).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToSameMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().lineageId(5).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._creatures.size());

    auto actualReconnectorCell = actualData.getCellRef(1);

    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToSameMutants_failed)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).lineageId(5).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToSameMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().lineageId(6).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._creatures.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_success)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).lineageId(5).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().lineageId(6).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._creatures.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_failed)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).lineageId(5).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToOtherMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().lineageId(5).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(2, actualData._creatures.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToStructures_success)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToStructures)),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(StructureCellDescription()),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToZeroMutants_failed)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToStructures)),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(BaseDescription()),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToFreeCells_success)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToFreeCells)),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(FreeCellDescription()),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(actualData.hasConnection(1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToFreeCells_failed)
{
    CollectionDescription data;
    data._cells = {
        CellDescription().id(1).pos({10.0f, 10.0f}).cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToFreeCells)),
        CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).cellTypeData(BaseDescription()),
    };
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = actualData.getCellRef(1);
    auto actualTargetCell = actualData.getCellRef(3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToLessComplexMutants_success)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToLessComplexMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().genomeComplexity(999.0f).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(actualData.hasConnection(1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToLessComplexMutants_failed)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToLessComplexMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().genomeComplexity(1001.0f).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(actualData.hasConnection(1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToMoreComplexMutants_success)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().genomeComplexity(1001.0f).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(actualData.hasConnection(1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToMoreComplexMutants_failed)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).genomeComplexity(1000.0f).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription().restrictToCreatures(ReconnectorRestrictToCreatures_RestrictToMoreComplexMutants)),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().genomeComplexity(1000.0f).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(actualData.hasConnection(1, 3));
}

TEST_F(ReconnectorTests, deleteConnections_success)
{
    CollectionDescription data;
    data.creatures({
        CreatureDescription().id(0).cells({
            CellDescription()
                .id(1)
                .pos({10.0f, 10.0f})
                .cellTypeData(ReconnectorDescription()),
            CellDescription().id(2).pos({11.0f, 10.0f}).signalAndRelaxTime({-1, 0, 0, 0, 0, 0, 0, 0}),
        }),
        CreatureDescription().lineageId(5).cells({
            CellDescription().id(3).pos({9.0f, 10.0f}),
        }),
        CreatureDescription().lineageId(5).cells({
            CellDescription().id(4).pos({9.0f, 11.0f}),
        }),
    });

    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._creatures.size());

    auto actualReconnectorCell = actualData.getCellRef(1);

    auto actualTargetCell1 = actualData.getCellRef(3);
    auto actualTargetCell2 = actualData.getCellRef(4);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_TRUE(actualTargetCell1._connections.empty());
    EXPECT_TRUE(actualTargetCell2._connections.empty());
    EXPECT_TRUE(actualData.hasConnection(1, 2));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
