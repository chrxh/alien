#include <gtest/gtest.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/GenomeDescriptions.h"
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
    data.addCells(
        {CellDescription()
             .id(1)
             .pos({10.0f, 10.0f})
             .cellType(ReconnectorDescription()),
         CellDescription()
             .id(2)
             .pos({11.0f, 10.0f})
             .cellType(OscillatorDescription())
             .signal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnectorCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData._cells.size());
    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnectorCell._signal->_channels[0]));
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_noRestriction_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(ReconnectorDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToColor_failed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(ReconnectorDescription().restrictToColor(1)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToColor_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(ReconnectorDescription()),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToSameMutants_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToSameMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(5),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToSameMutants_failed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToSameMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(4),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(4),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_failed_zeroMutant)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_failed_respawnedCell)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(1),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_failed_sameMutant)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(5),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToZeroMutants_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToStructures)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToZeroMutants_failed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToStructures)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(4),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToRespawned_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToFreeCells)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(1),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell._connections.size());
    EXPECT_EQ(1, actualTargetCell._connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToRespawned_failed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToFreeCells)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell._signal->_channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_EQ(0, actualTargetCell._connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToLessComplexMutants_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .genomeComplexity(1000.0f)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToLessComplexMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(1).genomeComplexity(999.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToLessComplexMutants_failed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .genomeComplexity(1000.0f)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToLessComplexMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(1).genomeComplexity(1001.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToMoreComplexMutants_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .genomeComplexity(1000.0f)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToMoreComplexMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(1).genomeComplexity(1001.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToMoreComplexMutants_failed)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .mutationId(5)
            .genomeComplexity(1000.0f)
            .cellType(ReconnectorDescription().restrictToMutants(ReconnectorRestrictToMutants_RestrictToMoreComplexMutants)),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().id(3).pos({9.0f, 10.0f}).mutationId(1).genomeComplexity(1000.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, deleteConnections_success)
{
    CollectionDescription data;
    data.addCells({
        CellDescription()
            .id(1)
            .pos({10.0f, 10.0f})
            .cellType(ReconnectorDescription())
            .creatureId(1),
        CellDescription()
            .id(2)
            .pos({11.0f, 10.0f})
            .cellType(OscillatorDescription())
            .signal({-1, 0, 0, 0, 0, 0, 0, 0})
            .creatureId(1),
        CellDescription().id(3).pos({9.0f, 10.0f}).creatureId(3),
        CellDescription().id(4).pos({9.0f, 11.0f}).creatureId(4),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(4, actualData._cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell1 = getCell(actualData, 3);
    auto actualTargetCell2 = getCell(actualData, 4);

    EXPECT_TRUE(actualReconnectorCell._signal->_channels[0] > NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell._connections.size());
    EXPECT_TRUE(actualTargetCell1._connections.empty());
    EXPECT_TRUE(actualTargetCell2._connections.empty());
    EXPECT_TRUE(hasConnection(actualData, 1, 2));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
