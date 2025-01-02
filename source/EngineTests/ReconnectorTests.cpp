#include <gtest/gtest.h>

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionService.h"
#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class ReconnectorTests : public IntegrationTestFramework
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
    ReconnectorTests()
        : IntegrationTestFramework(getParameters())
    {}

    ~ReconnectorTests() = default;
};

TEST_F(ReconnectorTests, establishConnection_noRestriction_nothingFound)
{
    DataDescription data;
    data.addCells(
        {CellDescription()
             .setId(1)
             .setPos({10.0f, 10.0f})
             .setCellFunction(ReconnectorDescription()),
         CellDescription()
             .setId(2)
             .setPos({11.0f, 10.0f})
             .setCellFunction(OscillatorDescription())
             .setSignal({1, 0, 0, 0, 0, 0, 0, 0})});
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualReconnectorCell = getCell(actualData, 1);

    EXPECT_EQ(2, actualData.cells.size());
    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(0.0f, actualReconnectorCell.signal->channels[0]));
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
}

TEST_F(ReconnectorTests, establishConnection_noRestriction_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellFunction(ReconnectorDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell.signal->channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell.connections.size());
    EXPECT_EQ(1, actualTargetCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToColor_failed)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellFunction(ReconnectorDescription().setRestrictToColor(1)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_EQ(0, actualTargetCell.connections.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToColor_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellFunction(ReconnectorDescription()),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell.signal->channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell.connections.size());
    EXPECT_EQ(1, actualTargetCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToSameMutants_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToSameMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(5),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell.signal->channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell.connections.size());
    EXPECT_EQ(1, actualTargetCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToSameMutants_failed)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToSameMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(4),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_EQ(0, actualTargetCell.connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(4),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell.signal->channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell.connections.size());
    EXPECT_EQ(1, actualTargetCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_failed_zeroMutant)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_EQ(0, actualTargetCell.connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_failed_respawnedCell)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(1),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_EQ(0, actualTargetCell.connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToOtherMutants_failed_sameMutant)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToOtherMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(5),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_EQ(0, actualTargetCell.connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToZeroMutants_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToHandcraftedCells)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell.signal->channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell.connections.size());
    EXPECT_EQ(1, actualTargetCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToZeroMutants_failed)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToHandcraftedCells)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(4),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_EQ(0, actualTargetCell.connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToRespawned_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToFreeCells)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(1),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(actualReconnectorCell.signal->channels[0] > NEAR_ZERO);
    EXPECT_EQ(2, actualReconnectorCell.connections.size());
    EXPECT_EQ(1, actualTargetCell.connections.size());
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}

TEST_F(ReconnectorTests, establishConnection_restrictToRespawned_failed)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToFreeCells)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(0),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);
    auto actualTargetCell = getCell(actualData, 3);

    EXPECT_TRUE(std::abs(actualReconnectorCell.signal->channels[0]) < NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_EQ(0, actualTargetCell.connections.size());
}

TEST_F(ReconnectorTests, establishConnection_restrictToLessComplexMutants_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setGenomeComplexity(1000.0f)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToLessComplexMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(1).setGenomeComplexity(999.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToLessComplexMutants_failed)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setGenomeComplexity(1000.0f)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToLessComplexMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(1).setGenomeComplexity(1001.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToMoreComplexMutants_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setGenomeComplexity(1000.0f)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToMoreComplexMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(1).setGenomeComplexity(1001.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_TRUE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, establishConnection_restrictToMoreComplexMutants_failed)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setMutationId(5)
            .setGenomeComplexity(1000.0f)
            .setCellFunction(ReconnectorDescription().setRestrictToMutants(ReconnectorRestrictToMutants_RestrictToMoreComplexMutants)),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({1, 0, 0, 0, 0, 0, 0, 0}),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setMutationId(1).setGenomeComplexity(1000.0f),
    });
    data.addConnection(1, 2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_FALSE(hasConnection(actualData, 1, 3));
}

TEST_F(ReconnectorTests, deleteConnections_success)
{
    DataDescription data;
    data.addCells({
        CellDescription()
            .setId(1)
            .setPos({10.0f, 10.0f})
            .setCellFunction(ReconnectorDescription())
            .setCreatureId(1),
        CellDescription()
            .setId(2)
            .setPos({11.0f, 10.0f})
            .setCellFunction(OscillatorDescription())
            .setSignal({-1, 0, 0, 0, 0, 0, 0, 0})
            .setCreatureId(1),
        CellDescription().setId(3).setPos({9.0f, 10.0f}).setCreatureId(3),
        CellDescription().setId(4).setPos({9.0f, 11.0f}).setCreatureId(4),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    data.addConnection(1, 4);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(4, actualData.cells.size());

    auto actualReconnectorCell = getCell(actualData, 1);

    auto actualTargetCell1 = getCell(actualData, 3);
    auto actualTargetCell2 = getCell(actualData, 4);

    EXPECT_TRUE(actualReconnectorCell.signal->channels[0] > NEAR_ZERO);
    EXPECT_EQ(1, actualReconnectorCell.connections.size());
    EXPECT_TRUE(actualTargetCell1.connections.empty());
    EXPECT_TRUE(actualTargetCell2.connections.empty());
    EXPECT_TRUE(hasConnection(actualData, 1, 2));
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
}
