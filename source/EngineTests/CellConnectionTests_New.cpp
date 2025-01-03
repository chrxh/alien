#include <gtest/gtest.h>

#include "Base/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class CellConnectionTests_New : public IntegrationTestFramework
{
public:
    CellConnectionTests_New()
        : IntegrationTestFramework()
    {}

    ~CellConnectionTests_New() = default;
};

TEST_F(CellConnectionTests_New, decay)
{
    _parameters.baseValues.radiationAbsorption[0] = 0;
    _parameters.cellDeathConsequences = CellDeathConsquences_CreatureDies;
    _parameters.baseValues.cellDeathProbability[0] = 0.5f;

    _simulationFacade->setSimulationParameters(_parameters);
    auto origData = DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().width(10).height(10).energy(_parameters.baseValues.cellMinEnergy[0] / 2));

    _simulationFacade->setSimulationData(origData);
    _simulationFacade->calcTimesteps(1000);

    auto data = _simulationFacade->getSimulationData();
    EXPECT_EQ(0, data.cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(origData)));
}

TEST_F(CellConnectionTests_New, addFirstConnection)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}),
        CellDescription().setId(2).setPos({1, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    EXPECT_EQ(2, actualData.cells.size());

    auto cell1 = actualCellById.at(1);
    EXPECT_EQ(1, cell1.connections.size());
    EXPECT_TRUE(approxCompare(360.0f, cell1.connections.front().angleFromPrevious));
    EXPECT_TRUE(approxCompare(1.0f, cell1.connections.front().distance));

    auto cell2 = actualCellById.at(2);
    EXPECT_EQ(1, cell2.connections.size());
    EXPECT_TRUE(approxCompare(360.0f, cell2.connections.front().angleFromPrevious));
    EXPECT_TRUE(approxCompare(1.0f, cell2.connections.front().distance));
}

TEST_F(CellConnectionTests_New, addSecondConnection)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}),
        CellDescription().setId(2).setPos({1, 0}),
        CellDescription().setId(3).setPos({0, 1}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData.cells.size());

    auto actualCellById = getCellById(actualData);
    auto cell = actualCellById.at(1);
    ASSERT_EQ(2, cell.connections.size());

    auto connection1 = cell.connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1.distance));
    EXPECT_TRUE(approxCompare(270.0f, connection1.angleFromPrevious));

    auto connection2 = cell.connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2.angleFromPrevious));
}

TEST_F(CellConnectionTests_New, addThirdConnection1)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}),
        CellDescription().setId(2).setPos({1, 0}),
        CellDescription().setId(3).setPos({0, 1}),
        CellDescription().setId(4).setPos({0, -1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 4);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(4, actualData.cells.size());

    auto actualCellById = getCellById(actualData);

    auto cell = actualCellById.at(1);
    EXPECT_EQ(3, cell.connections.size());

    auto connection1 = cell.connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection1.angleFromPrevious));

    auto connection2 = cell.connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2.angleFromPrevious));

    auto connection3 = cell.connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3.distance));
    EXPECT_TRUE(approxCompare(180.0f, connection3.angleFromPrevious));
}


TEST_F(CellConnectionTests_New, addThirdConnection2)
{
    auto data = DataDescription().addCells({
        CellDescription().setId(1).setPos({0, 0}),
        CellDescription().setId(2).setPos({1, 0}),
        CellDescription().setId(3).setPos({-1, 0}),
        CellDescription().setId(4).setPos({0, 1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 4);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(4, actualData.cells.size());

    auto actualCellById = getCellById(actualData);

    auto cell = actualCellById.at(1);
    EXPECT_EQ(3, cell.connections.size());

    auto connection1 = cell.connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1.distance));
    EXPECT_TRUE(approxCompare(180.0f, connection1.angleFromPrevious));

    auto connection2 = cell.connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2.angleFromPrevious));

    auto connection3 = cell.connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3.distance));
    EXPECT_TRUE(approxCompare(90.0f, connection3.angleFromPrevious));
}
