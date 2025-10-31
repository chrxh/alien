#include <gtest/gtest.h>

#include "EngineInterface/NumberGenerator.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"

class CellConnectionTests : public IntegrationTestFramework
{
public:
    CellConnectionTests()
        : IntegrationTestFramework()
    {}

    ~CellConnectionTests() = default;
};

TEST_F(CellConnectionTests, decay)
{
    _parameters.radiationAbsorption.baseValue[0] = 0;
    _parameters.cellDeathConsequences.value = CellDeathConsquences_CreatureDies;
    _parameters.cellDeathProbability.baseValue[0] = 0.5f;

    _simulationFacade->setSimulationParameters(_parameters);
    auto origData = DescriptionEditService::get().createRect(
        DescriptionEditService::CreateRectParameters().width(1).height(1).energy(_parameters.minCellEnergy.baseValue[0] / 2));

    _simulationFacade->setSimulationData(origData);
    _simulationFacade->calcTimesteps(1000);

    auto data = _simulationFacade->getSimulationData();
    EXPECT_EQ(0, data._cells.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(origData)));
}

TEST_F(CellConnectionTests, addFirstConnection)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 2);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(2, actualData._cells.size());

    auto cell1 = actualData.getCellRef(1);
    EXPECT_EQ(1, cell1._connections.size());
    EXPECT_TRUE(approxCompare(360.0f, cell1._connections.front()._angleFromPrevious));
    EXPECT_TRUE(approxCompare(1.0f, cell1._connections.front()._distance));

    auto cell2 = actualData.getCellRef(2);
    EXPECT_EQ(1, cell2._connections.size());
    EXPECT_TRUE(approxCompare(360.0f, cell2._connections.front()._angleFromPrevious));
    EXPECT_TRUE(approxCompare(1.0f, cell2._connections.front()._distance));
}

TEST_F(CellConnectionTests, addSecondConnection)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({0, 1}),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._cells.size());

    auto cell = actualData.getCellRef(1);
    ASSERT_EQ(2, cell._connections.size());

    auto connection1 = cell._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(270.0f, connection1._angleFromPrevious));

    auto connection2 = cell._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));
}

TEST_F(CellConnectionTests, addThirdConnection1)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({0, 1}),
        CellDescription().id(4).pos({0, -1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 4);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(4, actualData._cells.size());

    auto cell = actualData.getCellRef(1);
    EXPECT_EQ(3, cell._connections.size());

    auto connection1 = cell._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection1._angleFromPrevious));

    auto connection2 = cell._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = cell._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection3._angleFromPrevious));
}


TEST_F(CellConnectionTests, addThirdConnection2)
{
    auto data = Description().cells({
        CellDescription().id(1).pos({0, 0}),
        CellDescription().id(2).pos({1, 0}),
        CellDescription().id(3).pos({-1, 0}),
        CellDescription().id(4).pos({0, 1}),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 4);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(4, actualData._cells.size());

    auto cell = actualData.getCellRef(1);
    EXPECT_EQ(3, cell._connections.size());

    auto connection1 = cell._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection1._angleFromPrevious));

    auto connection2 = cell._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = cell._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection3._angleFromPrevious));
}
