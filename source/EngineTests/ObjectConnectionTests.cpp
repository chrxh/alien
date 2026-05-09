#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class ObjectConnectionTests : public IntegrationTestFramework
{
public:
    ObjectConnectionTests()
        : IntegrationTestFramework()
    {}

    ~ObjectConnectionTests() = default;
};

TEST_F(ObjectConnectionTests, decay)
{
    _parameters.radiationAbsorption.baseValue[0] = 0;
    _parameters.cellDeathProbability.baseValue[0] = 0.5f;

    _simulationFacade->setSimulationParameters(_parameters);
    auto origData = DescEditService::get().createRect(
        DescEditService::CreateRectParameters().width(1).height(1).objectType(FreeCellDesc().energy(_parameters.minCellEnergy.baseValue[0] / 2)));

    _simulationFacade->setSimulationData(origData);
    _simulationFacade->calcTimesteps(1000);

    auto data = _simulationFacade->getSimulationData();
    EXPECT_EQ(0, data._objects.size());
    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(origData)));
}

TEST_F(ObjectConnectionTests, addFirstConnection)
{
    auto data = Desc().objects({
        ObjectDesc().id(1).pos({0, 0}).type(SolidDesc()),
        ObjectDesc().id(2).pos({1, 0}).type(SolidDesc()),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 2);

    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_EQ(2, actualData._objects.size());

    auto object1 = actualData.getObjectRef(1);
    EXPECT_EQ(1, object1._connections.size());
    EXPECT_TRUE(approxCompare(360.0f, object1._connections.front()._angleFromPrevious));
    EXPECT_TRUE(approxCompare(1.0f, object1._connections.front()._distance));

    auto object2 = actualData.getObjectRef(2);
    EXPECT_EQ(1, object2._connections.size());
    EXPECT_TRUE(approxCompare(360.0f, object2._connections.front()._angleFromPrevious));
    EXPECT_TRUE(approxCompare(1.0f, object2._connections.front()._distance));
}

TEST_F(ObjectConnectionTests, addSecondConnection)
{
    auto data = Desc().objects({
        ObjectDesc().id(1).pos({0, 0}).type(SolidDesc()),
        ObjectDesc().id(2).pos({1, 0}).type(SolidDesc()),
        ObjectDesc().id(3).pos({0, 1}).type(SolidDesc()),
    });
    data.addConnection(1, 2);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 3);

    auto actualData = _simulationFacade->getSimulationData();
    ASSERT_EQ(3, actualData._objects.size());

    auto object = actualData.getObjectRef(1);
    ASSERT_EQ(2, object._connections.size());

    auto connection1 = object._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(270.0f, connection1._angleFromPrevious));

    auto connection2 = object._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));
}

TEST_F(ObjectConnectionTests, addThirdConnection1)
{
    auto data = Desc().objects({
        ObjectDesc().id(1).pos({0, 0}).type(SolidDesc()),
        ObjectDesc().id(2).pos({1, 0}).type(SolidDesc()),
        ObjectDesc().id(3).pos({0, 1}).type(SolidDesc()),
        ObjectDesc().id(4).pos({0, -1}).type(SolidDesc()),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 4);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(4, actualData._objects.size());

    auto object = actualData.getObjectRef(1);
    EXPECT_EQ(3, object._connections.size());

    auto connection1 = object._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection1._angleFromPrevious));

    auto connection2 = object._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = object._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection3._angleFromPrevious));
}


TEST_F(ObjectConnectionTests, addThirdConnection2)
{
    auto data = Desc().objects({
        ObjectDesc().id(1).pos({0, 0}).type(SolidDesc()),
        ObjectDesc().id(2).pos({1, 0}).type(SolidDesc()),
        ObjectDesc().id(3).pos({-1, 0}).type(SolidDesc()),
        ObjectDesc().id(4).pos({0, 1}).type(SolidDesc()),
    });
    data.addConnection(1, 2);
    data.addConnection(1, 3);
    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_createConnection(1, 4);

    auto actualData = _simulationFacade->getSimulationData();
    EXPECT_EQ(4, actualData._objects.size());

    auto object = actualData.getObjectRef(1);
    EXPECT_EQ(3, object._connections.size());

    auto connection1 = object._connections.at(0);
    EXPECT_TRUE(approxCompare(1.0f, connection1._distance));
    EXPECT_TRUE(approxCompare(180.0f, connection1._angleFromPrevious));

    auto connection2 = object._connections.at(1);
    EXPECT_TRUE(approxCompare(1.0f, connection2._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection2._angleFromPrevious));

    auto connection3 = object._connections.at(2);
    EXPECT_TRUE(approxCompare(1.0f, connection3._distance));
    EXPECT_TRUE(approxCompare(90.0f, connection3._angleFromPrevious));
}
