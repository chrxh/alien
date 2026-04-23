#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

#include "PersisterInterface/DeserializedSimulation.h"
#include "PersisterInterface/SerializerService.h"

class PhysicsTests : public IntegrationTestFramework
{
public:
    PhysicsTests()
        : IntegrationTestFramework()
    {
        _parameters.friction.baseValue = 0.1f;
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~PhysicsTests() = default;
};

struct Angles
{
    float angle1 = 0;
    float angleOffset2 = 0;
    float refAngle = 0;
};

class PhysicsTests_TwoAngles
    : public PhysicsTests
    , public testing::WithParamInterface<Angles>
{};

INSTANTIATE_TEST_SUITE_P(
    PhysicsTests_TwoAngles,
    PhysicsTests_TwoAngles,
    ::testing::Values(
        Angles{0, 1.0f, 90.0f},
        Angles{0, 30.0f, 90.0f},
        Angles{0, 90.0f, 90.0f},
        Angles{0, 120.0f, 90.0f},
        Angles{0, 180.0f, 90.0f},
        Angles{0, 240.0f, 90.0f},
        Angles{0, 270.0f, 90.0f},
        Angles{0, 359.0f, 90.0f},

        Angles{-30.0f, 1.0f, 90.0f},
        Angles{-30.0f, 30.0f, 90.0f},
        Angles{-30.0f, 90.0f, 90.0f},
        Angles{-30.0f, 120.0f, 90.0f},
        Angles{-30.0f, 180.0f, 90.0f},
        Angles{-30.0f, 240.0f, 90.0f},
        Angles{-30.0f, 270.0f, 90.0f},
        Angles{-30.0f, 359.0f, 90.0f},

        Angles{-30.0f, 1.0f, 180.0f},
        Angles{-30.0f, 30.0f, 180.0f},
        Angles{-30.0f, 120.0f, 180.0f},
        Angles{-30.0f, 180.0f, 180.0f},
        Angles{-30.0f, 270.0f, 180.0f},
        Angles{-30.0f, 359.0f, 180.0f},

        Angles{-30.0f, 1.0f, 270.0f},
        Angles{-30.0f, 30.0f, 270.0f},
        Angles{-30.0f, 120.0f, 270.0f},
        Angles{-30.0f, 180.0f, 270.0f},
        Angles{-30.0f, 270.0f, 270.0f},
        Angles{-30.0f, 359.0f, 270.0f}));

TEST_P(PhysicsTests_TwoAngles, angularForces)
{
    auto [angle1, angleOffset2, refAngle] = GetParam();
    auto angle2 = angle1 + angleOffset2;

    auto pos1 = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(angle1) * 1.5f;
    auto pos2 = RealVector2D{100.0f, 100.0f};
    auto pos3 = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(angle2) * 1.5f;
    auto pos3ref = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(angle1 + refAngle) * 1.5f;
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos(pos1).type(CellDesc().headCell(true)),
        ObjectDesc().id(2).pos(pos2),
        ObjectDesc().id(3).pos(pos3),
    });
    data.addConnection(2, 1);
    data.addConnection(2, 3, pos3ref);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    auto object2 = actualData.getObjectRef(2);
    auto cell3 = actualData.getObjectRef(3);
    auto actualAngle = Math::angle(object1._pos, object2._pos, cell3._pos);
    EXPECT_TRUE(abs(Math::getNormalizedAngle(refAngle - actualAngle, -180.0f)) < 1.0f);
}

TEST_F(PhysicsTests, noGhostRotations)
{
    auto constexpr Angle1 = 0.0f;
    auto constexpr Angle2 = 120.0f;
    auto constexpr RefAngle = 90.0f;

    auto pos1 = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(Angle1);
    auto pos2 = RealVector2D{100.0f, 100.0f};
    auto pos3 = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(Angle2);
    auto pos3ref = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(Angle1 + RefAngle);
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos(pos1).type(CellDesc().headCell(true)),
        ObjectDesc().id(2).pos(pos2),
        ObjectDesc().id(3).pos(pos3),
    });
    data.addConnection(2, 1);
    data.addConnection(2, 3, pos3ref);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto savedPos1 = actualData.getObjectRef(1)._pos;
    auto savedPos2 = actualData.getObjectRef(2)._pos;
    auto savedPos3 = actualData.getObjectRef(3)._pos;

    _simulationFacade->calcTimesteps(1000);
    actualData = _simulationFacade->getSimulationData();
    auto currentPos1 = actualData.getObjectRef(1)._pos;
    auto currentPos2 = actualData.getObjectRef(2)._pos;
    auto currentPos3 = actualData.getObjectRef(3)._pos;

    EXPECT_TRUE(Math::length(savedPos1 - currentPos1) < 0.01f);
    EXPECT_TRUE(Math::length(savedPos2 - currentPos2) < 0.01f);
    EXPECT_TRUE(Math::length(savedPos3 - currentPos3) < 0.01f);
}

TEST_F(PhysicsTests, noGhostMovements)
{
    _parameters.friction.baseValue = 0.001f;
    _simulationFacade->setSimulationParameters(_parameters);

    auto data = Desc()
                    .addCreature({
                        ObjectDesc().id(1).pos({34.97, 11.14}).type(CellDesc().headCell(true)),
                    })
                    .addCreature({
                        ObjectDesc().id(2).pos({33.95, 11.14}).type(CellDesc().headCell(true)),
                        ObjectDesc().id(3).pos({34.01, 12.16}),
                        ObjectDesc().id(4).pos({34.88, 12.66}),
                    });
    data.addConnection(1, 2);
    data.addConnection(2, 3);
    data.addConnection(3, 4);

    data.getConnectionRef(2, 1)._angleFromPrevious = 300.0f;
    data.getConnectionRef(2, 3)._angleFromPrevious = 60.0f;
    data.getConnectionRef(3, 2)._angleFromPrevious = 240.0f;
    data.getConnectionRef(3, 4)._angleFromPrevious = 120.0f;

    auto center = DescEditService::get().calcCenter(data);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1000);

    auto actualData = _simulationFacade->getSimulationData();
    auto newCenter = DescEditService::get().calcCenter(actualData);

    EXPECT_TRUE(approxCompare(center, newCenter, 0.01f));
}

TEST_F(PhysicsTests, angularForcesBetweenFixedAndNonFixedObject)
{
    auto data = Desc().addCreature({
        ObjectDesc().id(1).pos({10.0f, 10.0f}).fixed(true),
        ObjectDesc().id(2).pos({10.0f, 11.0f}).fixed(true),
        ObjectDesc().id(3).pos({11.0f, 11.0f}).type(CellDesc().headCell(true)),
    });
    data.addConnection(2, 1);
    data.addConnection(2, 3);
    data.getConnectionRef(2, 1)._angleFromPrevious = 180.0f;
    data.getConnectionRef(2, 3)._angleFromPrevious = 180.0f;

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(100);

    auto actualData = _simulationFacade->getSimulationData();

    auto object1 = actualData.getObjectRef(1);
    auto object2 = actualData.getObjectRef(2);
    auto object3 = actualData.getObjectRef(3);
    auto actualAngle = Math::angle(object1._pos, object2._pos, object3._pos);
    EXPECT_TRUE(abs(Math::getNormalizedAngle(actualAngle, 0.0f) - 180.0f) < 10.0f);
}
