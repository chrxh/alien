#include <gtest/gtest.h>

#include "Base/Math.h"
#include "EngineInterface/Description.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "IntegrationTestFramework.h"

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
    float angle2 = 0;
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
        Angles{0, 60.0f, 90.0f},
        Angles{0, 90.0f, 90.0f},
        Angles{0, 120.0f, 90.0f},
        Angles{0, 150.0f, 90.0f},
        Angles{0, 180.0f, 90.0f},
        Angles{0, 210.0f, 90.0f},
        Angles{0, 240.0f, 90.0f},
        Angles{0, 270.0f, 90.0f},
        Angles{0, 300.0f, 90.0f},
        Angles{0, 330.0f, 90.0f},
        Angles{0, 359.0f, 90.0f}));

TEST_P(PhysicsTests_TwoAngles, angularForces)
{
    auto [angle1, angle2, refAngle] = GetParam();

    auto pos1 = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(angle1) * 1.5f;
    auto pos2 = RealVector2D{100.0f, 100.0f};
    auto pos3 = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(angle2) * 1.5f;
    auto pos3ref = RealVector2D{100.0f, 100.0f} + Math::unitVectorOfAngle(angle1 + refAngle) * 1.5f;
    Description data;
    data._cells = {
        CellDescription().id(1).pos(pos1),
        CellDescription().id(2).pos(pos2),
        CellDescription().id(3).pos(pos3),
    };
    data.addConnection(2, 1);
    data.addConnection(2, 3, pos3ref);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 1; ++i) {
        _simulationFacade->calcTimesteps(1000);

        auto actualData = _simulationFacade->getSimulationData();

        auto cell1 = actualData.getCellRef(1);
        auto cell2 = actualData.getCellRef(2);
        auto cell3 = actualData.getCellRef(3);
        auto actualAngle = Math::angle(cell1._pos, cell2._pos, cell3._pos);
        printf("angle: %f\n", actualAngle);
        EXPECT_TRUE(abs(refAngle - actualAngle) < 1.0f);
    }
}
