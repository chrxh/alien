#include <gtest/gtest.h>

#include "EngineTestData/DescriptionTestDataFactory.h"


class SerializerServiceTests : public ::testing::Test
{
};

using CellParameter = DescriptionTestDataFactory::CellParameter;
class SerializerServiceTests_AllCellTypes
    : public SerializerServiceTests
    , public testing::WithParamInterface<CellParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    SerializerServiceTests_AllCellTypes,
    SerializerServiceTests_AllCellTypes,
    ::testing::Values(
        CellParameter{CellType_Structure},
        CellParameter{CellType_Free},
        CellParameter{CellType_Base},
        CellParameter{CellType_Depot},
        CellParameter{CellType_Constructor},
        CellParameter{CellType_Sensor},
        CellParameter{CellType_Oscillator},
        CellParameter{CellType_Attacker},
        CellParameter{CellType_Injector},
        CellParameter{CellType_Muscle, MuscleMode_AutoBending},
        CellParameter{CellType_Muscle, MuscleMode_ManualBending},
        CellParameter{CellType_Muscle, MuscleMode_AngleBending},
        CellParameter{CellType_Muscle, MuscleMode_AutoCrawling},
        CellParameter{CellType_Muscle, MuscleMode_ManualCrawling},
        CellParameter{CellType_Muscle, MuscleMode_DirectMovement},
        CellParameter{CellType_Defender},
        CellParameter{CellType_Reconnector},
        CellParameter{CellType_Detonator}));

TEST_F(SerializerServiceTests, singleCellWithoutCreature) {
}
