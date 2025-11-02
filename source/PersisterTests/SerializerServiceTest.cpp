#include <gtest/gtest.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include <PersisterInterface/SerializerService.h>


class SerializerServiceTests : public ::testing::Test
{
public:
    SerializerServiceTests()
    {
        _descriptionTestDataFactory = &DescriptionTestDataFactory::get();
        _serializerService = &SerializerService::get();
    }

    void testSerializationAndDeserialization(Description const& data)
    {
        DeserializedSimulation deserializedSimulationBefore{.mainData = data};
        SerializedSimulation serializedSimulation;
        _serializerService->serializeSimulationToStrings(serializedSimulation, deserializedSimulationBefore);

        DeserializedSimulation deserializedSimulationAfter;
        _serializerService->deserializeSimulationFromStrings(deserializedSimulationAfter, serializedSimulation);

        EXPECT_TRUE(_descriptionTestDataFactory->compare(deserializedSimulationBefore.mainData, deserializedSimulationAfter.mainData));
    }

protected:
    DescriptionTestDataFactory* _descriptionTestDataFactory;
    SerializerService* _serializerService;
};

TEST_F(SerializerServiceTests, singleParticle)
{
    Description data;
    data._particles.emplace_back(_descriptionTestDataFactory->createNonDefaultParticleDescription());

    testSerializationAndDeserialization(data);
}

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
        CellParameter{CellType_Generator},
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

TEST_P(SerializerServiceTests_AllCellTypes, cellWithoutCreature)
{
    auto cellParameter = GetParam();

    Description data;
    data._cells.emplace_back(_descriptionTestDataFactory->createNonDefaultCellDescription(cellParameter));

    testSerializationAndDeserialization(data);
}

using NodeParameter = DescriptionTestDataFactory::NodeParameter;
class SerializerServiceTests_AllNodeTypes
    : public SerializerServiceTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    SerializerServiceTests_AllNodeTypes,
    SerializerServiceTests_AllNodeTypes,
    ::testing::Values(
        NodeParameter{CellTypeGenome_Base},
        NodeParameter{CellTypeGenome_Depot},
        NodeParameter{CellTypeGenome_Constructor},
        NodeParameter{CellTypeGenome_Sensor},
        NodeParameter{CellTypeGenome_Generator},
        NodeParameter{CellTypeGenome_Attacker},
        NodeParameter{CellTypeGenome_Injector},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AutoBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_ManualBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AngleBending},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_AutoCrawling},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_ManualCrawling},
        NodeParameter{CellTypeGenome_Muscle, MuscleMode_DirectMovement},
        NodeParameter{CellTypeGenome_Defender},
        NodeParameter{CellTypeGenome_Reconnector},
        NodeParameter{CellTypeGenome_Detonator}));

TEST_P(SerializerServiceTests_AllNodeTypes, cellWithCreature)
{
    auto nodeParameter = GetParam();

    auto [creature, genome] = _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter);

    auto data = Description().addCreature(creature.cells({CellDescription()}), genome);

    testSerializationAndDeserialization(data);
}
