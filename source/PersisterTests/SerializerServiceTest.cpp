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

TEST_F(SerializerServiceTests, singleEnergyParticle)
{
    Description data;
    data._energies.emplace_back(_descriptionTestDataFactory->createNonDefaultEnergyDescription());

    testSerializationAndDeserialization(data);
}

using ObjectParameter = DescriptionTestDataFactory::ObjectParameter;
class SerializerServiceTests_AllCellTypes
    : public SerializerServiceTests
    , public testing::WithParamInterface<ObjectParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    SerializerServiceTests_AllCellTypes,
    SerializerServiceTests_AllCellTypes,
    ::testing::ValuesIn(DescriptionTestDataFactory::get().getAllObjectParameters()));

TEST_P(SerializerServiceTests_AllCellTypes, objectWithEmptyGenome)
{
    auto objectParameter = GetParam();

    Description data;
    if (objectParameter.objectType == ObjectType_Cell) {
        data.addCreature({_descriptionTestDataFactory->createNonDefaultObjectDescription(objectParameter)}, CreatureDescription(), GenomeDescription());
    } else {
        data.objects({_descriptionTestDataFactory->createNonDefaultObjectDescription(objectParameter)});
    }


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
    ::testing::ValuesIn(DescriptionTestDataFactory::get().getAllNodeParameters()));

TEST_P(SerializerServiceTests_AllNodeTypes, objectWithNonEmptyGenome)
{
    auto nodeParameter = GetParam();

    auto [creature, genome] = _descriptionTestDataFactory->createNonDefaultCreatureDescription(nodeParameter);

    auto data = Description().addCreature({ObjectDescription()}, creature, genome);

    testSerializationAndDeserialization(data);
}
