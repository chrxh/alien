#include <gtest/gtest.h>

#include <EngineTestData/DescTestDataFactory.h>

#include <PersisterInterface/SerializerService.h>


class SerializerServiceTests : public ::testing::Test
{
public:
    SerializerServiceTests()
    {
        _descTestDataFactory = &DescTestDataFactory::get();
        _serializerService = &SerializerService::get();
    }

    void testSerializationAndDeserialization(Desc const& data)
    {
        DeserializedSimulation deserializedSimulationBefore{.mainData = data};
        SerializedSimulation serializedSimulation;
        _serializerService->serializeSimulationToStrings(serializedSimulation, deserializedSimulationBefore);

        DeserializedSimulation deserializedSimulationAfter;
        _serializerService->deserializeSimulationFromStrings(deserializedSimulationAfter, serializedSimulation);

        EXPECT_TRUE(_descTestDataFactory->compare(deserializedSimulationBefore.mainData, deserializedSimulationAfter.mainData));
    }

protected:
    DescTestDataFactory* _descTestDataFactory;
    SerializerService* _serializerService;
};

TEST_F(SerializerServiceTests, singleEnergyParticle)
{
    Desc data;
    data._energies.emplace_back(_descTestDataFactory->createNonDefaultEnergyDesc());

    testSerializationAndDeserialization(data);
}

using ObjectParameter = DescTestDataFactory::ObjectParameter;
class SerializerServiceTests_AllCellTypes
    : public SerializerServiceTests
    , public testing::WithParamInterface<ObjectParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    SerializerServiceTests_AllCellTypes,
    SerializerServiceTests_AllCellTypes,
    ::testing::ValuesIn(DescTestDataFactory::get().getAllObjectParameters()));

TEST_P(SerializerServiceTests_AllCellTypes, objectWithEmptyGenome)
{
    auto objectParameter = GetParam();

    Desc data;
    if (objectParameter.objectType == ObjectType_Cell) {
        data.addCreature({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter)}, CreatureDesc(), GenomeDesc());
    } else {
        data.objects({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter)});
    }


    testSerializationAndDeserialization(data);
}

using NodeParameter = DescTestDataFactory::NodeParameter;
class SerializerServiceTests_AllNodeTypes
    : public SerializerServiceTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    SerializerServiceTests_AllNodeTypes,
    SerializerServiceTests_AllNodeTypes,
    ::testing::ValuesIn(DescTestDataFactory::get().getAllNodeParameters()));

TEST_P(SerializerServiceTests_AllNodeTypes, objectWithNonEmptyGenome)
{
    auto nodeParameter = GetParam();

    auto [creature, genome] = _descTestDataFactory->createNonDefaultCreatureDesc(nodeParameter);

    auto data = Desc().addCreature({ObjectDesc()}, creature, genome);

    testSerializationAndDeserialization(data);
}

TEST_F(SerializerServiceTests, oldCustomConstructorShapeLoadsAsSegment)
{
    Desc data;
    data._genomes.emplace_back(GenomeDesc().genes({GeneDesc().shape(0).nodes({NodeDesc()})}));

    DeserializedSimulation deserializedSimulationBefore{.mainData = data};
    SerializedSimulation serializedSimulation;
    ASSERT_TRUE(_serializerService->serializeSimulationToStrings(serializedSimulation, deserializedSimulationBefore));

    DeserializedSimulation deserializedSimulationAfter;
    ASSERT_TRUE(_serializerService->deserializeSimulationFromStrings(deserializedSimulationAfter, serializedSimulation));

    ASSERT_EQ(1, deserializedSimulationAfter.mainData._genomes.size());
    ASSERT_EQ(1, deserializedSimulationAfter.mainData._genomes.at(0)._genes.size());
    EXPECT_EQ(ConstructorShape_Segment, deserializedSimulationAfter.mainData._genomes.at(0)._genes.at(0)._shape);
}
