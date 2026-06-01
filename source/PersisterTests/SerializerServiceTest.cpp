#include <gtest/gtest.h>

#include <cereal/archives/portable_binary.hpp>
#include <Base/Resources.h>
#include <EngineTestData/DescTestDataFactory.h>

#include <PersisterInterface/SerializerService.h>

#include <iterator>
#include <sstream>
#include <zstr.hpp>


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

TEST_F(SerializerServiceTests, legacyArchiveWithoutVersionHeader)
{
    Desc data;
    data._energies.emplace_back(_descTestDataFactory->createNonDefaultEnergyDesc());

    DeserializedSimulation deserializedSimulationBefore{.mainData = data};
    SerializedSimulation serializedSimulation;
    ASSERT_TRUE(_serializerService->serializeSimulationToStrings(serializedSimulation, deserializedSimulationBefore));

    std::string rawArchive;
    {
        std::stringstream compressedInput(serializedSimulation.mainData);
        zstr::istream decompressedInput(compressedInput, std::ios::binary);
        rawArchive.assign(std::istreambuf_iterator<char>(decompressedInput), std::istreambuf_iterator<char>());
    }

    std::string versionPrefix;
    {
        std::stringstream versionStream;
        cereal::PortableBinaryOutputArchive archive(versionStream);
        archive(Const::ProgramVersion);
        versionPrefix = versionStream.str();
    }
    ASSERT_GE(rawArchive.size(), versionPrefix.size());
    ASSERT_EQ(rawArchive.substr(0, versionPrefix.size()), versionPrefix);

    SerializedSimulation legacySerializedSimulation = serializedSimulation;
    {
        std::stringstream compressedOutput;
        zstr::ostream compressedArchive(compressedOutput, std::ios::binary);
        auto const legacyRawArchive = rawArchive.substr(versionPrefix.size());
        compressedArchive.write(legacyRawArchive.data(), static_cast<std::streamsize>(legacyRawArchive.size()));
        compressedArchive.flush();
        legacySerializedSimulation.mainData = compressedOutput.str();
    }

    DeserializedSimulation deserializedSimulationAfter;
    EXPECT_TRUE(_serializerService->deserializeSimulationFromStrings(deserializedSimulationAfter, legacySerializedSimulation));
    EXPECT_TRUE(_descTestDataFactory->compare(deserializedSimulationBefore.mainData, deserializedSimulationAfter.mainData));
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
