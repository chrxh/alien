#include <stdexcept>

#include <gtest/gtest.h>

#include <boost/json.hpp>

#include <EngineTestData/DescTestDataFactory.h>

#include <PersisterInterface/SerializerService.h>

class JsonSerializationTests : public ::testing::Test
{
protected:
    template <typename T>
    T roundTrip(T const& desc, bool omitDefaultValues)
    {
        auto json = boost::json::parse(boost::json::serialize(SerializerService::get().serializeToJson(desc, omitDefaultValues)));
        T result;
        SerializerService::get().deserializeFromJson(result, json);
        return result;
    }

    std::string getErrorMessage(boost::json::value const& json)
    {
        try {
            GenomeDesc genome;
            SerializerService::get().deserializeFromJson(genome, json);
        } catch (std::invalid_argument const& exception) {
            return exception.what();
        }
        return {};
    }
};

using ObjectParameter = DescTestDataFactory::ObjectParameter;
class JsonSerializationTests_AllObjectTypes
    : public JsonSerializationTests
    , public testing::WithParamInterface<ObjectParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    JsonSerializationTests_AllObjectTypes,
    JsonSerializationTests_AllObjectTypes,
    ::testing::ValuesIn(DescTestDataFactory::get().getAllObjectParameters()));

TEST_P(JsonSerializationTests_AllObjectTypes, object)
{
    auto object = DescTestDataFactory::get().createNonDefaultObjectDesc(GetParam());

    EXPECT_TRUE(object == roundTrip(object, true));
    EXPECT_TRUE(object == roundTrip(object, false));
}

using NodeParameter = DescTestDataFactory::NodeParameter;
class JsonSerializationTests_AllNodeTypes
    : public JsonSerializationTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    JsonSerializationTests_AllNodeTypes,
    JsonSerializationTests_AllNodeTypes,
    ::testing::ValuesIn(DescTestDataFactory::get().getAllNodeParameters()));

TEST_P(JsonSerializationTests_AllNodeTypes, creatureAndGenome)
{
    auto [creature, genome] = DescTestDataFactory::get().createNonDefaultCreatureDesc(GetParam());

    EXPECT_TRUE(creature == roundTrip(creature, true));
    EXPECT_TRUE(genome == roundTrip(genome, true));
    EXPECT_TRUE(genome == roundTrip(genome, false));
}

TEST_F(JsonSerializationTests, energyParticle)
{
    auto energy = DescTestDataFactory::get().createNonDefaultEnergyDesc();

    EXPECT_TRUE(energy == roundTrip(energy, true));
}

TEST_F(JsonSerializationTests, idsAsStrings)
{
    auto object = ObjectDesc().id(18446744073709551615ull).type(CellDesc().creatureId(12345678901234567890ull));

    auto json = SerializerService::get().serializeToJson(object).as_object();

    EXPECT_EQ("18446744073709551615", json.at("id").as_string());
    EXPECT_EQ("12345678901234567890", json.at("type").at("cell").at("creatureId").as_string());
}

TEST_F(JsonSerializationTests, omitDefaultValues)
{
    auto json = SerializerService::get().serializeToJson(EnergyDesc().id(1).energy(5.0f)).as_object();

    EXPECT_EQ(2, json.size());
    EXPECT_EQ(5.0, json.at("energy").as_double());
}

TEST_F(JsonSerializationTests, partialGenome)
{
    auto json =
        boost::json::parse(R"({"name": "worm", "genes": [{"nodes": [{"cellType": {"depot": {"storageLimit": 3}}, "constructor": {"geneIndex": 1}}, {}]}]})");

    GenomeDesc genome;
    SerializerService::get().deserializeFromJson(genome, json);

    EXPECT_EQ("worm", genome._name);
    ASSERT_EQ(1, genome._genes.size());
    ASSERT_EQ(2, genome._genes.at(0)._nodes.size());
    auto const& node = genome._genes.at(0)._nodes.at(0);
    ASSERT_TRUE(std::holds_alternative<DepotGenomeDesc>(node._cellType));
    EXPECT_EQ(3.0f, std::get<DepotGenomeDesc>(node._cellType)._storageLimit);
    ASSERT_TRUE(node._constructor.has_value());
    EXPECT_EQ(1, node._constructor->_geneIndex);
    EXPECT_TRUE(genome._genes.at(0)._nodes.at(1) == NodeDesc());
    EXPECT_TRUE(genome._mutationRates == MutationRatesDesc());
}

TEST_F(JsonSerializationTests, patch)
{
    auto object = DescTestDataFactory::get().createNonDefaultObjectDesc({.objectType = ObjectType_Cell, .cellType = CellType_Depot});
    auto expectedObject = object;
    expectedObject._color = 5;
    std::get<CellDesc>(expectedObject._type)._usableEnergy = 7.0f;

    SerializerService::get().deserializeFromJson(object, boost::json::parse(R"({"color": 5, "type": {"cell": {"usableEnergy": 7}}})"), true);

    EXPECT_TRUE(expectedObject == object);
}

TEST_F(JsonSerializationTests, unknownField)
{
    auto message = getErrorMessage(boost::json::parse(R"({"genes": [{"nodes": [{"colour": 2}]}]})"));

    EXPECT_NE(std::string::npos, message.find("genes[0].nodes[0].colour"));
    EXPECT_NE(std::string::npos, message.find("color"));
}

TEST_F(JsonSerializationTests, unknownVariantAlternative)
{
    auto message = getErrorMessage(boost::json::parse(R"({"genes": [{"nodes": [{"cellType": {"teleporter": {}}}]}]})"));

    EXPECT_NE(std::string::npos, message.find("teleporter"));
    EXPECT_NE(std::string::npos, message.find("depot"));
}

TEST_F(JsonSerializationTests, wrongType)
{
    auto message = getErrorMessage(boost::json::parse(R"({"frontAngle": "north"})"));

    EXPECT_NE(std::string::npos, message.find("frontAngle"));
}

TEST_F(JsonSerializationTests, wrongSizeOfFixedSizeMember)
{
    auto message = getErrorMessage(boost::json::parse(R"({"genes": [{"nodes": [{"neuralNetwork": {"biases": [1.0]}}]}]})"));

    EXPECT_NE(std::string::npos, message.find("biases"));
}

TEST_F(JsonSerializationTests, format)
{
    auto format = SerializerService::get().getJsonFormatOfGenome().as_object();

    auto const& node = format.at("genes").as_array().at(0).at("nodes").as_array().at(0).as_object();
    auto const& cellTypes = node.at("cellType").at("oneOf").as_object();
    EXPECT_TRUE(cellTypes.contains("depot"));
    EXPECT_TRUE(cellTypes.contains("sensor"));
    EXPECT_TRUE(node.contains("constructor"));
    EXPECT_TRUE(cellTypes.at("sensor").as_object().at("mode").as_object().contains("oneOf"));
}
