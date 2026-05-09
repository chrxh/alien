#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>

#include <EngineTestData/DescTestDataFactory.h>

#include <PersisterInterface/SerializerService.h>

#include <sstream>


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

namespace
{
    DataPoint createDataPoint(double valueBase, double summedValue)
    {
        DataPoint result;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result.values[i] = valueBase + i;
        }
        result.summedValues = summedValue;
        return result;
    }

    std::vector<std::string> splitCsvLine(std::string const& line)
    {
        std::vector<std::string> result;
        boost::split(result, line, boost::is_any_of(","));
        for (auto& entry : result) {
            boost::algorithm::trim(entry);
        }
        return result;
    }

    int findColumn(std::vector<std::string> const& headerEntries, std::string const& column)
    {
        for (int i = 0; i < headerEntries.size(); ++i) {
            if (headerEntries.at(i) == column) {
                return i;
            }
        }
        return -1;
    }

    std::string createLegacyStatisticsCsv()
    {
        std::stringstream result;
        result << "Time step";
        for (int i = 0; i < MAX_COLORS; ++i) {
            result << ", Depot activities (color " << i << ")";
        }
        result << ", Depot activities (accumulated)";
        for (int i = 0; i < MAX_COLORS; ++i) {
            result << ", Defender activities (color " << i << ")";
        }
        result << ", Defender activities (accumulated)\n";

        result << "42";
        for (int i = 0; i < MAX_COLORS; ++i) {
            result << "," << 100 + i;
        }
        result << ",1000";
        for (int i = 0; i < MAX_COLORS; ++i) {
            result << "," << 200 + i;
        }
        result << ",2000\n";
        return result.str();
    }
}

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

TEST_F(SerializerServiceTests, statisticsSerializationLabelsDefenderAndDepotActivitiesCorrectly)
{
    DeserializedSimulation data;
    DataPointCollection dataPoints;
    dataPoints.time = 42;
    dataPoints.numDefenderActivities = createDataPoint(100, 1000);
    dataPoints.numDepotActivities = createDataPoint(200, 2000);
    data.statistics.emplace_back(dataPoints);

    SerializedSimulation serializedSimulation;
    ASSERT_TRUE(_serializerService->serializeSimulationToStrings(serializedSimulation, data));

    std::stringstream stream(serializedSimulation.statistics);
    std::string header;
    std::string row;
    ASSERT_TRUE(static_cast<bool>(std::getline(stream, header)));
    ASSERT_TRUE(static_cast<bool>(std::getline(stream, row)));

    auto headerEntries = splitCsvLine(header);
    auto rowEntries = splitCsvLine(row);
    auto defenderColumn = findColumn(headerEntries, "Defender activities (accumulated)");
    auto depotColumn = findColumn(headerEntries, "Depot activities (accumulated)");
    ASSERT_NE(-1, defenderColumn);
    ASSERT_NE(-1, depotColumn);
    ASSERT_LT(defenderColumn, rowEntries.size());
    ASSERT_LT(depotColumn, rowEntries.size());
    EXPECT_DOUBLE_EQ(1000, std::stod(rowEntries.at(defenderColumn)));
    EXPECT_DOUBLE_EQ(2000, std::stod(rowEntries.at(depotColumn)));
}

TEST_F(SerializerServiceTests, statisticsDeserializationSupportsLegacyDefenderAndDepotActivityOrder)
{
    DeserializedSimulation data;
    SerializedSimulation serializedSimulation;
    ASSERT_TRUE(_serializerService->serializeSimulationToStrings(serializedSimulation, data));
    serializedSimulation.statistics = createLegacyStatisticsCsv();

    DeserializedSimulation deserializedSimulation;
    ASSERT_TRUE(_serializerService->deserializeSimulationFromStrings(deserializedSimulation, serializedSimulation));

    ASSERT_EQ(1, deserializedSimulation.statistics.size());
    auto const& dataPoints = deserializedSimulation.statistics.at(0);
    EXPECT_DOUBLE_EQ(100, dataPoints.numDefenderActivities.values[0]);
    EXPECT_DOUBLE_EQ(1000, dataPoints.numDefenderActivities.summedValues);
    EXPECT_DOUBLE_EQ(200, dataPoints.numDepotActivities.values[0]);
    EXPECT_DOUBLE_EQ(2000, dataPoints.numDepotActivities.summedValues);
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
