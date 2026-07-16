#include <ranges>

#include <gtest/gtest.h>

#include <Base/Resources.h>
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

namespace
{
    OverallSample createOverallSample(double base)
    {
        OverallSample result;
        result.time = base;
        result.timestep = base + 1;
        result.systemClock = base + 2;
        result.data.numCreatures = base + 3;
        result.data.averageCreatureCells = base + 4;
        result.data.averageGenomeNodes = base + 5;
        result.data.creatureEnergy = base + 6;
        result.data.averageMutationRate = base + 7;
        result.data.averageGeneration = base + 8;
        result.data.numLineages = base + 9;
        result.data.numSolidObjects = base + 10;
        result.data.numFluidObjects = base + 11;
        result.data.numCellObjects = base + 12;
        result.data.numEnergyParticles = base + 13;
        result.data.accumCreatedCreatures = base + 14;
        result.data.accumMutations = base + 15;
        return result;
    }

    LineageSample createLineageSample(double base)
    {
        LineageSample result;
        result.time = base;
        result.timestep = base + 1;
        result.systemClock = base + 2;
        result.data.colorBitset = static_cast<uint32_t>(base) + 3;
        result.data.representativeCellId = (1ull << 60) + static_cast<uint64_t>(base);
        result.data.numCreatures = base + 4;
        result.data.numGenomes = base + 5;
        result.data.sumCreatureCells = base + 6;
        result.data.sumCreatureGenerations = base + 7;
        result.data.sumGenomeNodes = base + 8;
        result.data.sumMutationRates = base + 9;
        result.data.sumCreatureEnergy = base + 10;
        result.data.numCreatedCreatures = base + 11;
        result.data.totalMutations = base + 12;
        return result;
    }

    void compare(OverallSample const& expected, OverallSample const& actual)
    {
        EXPECT_EQ(expected.time, actual.time);
        EXPECT_EQ(expected.timestep, actual.timestep);
        EXPECT_EQ(expected.systemClock, actual.systemClock);
        EXPECT_EQ(expected.data.numCreatures, actual.data.numCreatures);
        EXPECT_EQ(expected.data.averageCreatureCells, actual.data.averageCreatureCells);
        EXPECT_EQ(expected.data.averageGenomeNodes, actual.data.averageGenomeNodes);
        EXPECT_EQ(expected.data.creatureEnergy, actual.data.creatureEnergy);
        EXPECT_EQ(expected.data.averageMutationRate, actual.data.averageMutationRate);
        EXPECT_EQ(expected.data.averageGeneration, actual.data.averageGeneration);
        EXPECT_EQ(expected.data.numLineages, actual.data.numLineages);
        EXPECT_EQ(expected.data.numSolidObjects, actual.data.numSolidObjects);
        EXPECT_EQ(expected.data.numFluidObjects, actual.data.numFluidObjects);
        EXPECT_EQ(expected.data.numCellObjects, actual.data.numCellObjects);
        EXPECT_EQ(expected.data.numEnergyParticles, actual.data.numEnergyParticles);
        EXPECT_EQ(expected.data.accumCreatedCreatures, actual.data.accumCreatedCreatures);
        EXPECT_EQ(expected.data.accumMutations, actual.data.accumMutations);
    }

    void compare(LineageSample const& expected, LineageSample const& actual)
    {
        EXPECT_EQ(expected.time, actual.time);
        EXPECT_EQ(expected.timestep, actual.timestep);
        EXPECT_EQ(expected.systemClock, actual.systemClock);
        EXPECT_EQ(expected.data.colorBitset, actual.data.colorBitset);
        EXPECT_EQ(expected.data.representativeCellId, actual.data.representativeCellId);
        EXPECT_EQ(expected.data.numCreatures, actual.data.numCreatures);
        EXPECT_EQ(expected.data.numGenomes, actual.data.numGenomes);
        EXPECT_EQ(expected.data.sumCreatureCells, actual.data.sumCreatureCells);
        EXPECT_EQ(expected.data.sumCreatureGenerations, actual.data.sumCreatureGenerations);
        EXPECT_EQ(expected.data.sumGenomeNodes, actual.data.sumGenomeNodes);
        EXPECT_EQ(expected.data.sumMutationRates, actual.data.sumMutationRates);
        EXPECT_EQ(expected.data.sumCreatureEnergy, actual.data.sumCreatureEnergy);
        EXPECT_EQ(expected.data.numCreatedCreatures, actual.data.numCreatedCreatures);
        EXPECT_EQ(expected.data.totalMutations, actual.data.totalMutations);
    }

    void compare(StatisticsHistoryData const& expected, StatisticsHistoryData const& actual)
    {
        ASSERT_EQ(expected.overall.size(), actual.overall.size());
        for (auto const& [expectedSample, actualSample] : std::views::zip(expected.overall, actual.overall)) {
            compare(expectedSample, actualSample);
        }
        ASSERT_EQ(expected.lineages.size(), actual.lineages.size());
        for (auto const& [lineageId, expectedSamples] : expected.lineages) {
            auto actualSamplesIt = actual.lineages.find(lineageId);
            ASSERT_TRUE(actualSamplesIt != actual.lineages.end());
            ASSERT_EQ(expectedSamples.size(), actualSamplesIt->second.size());
            for (auto const& [expectedSample, actualSample] : std::views::zip(expectedSamples, actualSamplesIt->second)) {
                compare(expectedSample, actualSample);
            }
        }
    }
}

TEST_F(SerializerServiceTests, statisticsHistory)
{
    DeserializedSimulation before;
    for (int i = 0; i < 5; ++i) {
        before.statistics.overall.emplace_back(createOverallSample(toDouble(i) * 100));
    }
    before.statistics.lineages.emplace(7, std::vector{createLineageSample(1000), createLineageSample(2000)});
    before.statistics.lineages.emplace(42, std::vector{createLineageSample(3000)});
    before.statistics.lineages.emplace(43, std::vector<LineageSample>{});

    SerializedSimulation serialized;
    ASSERT_TRUE(_serializerService->serializeSimulationToStrings(serialized, before));

    DeserializedSimulation after;
    ASSERT_TRUE(_serializerService->deserializeSimulationFromStrings(after, serialized));

    compare(before.statistics, after.statistics);
}

TEST_F(SerializerServiceTests, emptyStatisticsHistory)
{
    DeserializedSimulation before;

    SerializedSimulation serialized;
    ASSERT_TRUE(_serializerService->serializeSimulationToStrings(serialized, before));

    DeserializedSimulation after;
    ASSERT_TRUE(_serializerService->deserializeSimulationFromStrings(after, serialized));

    EXPECT_TRUE(after.statistics.overall.empty());
    EXPECT_TRUE(after.statistics.lineages.empty());
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
