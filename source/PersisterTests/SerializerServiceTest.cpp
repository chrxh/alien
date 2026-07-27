#include <filesystem>
#include <fstream>
#include <ranges>
#include <sstream>

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
        _testDirectory = std::filesystem::temp_directory_path() / "alien-serializer-service-tests";
        std::filesystem::create_directories(_testDirectory);
    }

    ~SerializerServiceTests() override { std::filesystem::remove_all(_testDirectory); }

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
    ColorSamples createOverallSample(double base)
    {
        ColorSamples result;
        result.timestep = base + 1;
        result.systemClock = base + 2;

        // The same set of color combinations in every sample, so the column-wise round-trip is exact
        for (auto&& [colorBitset, offset] : {std::pair{0x1u, 3.0}, std::pair{0x6u, 12.0}}) {
            ColorOverallDataPoint colorPoint;
            colorPoint.numCreatures = base + offset;
            colorPoint.numGenomes = base + offset + 1;
            colorPoint.sumCreatureCells = base + offset + 2;
            colorPoint.sumCreatureGenerations = base + offset + 3;
            colorPoint.sumGenomeNodes = base + offset + 4;
            colorPoint.sumMutationRates = base + offset + 5;
            colorPoint.sumCreatureEnergy = base + offset + 6;
            colorPoint.numCreatedCreatures = base + offset + 7;
            colorPoint.totalMutations = base + offset + 8;
            colorPoint.totalAttackedEnergy = base + offset + 9;
            colorPoint.totalMuscleActivity = base + offset + 10;
            result.data.emplace(colorBitset, colorPoint);
        }
        return result;
    }

    LineageSample createLineageSample(double base)
    {
        LineageSample result;
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
        result.data.totalAttackedEnergy = base + 13;
        result.data.totalMuscleActivity = base + 14;
        return result;
    }

    LineageSample createLineageSample(double base, double numCreatures)
    {
        auto result = createLineageSample(base);
        result.data.numCreatures = numCreatures;
        return result;
    }


    void compare(ColorSamples const& expected, ColorSamples const& actual)
    {
        EXPECT_EQ(expected.timestep, actual.timestep);
        EXPECT_EQ(expected.systemClock, actual.systemClock);

        ASSERT_EQ(expected.data.size(), actual.data.size());
        for (auto const& [colorBitset, expectedColor] : expected.data) {
            auto actualColorIt = actual.data.find(colorBitset);
            ASSERT_TRUE(actualColorIt != actual.data.end());
            auto const& actualColor = actualColorIt->second;
            EXPECT_EQ(expectedColor.numCreatures, actualColor.numCreatures);
            EXPECT_EQ(expectedColor.numGenomes, actualColor.numGenomes);
            EXPECT_EQ(expectedColor.sumCreatureCells, actualColor.sumCreatureCells);
            EXPECT_EQ(expectedColor.sumCreatureGenerations, actualColor.sumCreatureGenerations);
            EXPECT_EQ(expectedColor.sumGenomeNodes, actualColor.sumGenomeNodes);
            EXPECT_EQ(expectedColor.sumMutationRates, actualColor.sumMutationRates);
            EXPECT_EQ(expectedColor.sumCreatureEnergy, actualColor.sumCreatureEnergy);
            EXPECT_EQ(expectedColor.numCreatedCreatures, actualColor.numCreatedCreatures);
            EXPECT_EQ(expectedColor.totalMutations, actualColor.totalMutations);
            EXPECT_EQ(expectedColor.totalAttackedEnergy, actualColor.totalAttackedEnergy);
            EXPECT_EQ(expectedColor.totalMuscleActivity, actualColor.totalMuscleActivity);
        }
    }

    void compare(LineageSample const& expected, LineageSample const& actual)
    {
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
        EXPECT_EQ(expected.data.totalAttackedEnergy, actual.data.totalAttackedEnergy);
        EXPECT_EQ(expected.data.totalMuscleActivity, actual.data.totalMuscleActivity);
    }

    void compare(StatisticsHistoryData const& expected, StatisticsHistoryData const& actual)
    {
        ASSERT_EQ(expected.colors.size(), actual.colors.size());
        for (auto const& [expectedSample, actualSample] : std::views::zip(expected.colors, actual.colors)) {
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

    std::filesystem::path getSettingsFilename(std::filesystem::path const& simulationFilename)
    {
        auto result = simulationFilename;
        return result.replace_extension(std::filesystem::path(".settings.json"));
    }

    void writeFile(std::filesystem::path const& filename, std::string const& content)
    {
        std::ofstream stream(filename, std::ios::binary);
        stream << content;
    }

    std::string readFile(std::filesystem::path const& filename)
    {
        std::ifstream stream(filename, std::ios::binary);
        std::stringstream result;
        result << stream.rdbuf();
        return result.str();
    }

    DescTestDataFactory* _descTestDataFactory;
    SerializerService* _serializerService;
    std::filesystem::path _testDirectory;
};

TEST_F(SerializerServiceTests, simulationFiles)
{
    auto filename = _testDirectory / "simulation.sim";

    DeserializedSimulation before;
    before.mainData._energies.emplace_back(_descTestDataFactory->createNonDefaultEnergyDesc());
    before.auxiliaryData.timestep = 1234;
    before.auxiliaryData.realTime = std::chrono::milliseconds(5678);
    before.auxiliaryData.zoom = 3.5f;
    before.auxiliaryData.center = {111.0f, 222.0f};
    before.auxiliaryData.worldSize = {700, 300};
    before.statistics.colors.emplace_back(createOverallSample(100));

    ASSERT_TRUE(_serializerService->serializeSimulationToFiles(filename, before));

    DeserializedSimulation after;
    ASSERT_TRUE(_serializerService->deserializeSimulationFromFiles(after, filename));

    EXPECT_EQ(before.auxiliaryData.timestep, after.auxiliaryData.timestep);
    EXPECT_EQ(before.auxiliaryData.realTime, after.auxiliaryData.realTime);
    EXPECT_EQ(before.auxiliaryData.zoom, after.auxiliaryData.zoom);
    EXPECT_EQ(before.auxiliaryData.center, after.auxiliaryData.center);
    EXPECT_EQ(before.auxiliaryData.worldSize, after.auxiliaryData.worldSize);
    EXPECT_TRUE(_descTestDataFactory->compare(before.mainData, after.mainData));
    compare(before.statistics, after.statistics);

    EXPECT_TRUE(_serializerService->deleteSimulation(filename));
    EXPECT_FALSE(std::filesystem::exists(filename));
    EXPECT_FALSE(std::filesystem::exists(getSettingsFilename(filename)));
}

TEST_F(SerializerServiceTests, settingsFileEqualsSimulationParametersFile)
{
    auto simulationFilename = _testDirectory / "simulation.sim";
    auto parametersFilename = _testDirectory / "simulation.parameters";

    DeserializedSimulation simulation;
    simulation.auxiliaryData.timestep = 1234;
    simulation.auxiliaryData.worldSize = {700, 300};

    ASSERT_TRUE(_serializerService->serializeSimulationToFiles(simulationFilename, simulation));
    ASSERT_TRUE(_serializerService->serializeSimulationParametersToFile(parametersFilename, simulation.auxiliaryData.simulationParameters));

    auto settings = readFile(getSettingsFilename(simulationFilename));
    EXPECT_EQ(readFile(parametersFilename), settings);
    EXPECT_NE(std::string::npos, settings.find(Const::ProgramVersion));
}

TEST_F(SerializerServiceTests, legacySimulationFiles)
{
    auto filename = _testDirectory / "simulation.sim";

    Desc mainData;
    mainData._energies.emplace_back(_descTestDataFactory->createNonDefaultEnergyDesc());
    ASSERT_TRUE(_serializerService->serializeContentToFile(filename, mainData));

    // Older versions stored the general settings in the settings file
    writeFile(
        getSettingsFilename(filename),
        R"({
            "General": {
                "Version": ")"
            + Const::ProgramVersion + R"(",
                "Time step": "1234",
                "Real time": "5678",
                "Zoom": "3.50000000",
                "Center": {"X": "111.00000000", "Y": "222.00000000"},
                "World size": {"X": "700", "Y": "300"}
            }
        })");

    DeserializedSimulation loaded;
    ASSERT_TRUE(_serializerService->deserializeSimulationFromFiles(loaded, filename));

    EXPECT_EQ(uint64_t{1234}, loaded.auxiliaryData.timestep);
    EXPECT_EQ(std::chrono::milliseconds(5678), loaded.auxiliaryData.realTime);
    EXPECT_EQ(3.5f, loaded.auxiliaryData.zoom);
    EXPECT_EQ(RealVector2D(111.0f, 222.0f), loaded.auxiliaryData.center);
    EXPECT_EQ(IntVector2D(700, 300), loaded.auxiliaryData.worldSize);
    EXPECT_TRUE(_descTestDataFactory->compare(mainData, loaded.mainData));
    EXPECT_TRUE(loaded.statistics.colors.empty());
    EXPECT_TRUE(loaded.statistics.lineages.empty());
}

TEST_F(SerializerServiceTests, statisticsHistory)
{
    DeserializedSimulation before;
    for (int i = 0; i < 5; ++i) {
        before.statistics.colors.emplace_back(createOverallSample(toDouble(i) * 100));
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

TEST_F(SerializerServiceTests, statisticsHistoryWithManyLineages)
{
    auto constexpr NumLineages = 260;
    auto constexpr MaxSavedLineages = 250;

    DeserializedSimulation before;
    for (uint32_t lineageId = 1; lineageId <= NumLineages; ++lineageId) {
        // The higher the lineage id, the more creatures; the history samples suggest the opposite order
        for (uint32_t i = 0; i < lineageId; ++i) {
            before.mainData._creatures.emplace_back(CreatureDesc().lineageId(toInt(lineageId)));
        }
        before.statistics.lineages.emplace(lineageId, std::vector{createLineageSample(1000, toDouble(NumLineages - lineageId))});
    }

    SerializedSimulation serialized;
    ASSERT_TRUE(_serializerService->serializeSimulationToStrings(serialized, before));

    DeserializedSimulation after;
    ASSERT_TRUE(_serializerService->deserializeSimulationFromStrings(after, serialized));

    // Only the lineages with the most creatures are saved
    StatisticsHistoryData expected;
    for (uint32_t lineageId = NumLineages - MaxSavedLineages + 1; lineageId <= NumLineages; ++lineageId) {
        expected.lineages.emplace(lineageId, before.statistics.lineages.at(lineageId));
    }
    compare(expected, after.statistics);
}

TEST_F(SerializerServiceTests, statisticsHistoryWithDeduplicatedColorTimelines)
{
    // Many color combinations share the same timeline; a few carry distinct data
    auto createSample = [](double base) {
        ColorSamples result;
        result.timestep = base + 1;
        result.systemClock = base + 2;
        for (uint32_t colorBitset = 1; colorBitset < 64; ++colorBitset) {
            ColorOverallDataPoint colorPoint;
            if (colorBitset == 0x1u) {
                colorPoint.numCreatures = base + 100;
            } else if (colorBitset == 0x2u) {
                colorPoint.numCreatures = base + 200;
            }
            result.data.emplace(colorBitset, colorPoint);
        }
        return result;
    };

    DeserializedSimulation before;
    for (int i = 0; i < 5; ++i) {
        before.statistics.colors.emplace_back(createSample(toDouble(i) * 100));
    }

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

    EXPECT_TRUE(after.statistics.colors.empty());
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
