#include <gtest/gtest.h>

#include <EngineInterface/LineageHistoryService.h>

class LineageHistoryServiceTests : public ::testing::Test
{
protected:
    static LineageStatisticsEntry createEntry(uint32_t lineageId)
    {
        LineageStatisticsEntry result;
        result.lineageId = lineageId;
        return result;
    }

    static RawLineageStatistics createRawStatistics(std::vector<LineageStatisticsEntry> entries)
    {
        RawLineageStatistics result;
        result.entries = std::move(entries);
        return result;
    }
};

TEST_F(LineageHistoryServiceTests, mergeSamples_disjointIds)
{
    LineageSample earlierSample;
    earlierSample.time = 100;
    earlierSample.systemClock = 10;
    earlierSample.entries = {createEntry(1), createEntry(3)};

    LineageSample laterSample;
    laterSample.time = 200;
    laterSample.systemClock = 20;
    laterSample.entries = {createEntry(2), createEntry(4)};

    auto result = LineageHistoryService::mergeSamples(earlierSample, laterSample);

    EXPECT_EQ(100, result.time);
    EXPECT_EQ(15, result.systemClock);
    ASSERT_EQ(4, result.entries.size());
    EXPECT_EQ(1, result.entries.at(0).lineageId);
    EXPECT_EQ(2, result.entries.at(1).lineageId);
    EXPECT_EQ(3, result.entries.at(2).lineageId);
    EXPECT_EQ(4, result.entries.at(3).lineageId);
}

TEST_F(LineageHistoryServiceTests, mergeSamples_commonIds)
{
    auto entry1 = createEntry(7);
    entry1.colorBitset = 0x01;
    entry1.numCreatures = 10;
    entry1.numGenomes = 8;
    entry1.sumCreatureCells = 100;
    entry1.sumCreatureGenerations = 50;
    entry1.sumGenomeNodes = 300;
    entry1.sumMutationRates = 4;
    entry1.sumCreatureEnergy = 1000;
    entry1.numCreatedCreatures = 40;
    entry1.totalMutations = 5;

    auto entry2 = createEntry(7);
    entry2.colorBitset = 0x02;
    entry2.numCreatures = 20;
    entry2.numGenomes = 12;
    entry2.sumCreatureCells = 200;
    entry2.sumCreatureGenerations = 70;
    entry2.sumGenomeNodes = 500;
    entry2.sumMutationRates = 6;
    entry2.sumCreatureEnergy = 3000;
    entry2.numCreatedCreatures = 60;
    entry2.totalMutations = 9;

    LineageSample earlierSample;
    earlierSample.time = 100;
    earlierSample.systemClock = 10;
    earlierSample.entries = {entry1};

    LineageSample laterSample;
    laterSample.time = 200;
    laterSample.systemClock = 20;
    laterSample.entries = {entry2};

    auto result = LineageHistoryService::mergeSamples(earlierSample, laterSample);

    ASSERT_EQ(1, result.entries.size());
    auto const& merged = result.entries.front();
    EXPECT_EQ(7, merged.lineageId);
    EXPECT_EQ(0x03, merged.colorBitset);
    EXPECT_EQ(15, merged.numCreatures);
    EXPECT_EQ(10, merged.numGenomes);
    EXPECT_FLOAT_EQ(150, merged.sumCreatureCells);
    EXPECT_FLOAT_EQ(60, merged.sumCreatureGenerations);
    EXPECT_FLOAT_EQ(400, merged.sumGenomeNodes);
    EXPECT_FLOAT_EQ(5, merged.sumMutationRates);
    EXPECT_FLOAT_EQ(2000, merged.sumCreatureEnergy);
    EXPECT_EQ(60, merged.numCreatedCreatures);  //accumulated values are maximized
    EXPECT_FLOAT_EQ(9, merged.totalMutations);
}

TEST_F(LineageHistoryServiceTests, addSample_sortsEntriesById)
{
    LineageHistoryService service;
    LineageHistory history;

    service.addSample(history, createRawStatistics({createEntry(5), createEntry(1), createEntry(3)}), 100);

    auto data = history.getCopiedData();
    ASSERT_EQ(1, data.size());
    ASSERT_EQ(3, data.front().entries.size());
    EXPECT_EQ(1, data.front().entries.at(0).lineageId);
    EXPECT_EQ(3, data.front().entries.at(1).lineageId);
    EXPECT_EQ(5, data.front().entries.at(2).lineageId);
}

TEST_F(LineageHistoryServiceTests, addSample_skipsSamplesBelowCadence)
{
    LineageHistoryService service;
    LineageHistory history;

    service.addSample(history, createRawStatistics({createEntry(1)}), 100);
    service.addSample(history, createRawStatistics({createEntry(2)}), 105);  //below the default cadence of 10 timesteps
    service.addSample(history, createRawStatistics({createEntry(3)}), 120);

    auto data = history.getCopiedData();
    ASSERT_EQ(2, data.size());
    EXPECT_EQ(100, data.at(0).time);
    EXPECT_EQ(120, data.at(1).time);
}

TEST_F(LineageHistoryServiceTests, addSample_clearsHistoryOnTimeRegression)
{
    LineageHistoryService service;
    LineageHistory history;

    service.addSample(history, createRawStatistics({createEntry(1)}), 1000);
    service.addSample(history, createRawStatistics({createEntry(2)}), 100);

    auto data = history.getCopiedData();
    ASSERT_EQ(1, data.size());
    EXPECT_EQ(100, data.front().time);
    ASSERT_EQ(1, data.front().entries.size());
    EXPECT_EQ(2, data.front().entries.front().lineageId);
}

TEST_F(LineageHistoryServiceTests, addSample_compressesHistory)
{
    LineageHistoryService service;
    LineageHistory history;

    auto timestep = uint64_t(0);
    for (int i = 0; i < 1100; ++i) {
        service.addSample(history, createRawStatistics({createEntry(1)}), timestep);
        timestep += 10;
    }

    auto data = history.getCopiedData();
    EXPECT_LE(data.size(), 1001);
    for (size_t i = 1; i < data.size(); ++i) {
        EXPECT_LT(data.at(i - 1).time, data.at(i).time);
    }
}
