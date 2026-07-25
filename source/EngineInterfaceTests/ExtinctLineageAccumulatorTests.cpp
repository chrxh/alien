#include <gtest/gtest.h>

#include <Base/Definitions.h>

#include <EngineInterface/ExtinctLineageAccumulator.h>
#include <EngineInterface/StatisticsConverterService.h>
#include <EngineInterface/StatisticsEntry.h>

class ExtinctLineageAccumulatorTests : public ::testing::Test
{
public:
    ExtinctLineageAccumulatorTests() = default;
    virtual ~ExtinctLineageAccumulatorTests() = default;

protected:
    LineageStatisticsEntry createLineageEntry(uint32_t lineageId, uint32_t colorBitset, uint64_t numCreatedCreatures, double totalMutations) const
    {
        LineageStatisticsEntry result;
        result.lineageId = lineageId;
        result.colorBitset = colorBitset;
        result.numCreatures = 1;
        result.numCreatedCreatures = numCreatedCreatures;
        result.totalMutations = totalMutations;
        return result;
    }

    DataPointCollection addDataPoint(std::vector<LineageStatisticsEntry> const& lineageEntries)
    {
        StatisticsEntry statisticsEntry;
        statisticsEntry.lineageEntries = lineageEntries;
        auto result = StatisticsConverterService::get().convert(statisticsEntry, 0, 0.0);
        _accumulator.addExtinctLineageValues(result);
        return result;
    }

    ExtinctLineageAccumulator _accumulator;
};

TEST_F(ExtinctLineageAccumulatorTests, livingLineagesOnly)
{
    auto dataPoints = addDataPoint({createLineageEntry(1, 0b01, 10, 2.0), createLineageEntry(2, 0b01, 20, 3.0)});

    EXPECT_EQ(1u, dataPoints.overall.size());
    EXPECT_DOUBLE_EQ(30.0, dataPoints.overall.at(0b01).numCreatedCreatures);
    EXPECT_DOUBLE_EQ(5.0, dataPoints.overall.at(0b01).totalMutations);
}

TEST_F(ExtinctLineageAccumulatorTests, extinctLineageStillContributes)
{
    addDataPoint({createLineageEntry(1, 0b01, 10, 2.0), createLineageEntry(2, 0b01, 20, 3.0)});
    auto dataPoints = addDataPoint({createLineageEntry(2, 0b01, 25, 4.0)});

    EXPECT_DOUBLE_EQ(35.0, dataPoints.overall.at(0b01).numCreatedCreatures);
    EXPECT_DOUBLE_EQ(6.0, dataPoints.overall.at(0b01).totalMutations);
}

TEST_F(ExtinctLineageAccumulatorTests, extinctLineageUsesLastKnownValues)
{
    addDataPoint({createLineageEntry(1, 0b01, 10, 2.0)});
    addDataPoint({createLineageEntry(1, 0b01, 15, 5.0)});
    auto dataPoints = addDataPoint({});

    EXPECT_DOUBLE_EQ(15.0, dataPoints.overall.at(0b01).numCreatedCreatures);
    EXPECT_DOUBLE_EQ(5.0, dataPoints.overall.at(0b01).totalMutations);
    EXPECT_DOUBLE_EQ(0.0, dataPoints.overall.at(0b01).numCreatures);
}

TEST_F(ExtinctLineageAccumulatorTests, extinctLineagesKeptPerColorBitset)
{
    addDataPoint({createLineageEntry(1, 0b01, 10, 2.0), createLineageEntry(2, 0b10, 20, 3.0)});
    auto dataPoints = addDataPoint({});

    EXPECT_DOUBLE_EQ(10.0, dataPoints.overall.at(0b01).numCreatedCreatures);
    EXPECT_DOUBLE_EQ(2.0, dataPoints.overall.at(0b01).totalMutations);
    EXPECT_DOUBLE_EQ(20.0, dataPoints.overall.at(0b10).numCreatedCreatures);
    EXPECT_DOUBLE_EQ(3.0, dataPoints.overall.at(0b10).totalMutations);
}

TEST_F(ExtinctLineageAccumulatorTests, extinctLineageUsesLastKnownColorBitset)
{
    addDataPoint({createLineageEntry(1, 0b01, 10, 2.0)});
    addDataPoint({createLineageEntry(1, 0b11, 10, 2.0)});
    auto dataPoints = addDataPoint({});

    EXPECT_EQ(1u, dataPoints.overall.size());
    EXPECT_DOUBLE_EQ(10.0, dataPoints.overall.at(0b11).numCreatedCreatures);
}

TEST_F(ExtinctLineageAccumulatorTests, monotonicallyIncreasingCounters)
{
    auto lastNumCreatedCreatures = 0.0;
    auto lastTotalMutations = 0.0;
    for (uint32_t lineageId = 1; lineageId < 10; ++lineageId) {
        auto dataPoints = addDataPoint({createLineageEntry(lineageId, 0b01, lineageId, toDouble(lineageId))});
        auto const& colorPoint = dataPoints.overall.at(0b01);
        EXPECT_GE(colorPoint.numCreatedCreatures, lastNumCreatedCreatures);
        EXPECT_GE(colorPoint.totalMutations, lastTotalMutations);
        lastNumCreatedCreatures = colorPoint.numCreatedCreatures;
        lastTotalMutations = colorPoint.totalMutations;
    }
}

TEST_F(ExtinctLineageAccumulatorTests, reset)
{
    addDataPoint({createLineageEntry(1, 0b01, 10, 2.0)});
    addDataPoint({});
    _accumulator.reset();
    auto dataPoints = addDataPoint({});

    EXPECT_TRUE(dataPoints.overall.empty());
}
