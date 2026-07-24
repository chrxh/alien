#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/StatisticsEntry.h>
#include <EngineInterface/StatisticsHistory.h>

#include "AlienWindow.h"
#include "Definitions.h"
#include "TimelineLiveStatistics.h"

class EvolutionDashboardWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(EvolutionDashboardWindow);

public:
    static auto constexpr NumMetrics = 8;

private:
    struct LineageDisplayData;

    EvolutionDashboardWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;

    void updateCellColors();
    void updateDisplayData();
    void updateTableData();
    void sortLineages();
    void updatePlotData();
    void rebuildPlotSeries(std::vector<DataPointCollection> const& source);
    void rebuildPlotSeries(StatisticsHistoryData const& source);
    void processHeader();
    void processCard(
        std::string const& label,
        std::string const& value,
        std::vector<std::pair<std::string, std::string>> const& subValues,
        float width,
        float height);
    void processFilterBar();
    void processLineageTable();
    void onOpenRepresentativeGenome(uint64_t cellId);
    void processTimelineSection();
    void processTimelineHeader();
    void processTimelinePlots(std::vector<LineageDisplayData const*> const& plottedLineages);
    void processTimelinePlot(std::vector<LineageDisplayData const*> const& plottedLineages, int metricIndex);
    std::pair<double, double> computePlotTimeRange(std::vector<LineageDisplayData const*> const& plottedLineages) const;
    void processTimeAxis(std::vector<LineageDisplayData const*> const& plottedLineages, float rightMargin);
    void drawValuesAtMouseCursor(
        std::vector<double> const& series,
        std::vector<double> const& timePoints,
        std::vector<double> const& systemClockPoints,
        double startTime,
        double endTime,
        double upperBound,
        int fracPartDecimals);

    void validateAndCorrect();

    struct LineageDisplayData
    {
        int64_t id = 0;  //-1 = "all lineages"
        int colorBitset = 0;
        uint64_t representativeCellId = 0;  //cell of a creature with the highest generation; 0 = not available
        std::array<double, NumMetrics> currentValues = {};
        std::array<std::vector<double>, NumMetrics> series = {};  //only filled for plotted lineages
        std::vector<double> timePoints;                           //only filled for plotted lineages
        std::vector<double> systemClockPoints;                    //only filled when the x-axis represents time steps
    };

    std::vector<LineageDisplayData> _lineages;  //table rows from the latest sample, sorted by the selected table column
    std::vector<LineageDisplayData> _plottedLineages;
    LineageDisplayData _allLineages;

    std::array<uint32_t, MAX_COLORS> _cellColors = {};

    std::set<int64_t> _selectedLineageIds;
    int _colorFilter = 0x3ff;  //bitset for MAX_COLORS cell colors

    int _sortColumnIndex = 1;  //0 = lineage column, 1..NumMetrics = metric columns
    bool _sortAscending = false;

    using TimelineMode = int;
    enum TimelineMode_
    {
        TimelineMode_RealTime,
        TimelineMode_LastSteps,
        TimelineMode_EntireHistory
    };
    TimelineMode _timelineMode = TimelineMode_EntireHistory;

    int _lastSteps = 100000;
    float _timeHorizon = 10.0f;  //in seconds
    float _timelinesHeight = 0;
    float _plotHeight = 60.0f;
    std::optional<float> _lastWindowHeight;

    struct RebuildKey
    {
        int timelineMode = 0;
        int lastSteps = 0;
        float timeHorizon = 0;
        int colorFilter = 0;
        double sourceBackTime = 0;
        std::set<int64_t> selectedLineageIds;
        std::vector<double> lineageBackTimes;  //only used for the long-term history where each lineage has its own timeline

        bool operator==(RebuildKey const&) const = default;
    };
    std::optional<RebuildKey> _lastRebuildKey;
    std::optional<double> _lastTableBackTime;
    int _lastTableColorFilter = -1;

    // Live statistics
    TimelineLiveStatistics _timelineLiveStatistics;
    StatisticsEntry _lastStatisticsEntry;  //latest raw engine snapshot for the header's global (non-per-color) counts
    std::optional<int> _lastSessionId;
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
};
