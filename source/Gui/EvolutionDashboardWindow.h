#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/EngineConstants.h>
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
    void processHeader();
    void processKpiCard(
        std::string const& label,
        std::string const& value,
        std::optional<double> delta,
        bool isPercentage,
        int sparklineIndex,
        float width,
        float height);
    void processEntitiesCard(float width, float height);
    void processFilterBar();
    void processLineageTable();
    void processTimelineSection();
    void processTimelineHeader();
    void processTimelinePlots();
    void processTimelinePlot(std::vector<LineageDisplayData const*> const& plottedLineages, int metricIndex, bool showTimeAxis);
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
        std::array<double, NumMetrics> currentValues = {};
        std::array<std::vector<double>, NumMetrics> series = {};  //only filled for plotted lineages
        std::vector<double> timePoints;                           //only filled for plotted lineages
        std::vector<double> systemClockPoints;                    //only filled when the x-axis represents time steps
    };

    std::vector<LineageDisplayData> _lineages;  //table rows from the latest sample, sorted by creature count
    std::vector<LineageDisplayData> _plottedLineages;
    LineageDisplayData _allLineages;
    std::array<std::vector<double>, 3> _headerSparklines = {};
    std::array<std::optional<double>, 3> _headerDeltas = {};

    std::array<uint32_t, MAX_COLORS> _cellColors = {};

    std::set<int64_t> _selectedLineageIds;
    int _colorFilter = 0x3ff;  //bitset for MAX_COLORS cell colors

    using TimelineMode = int;
    enum TimelineMode_
    {
        TimelineMode_RealTime,
        TimelineMode_LastSteps,
        TimelineMode_EntireHistory  //only selectable when all lineages are shown
    };
    TimelineMode _timelineMode = TimelineMode_EntireHistory;

    int _lastSteps = 100000;
    float _timeHorizon = 10.0f;  //in seconds
    float _timelinesHeight = 0;

    struct RebuildKey
    {
        int timelineMode = 0;
        int lastSteps = 0;
        float timeHorizon = 0;
        double liveBackTime = 0;
        double historyBackTime = 0;
        std::set<int64_t> selectedLineageIds;

        bool operator==(RebuildKey const&) const = default;
    };
    std::optional<RebuildKey> _lastRebuildKey;

    // Live statistics
    TimelineLiveStatistics _timelineLiveStatistics;
    std::vector<double> _externalEnergySeries;
    std::vector<double> _externalEnergyTimeSeries;
    double _timeSinceSimStart = 0;  //in seconds
    std::optional<int> _lastSessionId;
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
};
