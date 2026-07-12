#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/LineageHistory.h>
#include <EngineInterface/LineageStatistics.h>

#include "AlienWindow.h"
#include "Definitions.h"
#include "TimelineLiveStatistics.h"

class EvolutionDashboardWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(EvolutionDashboardWindow);

public:
    static auto constexpr NumMetrics = 8;

private:
    EvolutionDashboardWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;

    void updateCellColors();
    void updateDisplayData();
    void processHeader();
    void processKpiCard(std::string const& label, std::string const& value, float delta, int sparklineIndex, float width, float height);
    void processEntitiesCard(float width, float height);
    void processFilterBar();
    void processLineageTable();
    void processTimelineSection();
    void processTimelineHeader();
    void processTimelinePlots();
    void processTimelinePlot(int metricIndex, bool showTimeAxis);

    void validateAndCorrect();

    struct LineageDisplayData
    {
        int64_t id = 0;  //-1 = "all lineages"
        int colorBitset = 0;
        std::array<double, NumMetrics> currentValues = {};
        std::array<std::vector<double>, NumMetrics> series = {};  //only filled for plotted lineages
        std::vector<double> timePoints;                           //only filled for plotted lineages
    };

    std::vector<LineageDisplayData> _lineages;  //table rows from the latest sample, sorted by creature count
    std::vector<LineageDisplayData> _plottedLineages;
    LineageDisplayData _allLineages;
    std::array<double, NumMetrics> _metricWindowDeltas = {};
    std::array<std::vector<double>, 4> _headerSparklines = {};
    std::array<double, 4> _headerDeltas = {};
    bool _lineageMapOverflow = false;

    std::array<uint32_t, MAX_COLORS> _cellColors = {};

    std::set<int64_t> _selectedLineageIds;
    int _colorFilter = 0x3ff;  //bitset for MAX_COLORS cell colors

    using TimelineMode = int;
    enum TimelineMode_
    {
        TimelineMode_RealTime,
        TimelineMode_EntireHistory,
        TimelineMode_LastSteps
    };
    TimelineMode _timelineMode = TimelineMode_EntireHistory;

    using PlotScale = int;
    enum PlotScale_
    {
        PlotScale_Linear,
        PlotScale_Logarithmic
    };
    PlotScale _plotScale = PlotScale_Linear;

    int _lastSteps = 100000;
    float _timelinesHeight = 0;

    struct RebuildKey
    {
        int timelineMode = 0;
        int lastSteps = 0;
        double liveBackTime = 0;
        double historyBackTime = 0;
        std::set<int64_t> selectedLineageIds;

        bool operator==(RebuildKey const&) const = default;
    };
    std::optional<RebuildKey> _lastRebuildKey;

    //live statistics
    TimelineLiveStatistics _timelineLiveStatistics;
    LineageHistoryData _liveLineageHistory;
    std::vector<double> _externalEnergySeries;
    double _timeSinceSimStart = 0;  //in seconds
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
};
