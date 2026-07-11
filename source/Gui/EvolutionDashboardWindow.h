#pragma once

#include <array>
#include <set>
#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/EngineConstants.h>

#include "AlienWindow.h"
#include "Definitions.h"

class EvolutionDashboardWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(EvolutionDashboardWindow);

public:
    static auto constexpr NumMetrics = 9;
    static auto constexpr NumTimePoints = 300;

private:
    EvolutionDashboardWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;

    void processHeader();
    void processKpiCard(std::string const& label, std::string const& value, float delta, int sparklineIndex, float width);
    void processEntitiesCard(float width);
    void processFilterBar();
    void processLineageTable();
    void processTimelineSection();
    void processTimelineHeader();
    void processTimelinePlots();
    void processTimelinePlot(int metricIndex, bool showTimeAxis);

    void generateDummyData();
    void validateAndCorrect();

    struct DummyLineage
    {
        int id = 0;
        int color = 0;
        std::array<double, NumMetrics> currentValues = {};
        std::array<std::vector<double>, NumMetrics> series = {};
    };

    std::vector<DummyLineage> _lineages;
    DummyLineage _allLineages;
    std::vector<double> _timePoints;
    std::array<std::vector<double>, 4> _headerSparklines = {};

    std::set<int> _selectedLineageIds;
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
};
