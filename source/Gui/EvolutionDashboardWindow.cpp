#include "EvolutionDashboardWindow.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <cstdio>

#include <imgui.h>
#include <implot.h>

#include <Base/Definitions.h>
#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr LiveStatisticsDeltaTime = 50;  //in millisec
    auto constexpr MaxLiveHistory = 240.0;        //in seconds
    auto constexpr NumSparklinePoints = 40;

    auto constexpr SparklineWidth = 90.0f;
    auto constexpr SparklineHeight = 26.0f;
    auto constexpr ColorChipSize = 22.0f;
    auto constexpr SwatchSize = 11.0f;
    auto constexpr SwatchGap = 3.0f;
    auto constexpr PlotLabelColumnWidth = 150.0f;
    auto constexpr PlotHeight = 60.0f;
    auto constexpr PlotHeightWithAxis = 80.0f;
    auto constexpr DefaultTimelinesHeight = 380.0f;

    ImColor const CardBackgroundColor = ImColor(0.095f, 0.117f, 0.165f, 1.0f);
    ImColor const CardBorderColor = ImColor(0.165f, 0.196f, 0.270f, 1.0f);
    ImColor const PositiveDeltaColor = ImColor(0.19f, 0.77f, 0.55f, 1.0f);
    ImColor const NegativeDeltaColor = ImColor(0.94f, 0.32f, 0.32f, 1.0f);
    ImColor const SumSeriesColor = ImColor(0.78f, 0.78f, 0.78f, 1.0f);
    struct MetricDef
    {
        char const* tableHeader;
        char const* plotName;
        int decimals;  //-1: scientific notation
    };

    MetricDef const Metrics[EvolutionDashboardWindow::NumMetrics] = {
        {"Creatures", "Creatures", 0},
        {"Avg cells", "Avg cells", 1},
        {"Avg nodes", "Avg nodes", 1},
        {"Sum energy", "Sum energy", 0},
        {"Avg mut. rate", "Avg mutation rate", -1},
        {"Avg age (gen)", "Avg age (generations)", 0},
        {"Created /s", "Created creatures / s", 2},
        {"Mutations /s", "Mutations / s", 1},
    };

    ImColor toImColor(uint32_t rgb, float alpha = 1.0f)
    {
        return ImColor(toInt((rgb >> 16) & 0xff), toInt((rgb >> 8) & 0xff), toInt(rgb & 0xff), toInt(alpha * 255.0f));
    }

    std::string formatMetricValue(double value, int decimals)
    {
        if (decimals == -1) {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "%.1e", value);
            return buffer;
        }
        if (value >= 1e6) {
            return StringHelper::format(toFloat(value / 1e6), 2) + " M";
        }
        if (value >= 1e4) {
            return StringHelper::format(toFloat(value / 1e3), 0) + " K";
        }
        return StringHelper::format(toFloat(value), decimals);
    }

    void rightAlignedText(std::string const& text)
    {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(text.c_str()).x);
        ImGui::TextUnformatted(text.c_str());
    }

    int getFirstColor(int colorBitset)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (colorBitset & (1 << i)) {
                return i;
            }
        }
        return 0;
    }

    float drawColorSwatches(ImDrawList* drawList, ImVec2 const& pos, int colorBitset, float lineHeight, std::array<uint32_t, MAX_COLORS> const& cellColors)
    {
        auto swatchSize = scale(SwatchSize);
        auto gap = scale(SwatchGap);
        auto offsetY = (lineHeight - swatchSize) / 2;
        auto x = pos.x;
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (colorBitset & (1 << i)) {
                drawList->AddRectFilled({x, pos.y + offsetY}, {x + swatchSize, pos.y + offsetY + swatchSize}, toImColor(cellColors.at(i)), scale(3.0f));
                x += swatchSize + gap;
            }
        }
        return x - pos.x;
    }

    double calcRate(double accumValue, double lastAccumValue, double clock, double lastClock)
    {
        auto deltaClock = clock - lastClock;
        if (deltaClock <= 0) {
            return 0.0;
        }
        return std::max(0.0, (accumValue - lastAccumValue) / deltaClock);  //negative deltas occur after accumulated statistics have been reset
    }

    double
    getGlobalMetricValue(OverallDataPointCollection const& dataPoints, OverallDataPointCollection const* lastDataPoints, bool clockFromTime, int metricIndex)
    {
        switch (metricIndex) {
        case 0:
            return dataPoints.overall.numCreatures;
        case 1:
            return dataPoints.overall.averageCreatureCells;
        case 2:
            return dataPoints.overall.averageGenomeNodes;
        case 3:
            return dataPoints.overall.creatureEnergy;
        case 4:
            return dataPoints.overall.averageMutationRate;
        case 5:
            return dataPoints.overall.averageGeneration;
        case 6:
        case 7: {
            if (!lastDataPoints) {
                return 0.0;
            }
            auto clock = clockFromTime ? dataPoints.time : dataPoints.systemClock;
            auto lastClock = clockFromTime ? lastDataPoints->time : lastDataPoints->systemClock;
            if (metricIndex == 6) {
                return calcRate(dataPoints.overall.accumCreatedCreatures, lastDataPoints->overall.accumCreatedCreatures, clock, lastClock);
            } else {
                return calcRate(dataPoints.overall.accumMutations, lastDataPoints->overall.accumMutations, clock, lastClock);
            }
        }
        default:
            return 0.0;
        }
    }

    double getLineageMetricValue(LineageDataPoint const& entry, LineageDataPoint const* lastEntry, double clock, double lastClock, int metricIndex)
    {
        switch (metricIndex) {
        case 0:
            return entry.numCreatures;
        case 1:
            return entry.numCreatures > 0 ? entry.sumCreatureCells / entry.numCreatures : 0.0;
        case 2:
            return entry.numGenomes > 0 ? entry.sumGenomeNodes / entry.numGenomes : 0.0;
        case 3:
            return entry.sumCreatureEnergy;
        case 4:
            return entry.numGenomes > 0 ? entry.sumMutationRates / entry.numGenomes : 0.0;
        case 5:
            return entry.numCreatures > 0 ? entry.sumCreatureGenerations / entry.numCreatures : 0.0;
        case 6:
            return lastEntry ? calcRate(entry.numCreatedCreatures, lastEntry->numCreatedCreatures, clock, lastClock) : 0.0;
        case 7:
            return lastEntry ? calcRate(entry.totalMutations, lastEntry->totalMutations, clock, lastClock) : 0.0;
        default:
            return 0.0;
        }
    }

    LineageDataPoint const* findLineageEntry(DataPointCollection const& sample, uint32_t lineageId)
    {
        auto it = sample.lineages.find(lineageId);
        return it != sample.lineages.end() ? &it->second : nullptr;
    }

    double calcWindowDeltaPercent(std::vector<double> const& series)
    {
        if (series.size() < 2 || std::abs(series.front()) < NEAR_ZERO) {
            return 0.0;
        }
        return (series.back() / series.front() - 1.0) * 100;
    }

    double calcDeltaPercentPerMinute(std::vector<double> const& series, double windowInSeconds)
    {
        if (windowInSeconds < NEAR_ZERO) {
            return 0.0;
        }
        return calcWindowDeltaPercent(series) / (windowInSeconds / 60);
    }
}

EvolutionDashboardWindow::EvolutionDashboardWindow()
    : AlienWindow("Evolution Dashboard", "windows.evolution dashboard", false, true)
{}

void EvolutionDashboardWindow::initIntern()
{
    _timelinesHeight = GlobalSettings::get().getValue("windows.evolution dashboard.timelines height", scale(DefaultTimelinesHeight));
    _timelineMode = GlobalSettings::get().getValue("windows.evolution dashboard.timeline mode", _timelineMode);
    _lastSteps = GlobalSettings::get().getValue("windows.evolution dashboard.last steps", _lastSteps);
    _timeHorizon = GlobalSettings::get().getValue("windows.evolution dashboard.time horizon", _timeHorizon);
    _colorFilter = GlobalSettings::get().getValue("windows.evolution dashboard.color filter", _colorFilter);
    validateAndCorrect();
}

void EvolutionDashboardWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.evolution dashboard.timelines height", _timelinesHeight);
    GlobalSettings::get().setValue("windows.evolution dashboard.timeline mode", _timelineMode);
    GlobalSettings::get().setValue("windows.evolution dashboard.last steps", _lastSteps);
    GlobalSettings::get().setValue("windows.evolution dashboard.time horizon", _timeHorizon);
    GlobalSettings::get().setValue("windows.evolution dashboard.color filter", _colorFilter);
}

void EvolutionDashboardWindow::processBackground()
{
    auto timepoint = std::chrono::steady_clock::now();
    auto duration =
        _lastTimepoint.has_value() ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_lastTimepoint).count()) : 0;
    if (_lastTimepoint && duration <= LiveStatisticsDeltaTime) {
        return;
    }
    _lastTimepoint = timepoint;
    _timeSinceSimStart += toDouble(duration) / 1000;

    auto overallStatistics = _SimulationFacade::get()->getStatisticsEntry();
    _timelineLiveStatistics.update(overallStatistics, _SimulationFacade::get()->getCurrentTimestep());

    _externalEnergySeries.emplace_back(toDouble(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
    auto maxExternalEnergyValues = toInt(std::lround(MaxLiveHistory * 1000 / LiveStatisticsDeltaTime));
    while (toInt(_externalEnergySeries.size()) > maxExternalEnergyValues) {
        _externalEnergySeries.erase(_externalEnergySeries.begin());
    }
}

void EvolutionDashboardWindow::processIntern()
{
    if (!_selectedLineageIds.empty() && _timelineMode == TimelineMode_EntireHistory) {
        _timelineMode = TimelineMode_LastSteps;
    }
    updateCellColors();
    updateDisplayData();
    processHeader();
    processFilterBar();

    if (ImGui::BeginChild("##lineageTable", {0, -_timelinesHeight})) {
        processLineageTable();
    }
    ImGui::EndChild();

    AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _timelinesHeight);

    if (ImGui::BeginChild("##timelineSection", {0, 0})) {
        processTimelineSection();
    }
    ImGui::EndChild();
}

void EvolutionDashboardWindow::updateCellColors()
{
    auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;
    for (int i = 0; i < MAX_COLORS; ++i) {
        _cellColors.at(i) = customizationColors.values[i].toRgbColor();
    }
}

void EvolutionDashboardWindow::updateDisplayData()
{
    //rebuild only when the underlying data or view settings have changed
    auto const& liveHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    auto liveBackTime = !liveHistory.empty() ? liveHistory.back().time : -1.0;
    auto historyBackTime = -1.0;
    if (_timelineMode != TimelineMode_RealTime) {
        auto const& statisticsHistory = _SimulationFacade::get()->getStatisticsHistory();
        std::lock_guard lock(statisticsHistory.getMutex());
        if (!statisticsHistory.getDataRef().empty()) {
            historyBackTime = statisticsHistory.getDataRef().back().time;
        }
    }
    RebuildKey key{_timelineMode, _lastSteps, _timeHorizon, liveBackTime, historyBackTime, _selectedLineageIds};
    if (_lastRebuildKey && *_lastRebuildKey == key) {
        return;
    }
    _lastRebuildKey = std::move(key);

    //table rows from the latest live sample
    _lineages.clear();
    if (!liveHistory.empty()) {
        auto const& lastSample = liveHistory.back();
        auto const* previousSample = liveHistory.size() >= 2 ? &liveHistory.at(liveHistory.size() - 2) : nullptr;
        for (auto const& [lineageId, entry] : lastSample.lineages) {
            LineageDisplayData lineage;
            lineage.id = toInt(lineageId);
            lineage.colorBitset = toInt(entry.colorBitset);
            auto const* previousEntry = previousSample ? findLineageEntry(*previousSample, lineageId) : nullptr;
            for (int i = 0; i < NumMetrics; ++i) {
                lineage.currentValues.at(i) = getLineageMetricValue(entry, previousEntry, lastSample.time, previousSample ? previousSample->time : 0.0, i);
            }
            _lineages.emplace_back(std::move(lineage));
        }
        std::sort(_lineages.begin(), _lineages.end(), [](auto const& lhs, auto const& rhs) { return lhs.currentValues.at(0) > rhs.currentValues.at(0); });
    }

    //data source for the "all lineages" timeline plots
    auto useTimeAsClock = _timelineMode == TimelineMode_RealTime;
    StatisticsHistoryData globalSource;
    if (_timelineMode == TimelineMode_RealTime) {
        globalSource.reserve(liveHistory.size());
        for (auto const& sample : liveHistory) {
            globalSource.emplace_back(sample.toOverallDataPointCollection());
        }
        if (!globalSource.empty()) {
            auto startTime = globalSource.back().time - toDouble(_timeHorizon);
            std::erase_if(globalSource, [&](auto const& dataPoints) { return dataPoints.time < startTime; });
        }
    } else {
        auto const& statisticsHistory = _SimulationFacade::get()->getStatisticsHistory();
        {
            std::lock_guard lock(statisticsHistory.getMutex());
            globalSource = statisticsHistory.getDataRef();
        }
        if (_timelineMode == TimelineMode_LastSteps && !globalSource.empty()) {
            auto startTime = globalSource.back().time - toDouble(_lastSteps);
            std::erase_if(globalSource, [&](auto const& dataPoints) { return dataPoints.time < startTime; });
        }
    }

    //"all lineages" series
    _allLineages.id = -1;
    _allLineages.colorBitset = 0;
    _allLineages.timePoints.clear();
    for (int i = 0; i < NumMetrics; ++i) {
        _allLineages.series.at(i).clear();
    }
    for (size_t sampleIndex = 0; sampleIndex < globalSource.size(); ++sampleIndex) {
        auto const& dataPoints = globalSource.at(sampleIndex);
        auto const* lastDataPoints = sampleIndex > 0 ? &globalSource.at(sampleIndex - 1) : nullptr;
        _allLineages.timePoints.emplace_back(dataPoints.time);
        for (int i = 0; i < NumMetrics; ++i) {
            _allLineages.series.at(i).emplace_back(getGlobalMetricValue(dataPoints, lastDataPoints, useTimeAsClock, i));
        }
    }
    auto const& liveGlobalHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    if (!liveGlobalHistory.empty()) {
        auto lastDataPoints = liveGlobalHistory.back().toOverallDataPointCollection();
        std::optional<OverallDataPointCollection> previousDataPoints;
        if (liveGlobalHistory.size() >= 2) {
            previousDataPoints = liveGlobalHistory.at(liveGlobalHistory.size() - 2).toOverallDataPointCollection();
        }
        for (int i = 0; i < NumMetrics; ++i) {
            _allLineages.currentValues.at(i) = getGlobalMetricValue(lastDataPoints, previousDataPoints ? &*previousDataPoints : nullptr, true, i);
        }
    }
    //per-lineage series for the selected lineages (live statistics are the only source of lineage data)
    _plottedLineages.clear();
    if (!_selectedLineageIds.empty()) {
        auto startClock = std::numeric_limits<double>::lowest();
        if (!liveHistory.empty()) {
            if (_timelineMode == TimelineMode_RealTime) {
                startClock = liveHistory.back().time - toDouble(_timeHorizon);
            } else if (_timelineMode == TimelineMode_LastSteps) {
                startClock = liveHistory.back().timestep - toDouble(_lastSteps);
            }
        }
        for (auto const& selectedId : _selectedLineageIds) {
            LineageDisplayData lineage;
            lineage.id = selectedId;
            DataPointCollection const* lastSampleWithEntry = nullptr;
            LineageDataPoint const* lastEntry = nullptr;
            for (auto const& sample : liveHistory) {
                if ((useTimeAsClock ? sample.time : sample.timestep) < startClock) {
                    continue;
                }
                auto const* entry = findLineageEntry(sample, toUInt32(selectedId));
                if (!entry) {
                    continue;
                }
                lineage.colorBitset = toInt(entry->colorBitset);
                lineage.timePoints.emplace_back(useTimeAsClock ? sample.time : sample.timestep);
                auto clock = useTimeAsClock ? sample.time : sample.systemClock;
                auto lastClock = lastSampleWithEntry ? (useTimeAsClock ? lastSampleWithEntry->time : lastSampleWithEntry->systemClock) : 0.0;
                for (int i = 0; i < NumMetrics; ++i) {
                    lineage.series.at(i).emplace_back(getLineageMetricValue(*entry, lastEntry, clock, lastClock, i));
                }
                lastSampleWithEntry = &sample;
                lastEntry = entry;
            }
            for (auto const& tableLineage : _lineages) {
                if (tableLineage.id == selectedId) {
                    lineage.currentValues = tableLineage.currentValues;
                    if (lineage.colorBitset == 0) {
                        lineage.colorBitset = tableLineage.colorBitset;
                    }
                }
            }
            _plottedLineages.emplace_back(std::move(lineage));
        }
    }

    //header sparklines and deltas from the live statistics
    auto liveWindowInSeconds = liveGlobalHistory.size() >= 2 ? liveGlobalHistory.back().time - liveGlobalHistory.front().time : 0.0;
    for (int cardIndex = 0; cardIndex < 3; ++cardIndex) {
        auto& sparkline = _headerSparklines.at(cardIndex);
        sparkline.clear();
        std::vector<double> fullSeries;
        if (cardIndex == 0) {
            fullSeries = _externalEnergySeries;
        } else {
            fullSeries.reserve(liveGlobalHistory.size());
            for (auto const& dataPoints : liveGlobalHistory) {
                switch (cardIndex) {
                case 1:
                    fullSeries.emplace_back(dataPoints.overall.numLineages);
                    break;
                case 2:
                    fullSeries.emplace_back(dataPoints.overall.numCreatures);
                    break;
                }
            }
        }
        _headerDeltas.at(cardIndex) = calcDeltaPercentPerMinute(fullSeries, liveWindowInSeconds);
        auto stride = std::max(size_t(1), fullSeries.size() / NumSparklinePoints);
        for (size_t i = 0; i < fullSeries.size(); i += stride) {
            sparkline.emplace_back(fullSeries.at(i));
        }
    }
}

void EvolutionDashboardWindow::processHeader()
{
    auto const& style = ImGui::GetStyle();
    auto cardWidth = (ImGui::GetContentRegionAvail().x - 3 * style.ItemSpacing.x) / 5;
    auto cardHeight =
        style.WindowPadding.y * 2 + ImGui::GetTextLineHeight() * 3 + StyleRepository::get().getLargeFont()->FontSize + style.ItemSpacing.y * 3 + scale(6.0f);

    auto const& liveGlobalHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    auto const* lastDataPoints = !liveGlobalHistory.empty() ? &liveGlobalHistory.back() : nullptr;

    processEntitiesCard(cardWidth * 2, cardHeight);
    ImGui::SameLine();
    auto externalEnergy = !_externalEnergySeries.empty() ? _externalEnergySeries.back() : 0.0;
    processKpiCard("EXTERNAL ENERGY", formatMetricValue(externalEnergy, 0), toFloat(_headerDeltas.at(0)), 0, cardWidth, cardHeight);
    ImGui::SameLine();
    auto numLineages = lastDataPoints ? lastDataPoints->overall.numLineages : 0.0;
    processKpiCard("LINEAGES", formatMetricValue(numLineages, 0), toFloat(_headerDeltas.at(1)), 1, cardWidth, cardHeight);
    ImGui::SameLine();
    processKpiCard("CREATURES", formatMetricValue(_allLineages.currentValues.at(0), 0), toFloat(_headerDeltas.at(2)), 2, cardWidth, cardHeight);
}

void EvolutionDashboardWindow::processKpiCard(std::string const& label, std::string const& value, float delta, int sparklineIndex, float width, float height)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, scale(6.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)CardBackgroundColor);
    ImGui::PushStyleColor(ImGuiCol_Border, (ImU32)CardBorderColor);
    if (ImGui::BeginChild(("##card" + label).c_str(), {width, height}, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        AlienGui::Text(label);
        ImGui::PopStyleColor();

        ImGui::PushFont(StyleRepository::get().getLargeFont());
        AlienGui::Text(value);
        ImGui::PopFont();

        char deltaText[32];
        snprintf(deltaText, sizeof(deltaText), "%+.1f %% / min", delta);
        ImGui::PushStyleColor(ImGuiCol_Text, delta >= 0 ? (ImU32)PositiveDeltaColor : (ImU32)NegativeDeltaColor);
        AlienGui::Text(deltaText);
        ImGui::PopStyleColor();

        auto const& sparkline = _headerSparklines.at(sparklineIndex);
        if (sparkline.size() >= 2) {
            ImGui::SetCursorPos({width - scale(SparklineWidth + 12.0f), height - scale(SparklineHeight + 12.0f)});
            ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
            ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0, 0, 0, 0));
            ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0, 0, 0, 0));
            ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0, 0, 0, 0));
            if (ImPlot::BeginPlot(
                    ("##spark" + label).c_str(), {scale(SparklineWidth), scale(SparklineHeight)}, ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs)) {
                ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
                auto [minIt, maxIt] = std::minmax_element(sparkline.begin(), sparkline.end());
                ImPlot::SetupAxesLimits(0, toDouble(sparkline.size() - 1), *minIt * 0.9, *maxIt * 1.1 + NEAR_ZERO, ImGuiCond_Always);
                ImPlot::PushStyleColor(ImPlotCol_Line, delta >= 0 ? (ImU32)PositiveDeltaColor : (ImU32)NegativeDeltaColor);
                ImPlot::PlotLine("##", sparkline.data(), toInt(sparkline.size()));
                ImPlot::PopStyleColor();
                ImPlot::EndPlot();
            }
            ImPlot::PopStyleColor(3);
            ImPlot::PopStyleVar();
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
}

void EvolutionDashboardWindow::processEntitiesCard(float width, float height)
{
    auto const& liveGlobalHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    auto solids = 0.0;
    auto fluids = 0.0;
    auto cells = 0.0;
    auto total = 0.0;
    if (!liveGlobalHistory.empty()) {
        auto const& dataPoints = liveGlobalHistory.back();
        solids = dataPoints.overall.numSolidObjects;
        fluids = dataPoints.overall.numFluidObjects;
        cells = dataPoints.overall.numCellObjects;
        total = solids + fluids + cells;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, scale(6.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)CardBackgroundColor);
    ImGui::PushStyleColor(ImGuiCol_Border, (ImU32)CardBorderColor);
    if (ImGui::BeginChild("##cardEntities", {width, height}, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        AlienGui::Text("ENTITIES");
        ImGui::PopStyleColor();

        ImGui::PushFont(StyleRepository::get().getLargeFont());
        AlienGui::Text(formatMetricValue(total, 0));
        ImGui::PopFont();

        std::pair<char const*, double> const subValues[] = {
            {"Solids", solids},
            {"Fluid particles", fluids},
            {"Cells", cells},
        };
        for (auto const& [subLabel, subValue] : subValues) {
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
            AlienGui::Text(subLabel);
            ImGui::PopStyleColor();
            AlienGui::Text(formatMetricValue(subValue, 0));
            ImGui::EndGroup();
            ImGui::SameLine(0, scale(18.0f));
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
}

void EvolutionDashboardWindow::processFilterBar()
{
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
    AlienGui::Text("Filter by colors");
    ImGui::PopStyleColor();
    ImGui::SameLine(0, scale(12.0f));

    auto drawList = ImGui::GetWindowDrawList();
    auto chipSize = scale(ColorChipSize);
    for (int i = 0; i < MAX_COLORS; ++i) {
        ImGui::PushID(i);
        auto pos = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##chip", {chipSize, chipSize})) {
            _colorFilter ^= 1 << i;
        }
        auto active = (_colorFilter & (1 << i)) != 0;
        auto color = toImColor(_cellColors.at(i), active ? 1.0f : 0.2f);
        drawList->AddRectFilled(pos, {pos.x + chipSize, pos.y + chipSize}, color, scale(5.0f));
        if (active) {
            drawList->AddRect(
                {pos.x - scale(1.5f), pos.y - scale(1.5f)},
                {pos.x + chipSize + scale(1.5f), pos.y + chipSize + scale(1.5f)},
                ImColor(255, 255, 255, 255),
                scale(6.0f),
                0,
                scale(1.5f));
        }
        auto label = std::to_string(i + 1);
        auto labelSize = ImGui::CalcTextSize(label.c_str());
        drawList->AddText({pos.x + (chipSize - labelSize.x) / 2, pos.y + (chipSize - labelSize.y) / 2}, ImColor(0, 0, 0, active ? 220 : 120), label.c_str());
        ImGui::PopID();
        ImGui::SameLine(0, scale(5.0f));
    }
    ImGui::NewLine();
    ImGui::Spacing();
}

void EvolutionDashboardWindow::processLineageTable()
{
    auto flags = ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("##lineages", NumMetrics + 1, flags)) {
        ImGui::TableSetupScrollFreeze(1, 2);
        ImGui::TableSetupColumn("Lineage", ImGuiTableColumnFlags_WidthFixed, scale(150.0f));
        for (auto const& metric : Metrics) {
            ImGui::TableSetupColumn(metric.tableHeader, ImGuiTableColumnFlags_WidthFixed, scale(105.0f));
        }

        //header row with centered labels
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        for (int column = 0; column < NumMetrics + 1; ++column) {
            ImGui::TableSetColumnIndex(column);
            auto columnName = ImGui::TableGetColumnName(column);
            auto textWidth = ImGui::CalcTextSize(columnName).x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - textWidth) / 2));
            ImGui::TextUnformatted(columnName);
        }

        //summary row
        ImGui::TableNextRow();
        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImColor(0.13f, 0.16f, 0.23f, 1.0f));
        ImGui::TableSetColumnIndex(0);
        auto allSelected = _selectedLineageIds.empty();
        if (ImGui::Selectable("##rowAll", allSelected, ImGuiSelectableFlags_SpanAllColumns)) {
            _selectedLineageIds.clear();
        }
        ImGui::SameLine(0, 0);
        AlienGui::Text("All lineages (" + std::to_string(_lineages.size()) + ")");
        for (int i = 0; i < NumMetrics; ++i) {
            ImGui::TableSetColumnIndex(i + 1);
            rightAlignedText(formatMetricValue(_allLineages.currentValues.at(i), Metrics[i].decimals));
        }

        //lineage rows
        auto drawList = ImGui::GetWindowDrawList();
        auto maxNumColors = 1;
        for (auto const& lineage : _lineages) {
            if (_colorFilter & lineage.colorBitset) {
                maxNumColors = std::max(maxNumColors, std::popcount(static_cast<unsigned>(lineage.colorBitset)));
            }
        }
        auto swatchSlotWidth = toFloat(maxNumColors) * (scale(SwatchSize) + scale(SwatchGap));
        for (auto const& lineage : _lineages) {
            if ((_colorFilter & lineage.colorBitset) == 0) {
                continue;
            }
            ImGui::PushID(toInt(lineage.id));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            auto selected = _selectedLineageIds.contains(lineage.id);
            if (ImGui::Selectable("##row", selected, ImGuiSelectableFlags_SpanAllColumns)) {
                if (ImGui::GetIO().KeyCtrl) {
                    if (selected) {
                        _selectedLineageIds.erase(lineage.id);
                    } else {
                        _selectedLineageIds.insert(lineage.id);
                    }
                } else {
                    _selectedLineageIds.clear();
                    _selectedLineageIds.insert(lineage.id);
                }
            }
            ImGui::SameLine(0, 0);
            drawColorSwatches(drawList, ImGui::GetCursorScreenPos(), lineage.colorBitset, ImGui::GetTextLineHeight(), _cellColors);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + swatchSlotWidth + scale(4.0f));
            AlienGui::Text("Lineage #" + std::to_string(lineage.id));
            for (int i = 0; i < NumMetrics; ++i) {
                ImGui::TableSetColumnIndex(i + 1);
                rightAlignedText(formatMetricValue(lineage.currentValues.at(i), Metrics[i].decimals));
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void EvolutionDashboardWindow::processTimelineSection()
{
    processTimelineHeader();
    if (ImGui::BeginChild("##timelinePlots", {0, 0})) {
        processTimelinePlots();
    }
    ImGui::EndChild();
}

void EvolutionDashboardWindow::processTimelineHeader()
{
    ImGui::Spacing();
    std::vector<std::string> modeValues{"Real-time", "Last X steps"};
    if (_selectedLineageIds.empty()) {
        modeValues.emplace_back("Entire history");
    }
    AlienGui::Switcher(AlienGui::SwitcherParameters().name("Mode").width(260.0f).textWidth(45.0f).values(modeValues), &_timelineMode);
    if (_timelineMode == TimelineMode_RealTime) {
        ImGui::SameLine(0, scale(20.0f));
        if (ImGui::BeginChild("##timeHorizon", {scale(320.0f), ImGui::GetFrameHeight()})) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Time horizon").min(1.0f).max(TimelineLiveStatistics::MaxLiveHistory).format("%.1f s").textWidth(90.0f),
                &_timeHorizon);
            validateAndCorrect();
        }
        ImGui::EndChild();
    }
    if (_timelineMode == TimelineMode_LastSteps) {
        ImGui::SameLine(0, scale(20.0f));
        if (ImGui::BeginChild("##lastSteps", {scale(200.0f), ImGui::GetFrameHeight()})) {
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Steps").textWidth(45.0f), _lastSteps);
            validateAndCorrect();
        }
        ImGui::EndChild();
    }

    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
    AlienGui::Text("TIMELINE FILTER");
    ImGui::PopStyleColor();
    ImGui::SameLine(0, scale(12.0f));
    if (_selectedLineageIds.empty()) {
        AlienGui::Text("All lineages");
    } else {
        auto drawList = ImGui::GetWindowDrawList();
        for (auto const& lineage : _plottedLineages) {
            auto swatchesWidth = drawColorSwatches(drawList, ImGui::GetCursorScreenPos(), lineage.colorBitset, ImGui::GetTextLineHeight(), _cellColors);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + swatchesWidth + scale(4.0f));
            AlienGui::Text("Lineage #" + std::to_string(lineage.id));
            ImGui::SameLine(0, scale(14.0f));
        }
        ImGui::NewLine();
    }
    ImGui::Spacing();
}

void EvolutionDashboardWindow::processTimelinePlots()
{
    std::vector<LineageDisplayData const*> plottedLineages;
    if (_selectedLineageIds.empty()) {
        plottedLineages.emplace_back(&_allLineages);
    } else {
        for (auto const& lineage : _plottedLineages) {
            plottedLineages.emplace_back(&lineage);
        }
    }

    if (ImGui::BeginTable("##plots", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, scale(PlotLabelColumnWidth));
        ImGui::TableSetupColumn("##plot");
        for (int i = 0; i < NumMetrics; ++i) {
            auto showTimeAxis = i == NumMetrics - 1;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            AlienGui::Text(Metrics[i].plotName);
            auto compactLayout = plottedLineages.size() > 1;
            for (auto const* lineage : plottedLineages) {
                ImGui::PushFont(StyleRepository::get().getMediumBoldFont());
                if (lineage->id != -1) {
                    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)toImColor(_cellColors.at(getFirstColor(lineage->colorBitset))));
                }
                AlienGui::Text(formatMetricValue(lineage->currentValues.at(i), Metrics[i].decimals));
                if (lineage->id != -1) {
                    ImGui::PopStyleColor();
                }
                ImGui::PopFont();
                if (compactLayout) {
                    ImGui::SameLine();
                }
                auto delta = calcWindowDeltaPercent(lineage->series.at(i));
                char deltaText[32];
                snprintf(deltaText, sizeof(deltaText), "%+.1f %%", delta);
                ImGui::PushStyleColor(ImGuiCol_Text, delta >= 0 ? (ImU32)PositiveDeltaColor : (ImU32)NegativeDeltaColor);
                AlienGui::Text(deltaText);
                ImGui::PopStyleColor();
            }

            ImGui::TableSetColumnIndex(1);
            processTimelinePlot(plottedLineages, i, showTimeAxis);
        }
        ImGui::EndTable();
    }
}

void EvolutionDashboardWindow::processTimelinePlot(std::vector<LineageDisplayData const*> const& plottedLineages, int metricIndex, bool showTimeAxis)
{
    auto upperBound = 0.0;
    auto minTime = std::numeric_limits<double>::max();
    auto maxTime = std::numeric_limits<double>::lowest();
    auto hasData = false;
    for (auto const* lineage : plottedLineages) {
        if (lineage->timePoints.size() < 2) {
            continue;
        }
        hasData = true;
        minTime = std::min(minTime, lineage->timePoints.front());
        maxTime = std::max(maxTime, lineage->timePoints.back());
        for (auto const& value : lineage->series.at(metricIndex)) {
            upperBound = std::max(upperBound, value);
        }
    }
    if (!hasData) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        AlienGui::Text("No data");
        ImGui::PopStyleColor();
        return;
    }
    if (_timelineMode == TimelineMode_RealTime) {
        minTime = maxTime - toDouble(_timeHorizon);
    } else if (_timelineMode == TimelineMode_LastSteps) {
        minTime = maxTime - toDouble(_lastSteps);
    }
    upperBound *= 1.35;

    ImGui::PushID(metricIndex);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextAxesLimits(minTime, maxTime, 0, upperBound + NEAR_ZERO, ImGuiCond_Always);

    auto height = showTimeAxis ? PlotHeightWithAxis : PlotHeight;
    if (ImPlot::BeginPlot(
            "##plot", ImVec2(-1, scale(height)), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxis(ImAxis_X1, "", showTimeAxis ? ImPlotAxisFlags_None : ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxisFormat(ImAxis_Y1, "");
        for (auto const* lineage : plottedLineages) {
            auto count = toInt(lineage->timePoints.size());
            if (count < 2) {
                continue;
            }
            ImGui::PushID(toInt(lineage->id) + 1);
            auto color = lineage->id == -1 ? SumSeriesColor : toImColor(_cellColors.at(getFirstColor(lineage->colorBitset)));
            auto const& series = lineage->series.at(metricIndex);
            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            ImPlot::PlotLine("##line", lineage->timePoints.data(), series.data(), count);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##shaded", lineage->timePoints.data(), series.data(), count);
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
            if (ImGui::GetStyle().Alpha == 1.0f) {
                ImPlot::Annotation(
                    lineage->timePoints.back(),
                    series.back(),
                    color,
                    ImVec2(-10.0f, 10.0f),
                    true,
                    "%s",
                    formatMetricValue(series.back(), Metrics[metricIndex].decimals).c_str());
            }
            ImGui::PopID();
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void EvolutionDashboardWindow::validateAndCorrect()
{
    _timelinesHeight = std::max(scale(100.0f), _timelinesHeight);
    _timelineMode = std::clamp(_timelineMode, static_cast<TimelineMode>(TimelineMode_RealTime), static_cast<TimelineMode>(TimelineMode_EntireHistory));
    _lastSteps = std::clamp(_lastSteps, 1000, 1000000000);
    _timeHorizon = std::clamp(_timeHorizon, 1.0f, TimelineLiveStatistics::MaxLiveHistory);
    _colorFilter &= (1 << MAX_COLORS) - 1;
    if (_colorFilter == 0) {
        _colorFilter = (1 << MAX_COLORS) - 1;
    }
}
