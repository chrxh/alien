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
    ImColor const OverflowWarningColor = ImColor(0.94f, 0.62f, 0.22f, 1.0f);

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

    double getGlobalMetricValue(DataPointCollection const& dataPoints, DataPointCollection const* lastDataPoints, bool clockFromTime, int metricIndex)
    {
        switch (metricIndex) {
        case 0:
            return dataPoints.numCreatures;
        case 1:
            return dataPoints.averageCreatureCells;
        case 2:
            return dataPoints.averageGenomeNodes;
        case 3:
            return dataPoints.creatureEnergy;
        case 4:
            return dataPoints.averageMutationRate;
        case 5:
            return dataPoints.averageGeneration;
        case 6:
        case 7: {
            if (!lastDataPoints) {
                return 0.0;
            }
            auto clock = clockFromTime ? dataPoints.time : dataPoints.systemClock;
            auto lastClock = clockFromTime ? lastDataPoints->time : lastDataPoints->systemClock;
            if (metricIndex == 6) {
                return calcRate(dataPoints.accumCreatedCreatures, lastDataPoints->accumCreatedCreatures, clock, lastClock);
            } else {
                return calcRate(dataPoints.accumMutations, lastDataPoints->accumMutations, clock, lastClock);
            }
        }
        default:
            return 0.0;
        }
    }

    double getLineageMetricValue(LineageStatisticsEntry const& entry, LineageStatisticsEntry const* lastEntry, double clock, double lastClock, int metricIndex)
    {
        switch (metricIndex) {
        case 0:
            return toDouble(entry.numCreatures);
        case 1:
            return entry.numCreatures > 0 ? toDouble(entry.sumCreatureCells) / entry.numCreatures : 0.0;
        case 2:
            return entry.numGenomes > 0 ? toDouble(entry.sumGenomeNodes) / entry.numGenomes : 0.0;
        case 3:
            return toDouble(entry.sumCreatureEnergy);
        case 4:
            return entry.numGenomes > 0 ? toDouble(entry.sumMutationRates) / entry.numGenomes : 0.0;
        case 5:
            return entry.numCreatures > 0 ? toDouble(entry.sumCreatureGenerations) / entry.numCreatures : 0.0;
        case 6:
            return lastEntry ? calcRate(toDouble(entry.numCreatedCreatures), toDouble(lastEntry->numCreatedCreatures), clock, lastClock) : 0.0;
        case 7:
            return lastEntry ? calcRate(toDouble(entry.totalMutations), toDouble(lastEntry->totalMutations), clock, lastClock) : 0.0;
        default:
            return 0.0;
        }
    }

    LineageStatisticsEntry const* findLineageEntry(LineageSample const& sample, uint32_t lineageId)
    {
        auto it =
            std::lower_bound(sample.entries.begin(), sample.entries.end(), lineageId, [](auto const& entry, uint32_t id) { return entry.lineageId < id; });
        if (it != sample.entries.end() && it->lineageId == lineageId) {
            return &*it;
        }
        return nullptr;
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
    _plotScale = GlobalSettings::get().getValue("windows.evolution dashboard.plot scale", _plotScale);
    _lastSteps = GlobalSettings::get().getValue("windows.evolution dashboard.last steps", _lastSteps);
    _colorFilter = GlobalSettings::get().getValue("windows.evolution dashboard.color filter", _colorFilter);
    validateAndCorrect();
}

void EvolutionDashboardWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.evolution dashboard.timelines height", _timelinesHeight);
    GlobalSettings::get().setValue("windows.evolution dashboard.timeline mode", _timelineMode);
    GlobalSettings::get().setValue("windows.evolution dashboard.plot scale", _plotScale);
    GlobalSettings::get().setValue("windows.evolution dashboard.last steps", _lastSteps);
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

    auto rawStatistics = _SimulationFacade::get()->getStatisticsRawData();
    _timelineLiveStatistics.update(rawStatistics.timeline, _SimulationFacade::get()->getCurrentTimestep());
    _lineageMapOverflow = rawStatistics.timeline.timestep.evolution.lineageMapOverflow != 0;

    auto rawLineageStatistics = _SimulationFacade::get()->getRawLineageStatistics();
    LineageSample sample;
    sample.time = _timeSinceSimStart;
    sample.systemClock = _timeSinceSimStart;
    sample.entries = std::move(rawLineageStatistics.entries);
    std::sort(sample.entries.begin(), sample.entries.end(), [](auto const& lhs, auto const& rhs) { return lhs.lineageId < rhs.lineageId; });
    _liveLineageHistory.emplace_back(std::move(sample));
    while (!_liveLineageHistory.empty() && _liveLineageHistory.back().time - _liveLineageHistory.front().time > MaxLiveHistory + 1.0) {
        _liveLineageHistory.erase(_liveLineageHistory.begin());
    }

    _externalEnergySeries.emplace_back(toDouble(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
    auto maxExternalEnergyValues = toInt(std::lround(MaxLiveHistory * 1000 / LiveStatisticsDeltaTime));
    while (toInt(_externalEnergySeries.size()) > maxExternalEnergyValues) {
        _externalEnergySeries.erase(_externalEnergySeries.begin());
    }
}

void EvolutionDashboardWindow::processIntern()
{
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
    auto liveBackTime = !_liveLineageHistory.empty() ? _liveLineageHistory.back().time : -1.0;
    auto historyBackTime = -1.0;
    if (_timelineMode != TimelineMode_RealTime) {
        auto const& statisticsHistory = _SimulationFacade::get()->getStatisticsHistory();
        std::lock_guard lock(statisticsHistory.getMutex());
        if (!statisticsHistory.getDataRef().empty()) {
            historyBackTime = statisticsHistory.getDataRef().back().time;
        }
    }
    RebuildKey key{_timelineMode, _lastSteps, liveBackTime, historyBackTime, _selectedLineageIds};
    if (_lastRebuildKey && *_lastRebuildKey == key) {
        return;
    }
    _lastRebuildKey = std::move(key);

    //table rows from the latest live sample
    _lineages.clear();
    if (!_liveLineageHistory.empty()) {
        auto const& lastSample = _liveLineageHistory.back();
        auto const* previousSample = _liveLineageHistory.size() >= 2 ? &_liveLineageHistory.at(_liveLineageHistory.size() - 2) : nullptr;
        for (auto const& entry : lastSample.entries) {
            LineageDisplayData lineage;
            lineage.id = toInt(entry.lineageId);
            lineage.colorBitset = toInt(entry.colorBitset);
            auto const* previousEntry = previousSample ? findLineageEntry(*previousSample, entry.lineageId) : nullptr;
            for (int i = 0; i < NumMetrics; ++i) {
                lineage.currentValues.at(i) = getLineageMetricValue(entry, previousEntry, lastSample.time, previousSample ? previousSample->time : 0.0, i);
            }
            _lineages.emplace_back(std::move(lineage));
        }
        std::sort(_lineages.begin(), _lineages.end(), [](auto const& lhs, auto const& rhs) { return lhs.currentValues.at(0) > rhs.currentValues.at(0); });
    }

    //data source for the timeline plots
    auto useTimeAsClock = _timelineMode == TimelineMode_RealTime;
    std::vector<DataPointCollection> globalSource;
    LineageHistoryData lineageSource;
    if (_timelineMode == TimelineMode_RealTime) {
        globalSource = _timelineLiveStatistics.getDataPointCollectionHistory();
        lineageSource = _liveLineageHistory;
    } else {
        auto const& statisticsHistory = _SimulationFacade::get()->getStatisticsHistory();
        {
            std::lock_guard lock(statisticsHistory.getMutex());
            globalSource = statisticsHistory.getDataRef();
        }
        lineageSource = _SimulationFacade::get()->getLineageHistory().getCopiedData();
        if (_timelineMode == TimelineMode_LastSteps && !globalSource.empty()) {
            auto startTime = globalSource.back().time - toDouble(_lastSteps);
            std::erase_if(globalSource, [&](auto const& dataPoints) { return dataPoints.time < startTime; });
            std::erase_if(lineageSource, [&](auto const& sample) { return sample.time < startTime; });
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
        auto const& lastDataPoints = liveGlobalHistory.back();
        auto const* previousDataPoints = liveGlobalHistory.size() >= 2 ? &liveGlobalHistory.at(liveGlobalHistory.size() - 2) : nullptr;
        for (int i = 0; i < NumMetrics; ++i) {
            _allLineages.currentValues.at(i) = getGlobalMetricValue(lastDataPoints, previousDataPoints, true, i);
        }
    }
    for (int i = 0; i < NumMetrics; ++i) {
        _metricWindowDeltas.at(i) = calcWindowDeltaPercent(_allLineages.series.at(i));
    }

    //per-lineage series for the selected lineages
    _plottedLineages.clear();
    if (!_selectedLineageIds.empty()) {
        for (auto const& selectedId : _selectedLineageIds) {
            LineageDisplayData lineage;
            lineage.id = selectedId;
            LineageSample const* lastSampleWithEntry = nullptr;
            LineageStatisticsEntry const* lastEntry = nullptr;
            for (auto const& sample : lineageSource) {
                auto const* entry = findLineageEntry(sample, toUInt32(selectedId));
                if (!entry) {
                    continue;
                }
                lineage.colorBitset = toInt(entry->colorBitset);
                lineage.timePoints.emplace_back(sample.time);
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
    for (int cardIndex = 0; cardIndex < 4; ++cardIndex) {
        auto& sparkline = _headerSparklines.at(cardIndex);
        sparkline.clear();
        std::vector<double> fullSeries;
        if (cardIndex == 1) {
            fullSeries = _externalEnergySeries;
        } else {
            fullSeries.reserve(liveGlobalHistory.size());
            for (auto const& dataPoints : liveGlobalHistory) {
                switch (cardIndex) {
                case 0:
                    fullSeries.emplace_back(dataPoints.totalEnergy.summedValues);
                    break;
                case 2:
                    fullSeries.emplace_back(dataPoints.numLineages);
                    break;
                case 3:
                    fullSeries.emplace_back(dataPoints.numCreatures);
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
    auto cardWidth = (ImGui::GetContentRegionAvail().x - 4 * style.ItemSpacing.x) / 6;
    auto cardHeight =
        style.WindowPadding.y * 2 + ImGui::GetTextLineHeight() * 3 + StyleRepository::get().getLargeFont()->FontSize + style.ItemSpacing.y * 3 + scale(6.0f);

    auto const& liveGlobalHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    auto const* lastDataPoints = !liveGlobalHistory.empty() ? &liveGlobalHistory.back() : nullptr;

    processEntitiesCard(cardWidth * 2, cardHeight);
    ImGui::SameLine();
    auto sumEnergy = lastDataPoints ? lastDataPoints->totalEnergy.summedValues : 0.0;
    processKpiCard("SUM ENERGY", formatMetricValue(sumEnergy, 0), toFloat(_headerDeltas.at(0)), 0, cardWidth, cardHeight);
    ImGui::SameLine();
    auto externalEnergy = !_externalEnergySeries.empty() ? _externalEnergySeries.back() : 0.0;
    processKpiCard("EXTERNAL ENERGY", formatMetricValue(externalEnergy, 0), toFloat(_headerDeltas.at(1)), 1, cardWidth, cardHeight);
    ImGui::SameLine();
    auto numLineages = lastDataPoints ? lastDataPoints->numLineages : 0.0;
    processKpiCard("LINEAGES", formatMetricValue(numLineages, 0), toFloat(_headerDeltas.at(2)), 2, cardWidth, cardHeight);
    ImGui::SameLine();
    processKpiCard("CREATURES", formatMetricValue(_allLineages.currentValues.at(0), 0), toFloat(_headerDeltas.at(3)), 3, cardWidth, cardHeight);
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
    auto energyParticles = 0.0;
    auto total = 0.0;
    if (!liveGlobalHistory.empty()) {
        auto const& dataPoints = liveGlobalHistory.back();
        solids = dataPoints.numSolidObjects;
        fluids = dataPoints.numFluidObjects;
        cells = dataPoints.numCellObjects;
        energyParticles = dataPoints.numEnergyParticles.summedValues;
        total = dataPoints.numObjects.summedValues + energyParticles;
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
            {"Energy particles", energyParticles},
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
    if (_lineageMapOverflow) {
        ImGui::SameLine(0, scale(20.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)OverflowWarningColor);
        AlienGui::Text("Too many lineages: some are not tracked");
        ImGui::PopStyleColor();
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
    AlienGui::Switcher(
        AlienGui::SwitcherParameters().name("Mode").width(260.0f).textWidth(45.0f).values({"Real-time", "Entire history", "Last X steps"}), &_timelineMode);
    ImGui::SameLine(0, scale(20.0f));
    AlienGui::Switcher(AlienGui::SwitcherParameters().name("Scale").width(220.0f).textWidth(45.0f).values({"Linear", "Logarithmic"}), &_plotScale);
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
    if (ImGui::BeginTable("##plots", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, scale(PlotLabelColumnWidth));
        ImGui::TableSetupColumn("##plot");
        for (int i = 0; i < NumMetrics; ++i) {
            auto showTimeAxis = i == NumMetrics - 1;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            AlienGui::Text(Metrics[i].plotName);
            ImGui::PushFont(StyleRepository::get().getMediumBoldFont());
            AlienGui::Text(formatMetricValue(_allLineages.currentValues.at(i), Metrics[i].decimals));
            ImGui::PopFont();
            char deltaText[32];
            snprintf(deltaText, sizeof(deltaText), "%+.1f %%", _metricWindowDeltas.at(i));
            ImGui::PushStyleColor(ImGuiCol_Text, _metricWindowDeltas.at(i) >= 0 ? (ImU32)PositiveDeltaColor : (ImU32)NegativeDeltaColor);
            AlienGui::Text(deltaText);
            ImGui::PopStyleColor();

            ImGui::TableSetColumnIndex(1);
            processTimelinePlot(i, showTimeAxis);
        }
        ImGui::EndTable();
    }
}

void EvolutionDashboardWindow::processTimelinePlot(int metricIndex, bool showTimeAxis)
{
    std::vector<LineageDisplayData const*> plottedLineages;
    if (_selectedLineageIds.empty()) {
        plottedLineages.emplace_back(&_allLineages);
    } else {
        for (auto const& lineage : _plottedLineages) {
            plottedLineages.emplace_back(&lineage);
        }
    }

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
        if (_plotScale == PlotScale_Logarithmic) {
            ImPlot::SetupAxisScale(
                ImAxis_Y1,
                [](double value, void* userData) { return log(value * 1000 + 1.0) / log(2.0); },
                [](double value, void* userData) { return (pow(2.0, value) - 1.0) / 1000; });
        }
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
            if (_plotScale == PlotScale_Linear) {
                ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.15f * ImGui::GetStyle().Alpha);
                ImPlot::PlotShaded("##shaded", lineage->timePoints.data(), series.data(), count);
                ImPlot::PopStyleVar();
            }
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
    _timelineMode = std::clamp(_timelineMode, static_cast<TimelineMode>(TimelineMode_RealTime), static_cast<TimelineMode>(TimelineMode_LastSteps));
    _plotScale = std::clamp(_plotScale, static_cast<PlotScale>(PlotScale_Linear), static_cast<PlotScale>(PlotScale_Logarithmic));
    _lastSteps = std::clamp(_lastSteps, 1000, 1000000000);
    _colorFilter &= (1 << MAX_COLORS) - 1;
    if (_colorFilter == 0) {
        _colorFilter = (1 << MAX_COLORS) - 1;
    }
}
