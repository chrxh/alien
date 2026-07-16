#include "EvolutionDashboardWindow.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <ctime>
#include <limits>
#include <cstdio>

#include <imgui.h>
#include <implot.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Definitions.h>
#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParametersTypes.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr LiveStatisticsDeltaTime = 50;  //in millisec
    auto constexpr RateAveragingInterval = 30.0;  //in seconds
    auto constexpr MinRateTimesteps = 1000.0;     //at the start of the history, rates are already shown once this many time steps are covered

    auto constexpr ColorChipSize = 22.0f;
    auto constexpr SwatchSize = 11.0f;
    auto constexpr SwatchGap = 3.0f;
    auto constexpr PlotLabelColumnWidth = 150.0f;
    auto constexpr MinPlotHeight = 40.0f;
    auto constexpr MaxPlotHeight = 300.0f;
    auto constexpr TimeAxisExtraHeight = 20.0f;
    auto constexpr DefaultTimelinesHeight = 380.0f;

    ImColor const CardBackgroundColor = ImColor(0.095f, 0.117f, 0.165f, 1.0f);
    ImColor const CardBorderColor = ImColor(0.165f, 0.196f, 0.270f, 1.0f);

    struct MetricDef
    {
        char const* tableHeader;
        char const* plotName;
        int tableDecimals;
        int plotDecimals;
    };

    //the last two metrics are rates per 1K time steps, both in the table and in the timeline plots
    MetricDef const Metrics[EvolutionDashboardWindow::NumMetrics] = {
        {"Creatures", "Creatures", 0, 0},
        {"Avg cells", "Avg cells", 1, 1},
        {"Avg nodes", "Avg nodes", 1, 1},
        {"Internal energy", "Internal energy", 0, 0},
        {"Avg mut. rate", "Avg mutation rate", 4, 4},
        {"Avg generation", "Avg generation", 0, 0},
        {"Created /1K", "Created /1K", 2, 2},
        {"Mutations /1K", "Mutations /1K", 4, 4},
    };

    ImColor toImColor(uint32_t rgb, float alpha = 1.0f, float brightness = 1.0f)
    {
        return ImColor(
            toInt(toFloat((rgb >> 16) & 0xff) * brightness),
            toInt(toFloat((rgb >> 8) & 0xff) * brightness),
            toInt(toFloat(rgb & 0xff) * brightness),
            toInt(alpha * 255.0f));
    }

    //palette as used by the former statistics window; one colormap color per plot row,
    //or one per selected lineage if multiple lineages are plotted together;
    //stepping through the colormap with an alternating stride of 1 and 2 makes adjacent rows distinct
    ImColor getPlotColor(int metricIndex, int seriesIndex, int numSeries)
    {
        auto index = numSeries > 1 ? seriesIndex : metricIndex;
        return ImColor(ImPlot::GetColormapColor((index / 2 * 3 + index % 2) % 11, ImPlotColormap_Cool));
    }

    std::string formatMetricValue(double value, int decimals)
    {
        if (std::isnan(value)) {
            return "-";
        }
        if (value >= toDouble(Infinity<float>::value)) {  //parameters use FLT_MAX to represent infinity
            return "infinity";
        }
        return StringHelper::format(value, decimals);
    }

    std::string convertSystemClockToString(double systemClock)
    {
        auto time_t = static_cast<std::time_t>(systemClock);
        std::tm* tm = std::localtime(&time_t);

        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm);
        return std::string(buffer);
    }

    void rightAlignedText(std::string const& text)
    {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(text.c_str()).x);
        ImGui::TextUnformatted(text.c_str());
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

    double calcRate(double accumValue, double lastAccumValue, double delta)
    {
        if (delta <= 0) {
            return 0.0;
        }
        return std::max(0.0, (accumValue - lastAccumValue) / delta);  //negative deltas occur after accumulated statistics have been reset
    }

    OverallDataPoint const& overallDataOf(DataPointCollection const& sample)
    {
        return sample.overall;
    }

    OverallDataPoint const& overallDataOf(OverallSample const& sample)
    {
        return sample.data;
    }

    template <typename Sample>
    double getOverallMetricValue(Sample const& sample, Sample const* referenceSample, int metricIndex)
    {
        auto const& data = overallDataOf(sample);
        switch (metricIndex) {
        case 0:
            return data.numCreatures;
        case 1:
            return data.averageCreatureCells;
        case 2:
            return data.averageGenomeNodes;
        case 3:
            return data.creatureEnergy;
        case 4:
            return data.averageMutationRate;
        case 5:
            return data.averageGeneration;
        case 6:
        case 7: {
            if (!referenceSample) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            auto const& referenceData = overallDataOf(*referenceSample);
            auto deltaKSteps = (sample.timestep - referenceSample->timestep) / 1000.0;
            if (metricIndex == 6) {
                return calcRate(data.accumCreatedCreatures, referenceData.accumCreatedCreatures, deltaKSteps);
            } else {
                return calcRate(data.accumMutations, referenceData.accumMutations, deltaKSteps);
            }
        }
        default:
            return 0.0;
        }
    }

    double getLineageMetricValue(LineageDataPoint const& entry, LineageDataPoint const* lastEntry, double rateDelta, int metricIndex)
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
            return lastEntry ? calcRate(entry.numCreatedCreatures, lastEntry->numCreatedCreatures, rateDelta) : std::numeric_limits<double>::quiet_NaN();
        case 7:
            return lastEntry ? calcRate(entry.totalMutations, lastEntry->totalMutations, rateDelta) : std::numeric_limits<double>::quiet_NaN();
        default:
            return 0.0;
        }
    }

    double getLineageSampleMetricValue(LineageSample const& sample, LineageSample const* referenceSample, int metricIndex)
    {
        auto deltaKSteps = referenceSample ? (sample.timestep - referenceSample->timestep) / 1000.0 : 0.0;
        return getLineageMetricValue(sample.data, referenceSample ? &referenceSample->data : nullptr, deltaKSteps, metricIndex);
    }

    LineageDataPoint const* findLineageEntry(DataPointCollection const& sample, uint32_t lineageId)
    {
        auto it = sample.lineages.find(lineageId);
        return it != sample.lineages.end() ? &it->second : nullptr;
    }

    int formatTimestepsInThousands(double value, char* buff, int size, void*)
    {
        return snprintf(buff, size, "%s", StringHelper::formatInThousands(value).c_str());
    }

    //most recent sample that is at least RateAveragingInterval older; falls back to the oldest sample
    size_t findRateReferenceIndex(std::vector<DataPointCollection> const& history, double time)
    {
        size_t result = 0;
        for (size_t i = 0; i < history.size(); ++i) {
            if (history.at(i).time + RateAveragingInterval > time) {
                break;
            }
            result = i;
        }
        return result;
    }

    //builds the x points and metric series of one timeline; reference samples for the rate metrics are selected
    //via a trailing ~30s window: the live buffer carries the time since simulation start while the long-term
    //history only provides the system clock
    template <typename Target, typename Sample, typename MetricEvaluator>
    void buildPlotSeries(
        Target& target,
        std::vector<Sample> const& source,
        size_t firstIndex,
        bool useTimeAsX,
        bool useSystemClockForRateWindow,
        MetricEvaluator const& evaluateMetric)
    {
        auto getX = [&](Sample const& sample) { return useTimeAsX ? sample.time : sample.timestep; };
        auto getRateWindowClock = [&](Sample const& sample) { return useSystemClockForRateWindow ? sample.systemClock : sample.time; };

        target.series = {};
        target.timePoints.clear();
        target.systemClockPoints.clear();

        size_t referenceIndex = 0;  //reference samples may also lie before the visible range
        for (auto sampleIndex = firstIndex; sampleIndex < source.size(); ++sampleIndex) {
            auto const& sample = source.at(sampleIndex);
            auto rateWindowClock = getRateWindowClock(sample);
            while (referenceIndex + 1 < sampleIndex && getRateWindowClock(source.at(referenceIndex + 1)) + RateAveragingInterval <= rateWindowClock) {
                ++referenceIndex;
            }
            //rates over references younger than the averaging interval would produce spikes at the left plot border; NaN suppresses those samples;
            //at the start of the timeline the oldest sample serves as reference once it is at least MinRateTimesteps old (like in the table)
            auto const& referenceSample = source.at(referenceIndex);
            auto referenceIsOldEnough = referenceIndex < sampleIndex
                && (getRateWindowClock(referenceSample) + RateAveragingInterval <= rateWindowClock
                    || (referenceIndex == 0 && referenceSample.timestep + MinRateTimesteps <= sample.timestep));
            auto const* reference = referenceIsOldEnough ? &referenceSample : nullptr;
            target.timePoints.emplace_back(getX(sample));
            if (!useTimeAsX) {
                target.systemClockPoints.emplace_back(sample.systemClock);
            }
            for (int i = 0; i < EvolutionDashboardWindow::NumMetrics; ++i) {
                target.series.at(i).emplace_back(evaluateMetric(sample, reference, i));
            }
        }
    }
}

EvolutionDashboardWindow::EvolutionDashboardWindow()
    : AlienWindow("Evolution dashboard", "windows.evolution dashboard", false, true, {scale(700.0f), scale(500.0f)})
{}

void EvolutionDashboardWindow::initIntern()
{
    _timelinesHeight = GlobalSettings::get().getValue("windows.evolution dashboard.timelines height", scale(DefaultTimelinesHeight));
    _timelineMode = GlobalSettings::get().getValue("windows.evolution dashboard.timeline mode", _timelineMode);
    _lastSteps = GlobalSettings::get().getValue("windows.evolution dashboard.last steps", _lastSteps);
    _timeHorizon = GlobalSettings::get().getValue("windows.evolution dashboard.time horizon", _timeHorizon);
    _plotHeight = GlobalSettings::get().getValue("windows.evolution dashboard.plot height", _plotHeight);
    _colorFilter = GlobalSettings::get().getValue("windows.evolution dashboard.color filter", _colorFilter);
    _sortColumnIndex = GlobalSettings::get().getValue("windows.evolution dashboard.sort column", _sortColumnIndex);
    _sortAscending = GlobalSettings::get().getValue("windows.evolution dashboard.sort ascending", _sortAscending);
    validateAndCorrect();
}

void EvolutionDashboardWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.evolution dashboard.timelines height", _timelinesHeight);
    GlobalSettings::get().setValue("windows.evolution dashboard.timeline mode", _timelineMode);
    GlobalSettings::get().setValue("windows.evolution dashboard.last steps", _lastSteps);
    GlobalSettings::get().setValue("windows.evolution dashboard.time horizon", _timeHorizon);
    GlobalSettings::get().setValue("windows.evolution dashboard.plot height", _plotHeight);
    GlobalSettings::get().setValue("windows.evolution dashboard.color filter", _colorFilter);
    GlobalSettings::get().setValue("windows.evolution dashboard.sort column", _sortColumnIndex);
    GlobalSettings::get().setValue("windows.evolution dashboard.sort ascending", _sortAscending);
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

    auto sessionId = _SimulationFacade::get()->getSessionId();
    if (_lastSessionId.has_value() && *_lastSessionId != sessionId) {
        _timelineLiveStatistics.clear();
    }
    _lastSessionId = sessionId;

    auto overallStatistics = _SimulationFacade::get()->getStatisticsEntry();
    _timelineLiveStatistics.update(overallStatistics, _SimulationFacade::get()->getCurrentTimestep());
}

void EvolutionDashboardWindow::processIntern()
{
    //scale the plot section proportionally when the window is resized
    auto windowHeight = ImGui::GetWindowSize().y;
    if (_lastWindowHeight.has_value() && *_lastWindowHeight > 0 && *_lastWindowHeight != windowHeight) {
        _timelinesHeight *= windowHeight / *_lastWindowHeight;
        validateAndCorrect();
    }
    _lastWindowHeight = windowHeight;

    updateCellColors();
    updateDisplayData();
    processHeader();
    processFilterBar();

    if (ImGui::BeginChild("##lineageTable", {0, -_timelinesHeight})) {
        processLineageTable();
    }
    ImGui::EndChild();

    AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _timelinesHeight);

    //apply selection changes from the table immediately; otherwise the plots are empty for one frame
    updateDisplayData();

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
    updateTableData();
    updatePlotData();
}

void EvolutionDashboardWindow::updateTableData()
{
    auto const& liveHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    auto liveBackTime = !liveHistory.empty() ? liveHistory.back().time : -1.0;
    if (_lastTableBackTime.has_value() && *_lastTableBackTime == liveBackTime) {
        return;
    }
    _lastTableBackTime = liveBackTime;

    //table rows from the latest live sample (rates per 1K time steps)
    _lineages.clear();
    if (liveHistory.empty()) {
        return;
    }
    auto const& lastSample = liveHistory.back();
    auto referenceIndex = findRateReferenceIndex(liveHistory, lastSample.time);
    for (auto const& [lineageId, entry] : lastSample.lineages) {
        LineageDisplayData lineage;
        lineage.id = toInt(lineageId);
        lineage.colorBitset = toInt(entry.colorBitset);
        LineageDataPoint const* referenceEntry = nullptr;
        auto referenceTimestep = 0.0;
        for (auto sampleIndex = referenceIndex; sampleIndex + 1 < liveHistory.size(); ++sampleIndex) {  //young lineages: use their oldest sample
            if (auto const* candidate = findLineageEntry(liveHistory.at(sampleIndex), lineageId)) {
                referenceEntry = candidate;
                referenceTimestep = liveHistory.at(sampleIndex).timestep;
                break;
            }
        }
        for (int i = 0; i < NumMetrics; ++i) {
            lineage.currentValues.at(i) = getLineageMetricValue(entry, referenceEntry, (lastSample.timestep - referenceTimestep) / 1000.0, i);
        }
        _lineages.emplace_back(std::move(lineage));
    }
    sortLineages();

    //current values for the table summary row
    DataPointCollection const* referenceDataPoints = liveHistory.size() >= 2 ? &liveHistory.at(referenceIndex) : nullptr;
    for (int i = 0; i < NumMetrics; ++i) {
        _allLineages.currentValues.at(i) = getOverallMetricValue(lastSample, referenceDataPoints, i);
    }
}

void EvolutionDashboardWindow::sortLineages()
{
    std::sort(_lineages.begin(), _lineages.end(), [column = _sortColumnIndex, ascending = _sortAscending](auto const& lhs, auto const& rhs) {
        if (column == 0) {
            return ascending ? lhs.id < rhs.id : lhs.id > rhs.id;
        }
        auto lhsValue = lhs.currentValues.at(column - 1);
        auto rhsValue = rhs.currentValues.at(column - 1);
        auto lhsIsNan = std::isnan(lhsValue);
        auto rhsIsNan = std::isnan(rhsValue);
        if (lhsIsNan || rhsIsNan) {
            if (lhsIsNan != rhsIsNan) {
                return rhsIsNan;  //rows without a value are sorted to the bottom
            }
            return lhs.id < rhs.id;
        }
        if (lhsValue != rhsValue) {
            return ascending ? lhsValue < rhsValue : lhsValue > rhsValue;
        }
        return lhs.id < rhs.id;
    });
}

void EvolutionDashboardWindow::updatePlotData()
{
    //the long-term history carries per-lineage data and is therefore expensive to copy; process it under its mutex instead
    if (_timelineMode == TimelineMode_EntireHistory) {
        auto const& statisticsHistory = _SimulationFacade::get()->getStatisticsHistory();
        std::lock_guard lock(statisticsHistory.getMutex());
        rebuildPlotSeries(statisticsHistory.getDataRef());
    } else {
        rebuildPlotSeries(_timelineLiveStatistics.getDataPointCollectionHistory());
    }
}

void EvolutionDashboardWindow::rebuildPlotSeries(std::vector<DataPointCollection> const& source)
{
    //rebuild only when the underlying data or view settings have changed
    auto sourceBackTime = !source.empty() ? source.back().time : -1.0;
    RebuildKey key{_timelineMode, _lastSteps, _timeHorizon, sourceBackTime, _selectedLineageIds};
    if (_lastRebuildKey && *_lastRebuildKey == key) {
        return;
    }
    _lastRebuildKey = std::move(key);

    auto useTimeAsX = _timelineMode == TimelineMode_RealTime;
    auto getX = [&](DataPointCollection const& sample) { return useTimeAsX ? sample.time : sample.timestep; };

    size_t firstIndex = 0;
    if (!source.empty()) {
        auto startX = std::numeric_limits<double>::lowest();
        if (_timelineMode == TimelineMode_RealTime) {
            startX = source.back().time - toDouble(_timeHorizon);
        } else if (_timelineMode == TimelineMode_LastSteps) {
            startX = source.back().timestep - toDouble(_lastSteps);
        }
        while (firstIndex < source.size() && getX(source.at(firstIndex)) < startX) {
            ++firstIndex;
        }
        //keep one sample left of the visible range so the lines extend beyond the left plot border
        //(they are clipped there); otherwise a flickering gap remains between border and first sample
        if (firstIndex > 0) {
            --firstIndex;
        }
    }

    //"all lineages" series (the current values are maintained by updateTableData)
    _allLineages.id = -1;
    buildPlotSeries(_allLineages, source, firstIndex, useTimeAsX, false, getOverallMetricValue<DataPointCollection>);

    //per-lineage series for the selected lineages
    _plottedLineages.clear();
    for (auto const& selectedId : _selectedLineageIds) {
        LineageDisplayData lineage;
        lineage.id = selectedId;
        struct RateReference
        {
            double windowClock = 0;
            double timestep = 0;
            LineageDataPoint const* entry = nullptr;
        };
        std::vector<RateReference> rateReferences;  //already visited samples containing the lineage; may also lie before the visible range
        size_t rateReferenceIndex = 0;
        for (size_t sampleIndex = 0; sampleIndex < source.size(); ++sampleIndex) {
            auto const& sample = source.at(sampleIndex);
            auto const* entry = findLineageEntry(sample, toUInt32(selectedId));
            if (!entry) {
                continue;
            }
            lineage.colorBitset = toInt(entry->colorBitset);
            auto windowClock = sample.time;
            while (rateReferenceIndex + 1 < rateReferences.size()
                   && rateReferences.at(rateReferenceIndex + 1).windowClock + RateAveragingInterval <= windowClock) {
                ++rateReferenceIndex;
            }
            if (sampleIndex >= firstIndex) {
                lineage.timePoints.emplace_back(getX(sample));
                if (!useTimeAsX) {
                    lineage.systemClockPoints.emplace_back(sample.systemClock);
                }
                //rates over references younger than the averaging interval would produce spikes at the left plot border; NaN suppresses those samples;
                //at the start of the lineage the oldest sample serves as reference once it is at least MinRateTimesteps old (like in the table)
                auto referenceIsOldEnough = !rateReferences.empty()
                    && (rateReferences.at(rateReferenceIndex).windowClock + RateAveragingInterval <= windowClock
                        || (rateReferenceIndex == 0 && rateReferences.at(0).timestep + MinRateTimesteps <= sample.timestep));
                auto const* lastEntry = referenceIsOldEnough ? rateReferences.at(rateReferenceIndex).entry : nullptr;
                auto deltaKSteps = referenceIsOldEnough ? (sample.timestep - rateReferences.at(rateReferenceIndex).timestep) / 1000.0 : 0.0;
                for (int i = 0; i < NumMetrics; ++i) {
                    lineage.series.at(i).emplace_back(getLineageMetricValue(*entry, lastEntry, deltaKSteps, i));
                }
            }
            rateReferences.push_back({windowClock, sample.timestep, entry});
        }
        if (lineage.colorBitset == 0) {
            for (auto const& tableLineage : _lineages) {
                if (tableLineage.id == selectedId) {
                    lineage.colorBitset = tableLineage.colorBitset;
                }
            }
        }
        _plottedLineages.emplace_back(std::move(lineage));
    }
}

void EvolutionDashboardWindow::rebuildPlotSeries(StatisticsHistoryData const& source)
{
    //rebuild only when the underlying data or view settings have changed; the lineage timelines are
    //sampled independently of the overall timeline and therefore contribute their own back times
    auto sourceBackTime = !source.overall.empty() ? source.overall.back().time : -1.0;
    std::vector<double> lineageBackTimes;
    lineageBackTimes.reserve(_selectedLineageIds.size());
    for (auto const& selectedId : _selectedLineageIds) {
        auto timelineIt = source.lineages.find(toUInt32(selectedId));
        lineageBackTimes.emplace_back(timelineIt != source.lineages.end() && !timelineIt->second.empty() ? timelineIt->second.back().time : -1.0);
    }
    RebuildKey key{_timelineMode, _lastSteps, _timeHorizon, sourceBackTime, _selectedLineageIds, std::move(lineageBackTimes)};
    if (_lastRebuildKey && *_lastRebuildKey == key) {
        return;
    }
    _lastRebuildKey = std::move(key);

    //"all lineages" series (the current values are maintained by updateTableData)
    _allLineages.id = -1;
    buildPlotSeries(_allLineages, source.overall, 0, false, true, getOverallMetricValue<OverallSample>);

    //per-lineage series for the selected lineages
    _plottedLineages.clear();
    for (auto const& selectedId : _selectedLineageIds) {
        LineageDisplayData lineage;
        lineage.id = selectedId;
        if (auto timelineIt = source.lineages.find(toUInt32(selectedId)); timelineIt != source.lineages.end() && !timelineIt->second.empty()) {
            lineage.colorBitset = toInt(timelineIt->second.back().data.colorBitset);
            buildPlotSeries(lineage, timelineIt->second, 0, false, true, getLineageSampleMetricValue);
        }
        if (lineage.colorBitset == 0) {
            for (auto const& tableLineage : _lineages) {
                if (tableLineage.id == selectedId) {
                    lineage.colorBitset = tableLineage.colorBitset;
                }
            }
        }
        _plottedLineages.emplace_back(std::move(lineage));
    }
}

void EvolutionDashboardWindow::processHeader()
{
    auto const& style = ImGui::GetStyle();
    auto cardWidth = (ImGui::GetContentRegionAvail().x - 3 * style.ItemSpacing.x) / 6;
    auto cardHeight =
        style.WindowPadding.y * 2 + ImGui::GetTextLineHeight() * 3 + StyleRepository::get().getLargeFont()->FontSize + style.ItemSpacing.y * 3 + scale(6.0f);

    auto const& liveGlobalHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    auto const* lastDataPoints = !liveGlobalHistory.empty() ? &liveGlobalHistory.back() : nullptr;

    auto solids = lastDataPoints ? lastDataPoints->overall.numSolidObjects : 0.0;
    auto fluids = lastDataPoints ? lastDataPoints->overall.numFluidObjects : 0.0;
    auto cells = lastDataPoints ? lastDataPoints->overall.numCellObjects : 0.0;
    auto creatureCells = lastDataPoints ? lastDataPoints->overall.numCreatures * lastDataPoints->overall.averageCreatureCells : 0.0;
    auto freeCells = std::max(0.0, cells - creatureCells);
    auto energyParticles = lastDataPoints ? lastDataPoints->overall.numEnergyParticles : 0.0;
    processCard(
        "Entities",
        formatMetricValue(solids + fluids + cells + energyParticles, 0),
        {{"Solids", formatMetricValue(solids, 0)},
         {"Fluids", formatMetricValue(fluids, 0)},
         {"Cells", formatMetricValue(cells, 0)},
         {"Free cells", formatMetricValue(freeCells, 0)},
         {"Energy particles", formatMetricValue(energyParticles, 0)}},
        cardWidth * 2,
        cardHeight);
    ImGui::SameLine();

    auto numLineages = lastDataPoints ? lastDataPoints->overall.numLineages : 0.0;
    auto numLineagesAbove5Percent = 0;
    auto numLineagesAbove1Percent = 0;
    if (lastDataPoints) {
        for (auto const& [lineageId, entry] : lastDataPoints->lineages) {
            if (entry.numCreatures >= lastDataPoints->overall.numCreatures / 20) {
                ++numLineagesAbove5Percent;
            }
            if (entry.numCreatures >= lastDataPoints->overall.numCreatures / 100) {
                ++numLineagesAbove1Percent;
            }
        }
    }
    processCard(
        "Lineages",
        formatMetricValue(numLineages, 0),
        {{"> 1% creatures", formatMetricValue(toDouble(numLineagesAbove1Percent), 0)},
         {"> 5% creatures", formatMetricValue(toDouble(numLineagesAbove5Percent), 0)}},
        cardWidth,
        cardHeight);
    ImGui::SameLine();

    processCard("Creatures", formatMetricValue(_allLineages.currentValues.at(0), 0), {}, cardWidth, cardHeight);
    ImGui::SameLine();

    auto internalEnergy = lastDataPoints ? lastDataPoints->overall.creatureEnergy : 0.0;
    auto externalEnergy = toDouble(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value);
    processCard(
        "Total energy",
        formatMetricValue(internalEnergy + externalEnergy, 0),
        {{"Internal energy", formatMetricValue(internalEnergy, 0)}, {"External energy", formatMetricValue(externalEnergy, 0)}},
        cardWidth * 2,
        cardHeight);
}

void EvolutionDashboardWindow::processCard(
    std::string const& label,
    std::string const& value,
    std::vector<std::pair<std::string, std::string>> const& subValues,
    float width,
    float height)
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

        auto firstSubValue = true;
        for (auto const& [subLabel, subValue] : subValues) {
            if (!firstSubValue) {
                ImGui::SameLine(0, scale(18.0f));
            }
            firstSubValue = false;

            //draw an ellipsis instead of sub-values that do not fit into the card
            auto groupWidth = std::max(ImGui::CalcTextSize(subLabel.c_str()).x, ImGui::CalcTextSize(subValue.c_str()).x);
            if (ImGui::GetContentRegionAvail().x < groupWidth) {
                ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
                AlienGui::Text("...");
                ImGui::PopStyleColor();
                break;
            }

            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
            AlienGui::Text(subLabel);
            ImGui::PopStyleColor();
            AlienGui::Text(subValue);
            ImGui::EndGroup();
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
    AlienGui::Text("Filter by customizations");
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
        auto color = toImColor(_cellColors.at(i), active ? 1.0f : 0.2f, 0.75f);
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

        auto drawList = ImGui::GetWindowDrawList();

        //header row with labels centered within the visible part of each column; columns can be
        //partially hidden behind the frozen lineage column or cut off at the right window border
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        for (int column = 0; column < NumMetrics + 1; ++column) {
            ImGui::TableSetColumnIndex(column);
            ImGui::PushID(column);
            std::string label = ImGui::TableGetColumnName(column);
            if (column == _sortColumnIndex) {
                label += _sortAscending ? " " ICON_FA_CARET_UP : " " ICON_FA_CARET_DOWN;
            }
            auto textWidth = ImGui::CalcTextSize(label.c_str()).x;
            auto cellPos = ImGui::GetCursorScreenPos();
            auto cellMaxX = cellPos.x + ImGui::GetContentRegionAvail().x;
            if (ImGui::Selectable("##header", false)) {
                if (column == _sortColumnIndex) {
                    _sortAscending = !_sortAscending;
                } else {
                    _sortColumnIndex = column;
                    _sortAscending = column == 0;  //metric columns show the largest values on top by default
                }
                sortLineages();
            }
            auto visibleMinX = std::max(cellPos.x, drawList->GetClipRectMin().x);
            auto visibleMaxX = std::min(cellMaxX, drawList->GetClipRectMax().x);
            auto textPosX = std::clamp((visibleMinX + visibleMaxX - textWidth) / 2, cellPos.x, std::max(cellPos.x, cellMaxX - textWidth));
            drawList->AddText({textPosX, cellPos.y}, ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
            ImGui::PopID();
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

        //draw the pin icon slightly smaller and shifted so it aligns nicely with the row text
        auto iconSize = ImGui::GetFontSize() * 0.75f;
        auto iconPos = ImGui::GetCursorScreenPos();
        drawList->AddText(
            ImGui::GetFont(),
            iconSize,
            {iconPos.x + scale(2.0f), iconPos.y + (ImGui::GetTextLineHeight() - iconSize) / 2 + scale(1.0f)},
            ImGui::GetColorU32(ImGuiCol_Text),
            ICON_FA_THUMBTACK);
        auto iconWidth = ImGui::GetFont()->CalcTextSizeA(iconSize, FLT_MAX, 0.0f, ICON_FA_THUMBTACK).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + iconWidth + scale(7.0f));
        AlienGui::Text("All lineages (" + std::to_string(_lineages.size()) + ")");
        for (int i = 0; i < NumMetrics; ++i) {
            ImGui::TableSetColumnIndex(i + 1);
            rightAlignedText(formatMetricValue(_allLineages.currentValues.at(i), Metrics[i].tableDecimals));
        }

        //lineage rows
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
                rightAlignedText(formatMetricValue(lineage.currentValues.at(i), Metrics[i].tableDecimals));
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void EvolutionDashboardWindow::processTimelineSection()
{
    processTimelineHeader();

    //rebuild the plot series right after the header widgets changed the horizon; otherwise the widened
    //x-axis shows a gap on the left for one frame because the series are still trimmed to the old horizon
    updatePlotData();

    if (ImGui::BeginChild("##timelinePlots", {0, 0})) {
        processTimelinePlots();
    }
    ImGui::EndChild();
}

void EvolutionDashboardWindow::processTimelineHeader()
{
    ImGui::Spacing();
    std::vector<std::string> modeValues{"Real-time", "Last time steps", "Entire history"};
    AlienGui::Switcher(AlienGui::SwitcherParameters().name("Mode").width(260.0f).textWidth(45.0f).values(modeValues), &_timelineMode);

    //sliders keep their default width and only shrink when the window gets too narrow
    auto numSliders = _timelineMode == TimelineMode_EntireHistory ? 1.0f : 2.0f;
    ImGui::SameLine(0, scale(20.0f));
    auto sliderWidth = std::clamp((ImGui::GetContentRegionAvail().x - scale(20.0f) * (numSliders - 1.0f)) / numSliders, scale(140.0f), scale(320.0f));
    if (_timelineMode == TimelineMode_RealTime) {
        if (ImGui::BeginChild("##timeHorizon", {sliderWidth, ImGui::GetFrameHeight()})) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Time horizon").min(1.0f).max(TimelineLiveStatistics::MaxLiveHistory).format("%.1f s").textWidth(100.0f),
                &_timeHorizon);
            validateAndCorrect();
        }
        ImGui::EndChild();
        ImGui::SameLine(0, scale(20.0f));
    }
    if (_timelineMode == TimelineMode_LastSteps) {
        if (ImGui::BeginChild("##lastSteps", {sliderWidth, ImGui::GetFrameHeight()})) {
            AlienGui::SliderInt(
                AlienGui::SliderIntParameters().name("Steps").min(1000).max(TimelineLiveStatistics::MaxLiveSteps).logarithmic(true).textWidth(100.0f),
                &_lastSteps);
            validateAndCorrect();
        }
        ImGui::EndChild();
        ImGui::SameLine(0, scale(20.0f));
    }
    if (ImGui::BeginChild("##plotHeight", {sliderWidth, ImGui::GetFrameHeight()})) {
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Plot height").min(MinPlotHeight).max(MaxPlotHeight).format("%.0f").textWidth(100.0f), &_plotHeight);
        validateAndCorrect();
    }
    ImGui::EndChild();

    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
    AlienGui::Text("Timeline filter");
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

    //labels and current values are placed to the right of the widgets, matching the general ALIEN layout
    if (ImGui::BeginTable("##plots", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("##plot");
        ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, scale(PlotLabelColumnWidth));
        for (int i = 0; i < NumMetrics; ++i) {
            auto showTimeAxis = i == NumMetrics - 1 && _timelineMode == TimelineMode_EntireHistory;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            processTimelinePlot(plottedLineages, i, showTimeAxis);

            ImGui::TableSetColumnIndex(1);
            AlienGui::Text(Metrics[i].plotName);
            auto seriesIndex = 0;
            for (auto const* lineage : plottedLineages) {
                ImGui::PushFont(StyleRepository::get().getMediumBoldFont());
                ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)getPlotColor(i, seriesIndex, toInt(plottedLineages.size())));
                auto const& series = lineage->series.at(i);
                AlienGui::Text(!series.empty() ? formatMetricValue(series.back(), Metrics[i].plotDecimals) : "-");
                ImGui::PopStyleColor();
                ImGui::PopFont();
                ++seriesIndex;
            }
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
            if (std::isfinite(value)) {
                upperBound = std::max(upperBound, value);
            }
        }
    }
    if (!hasData) {
        minTime = 0.0;
        maxTime = 1.0;
    } else if (_timelineMode == TimelineMode_RealTime) {
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

    auto height = _plotHeight + (showTimeAxis ? TimeAxisExtraHeight : 0.0f);
    if (ImPlot::BeginPlot(
            "##plot", ImVec2(-1, scale(height)), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxis(ImAxis_X1, "", showTimeAxis ? ImPlotAxisFlags_None : ImPlotAxisFlags_NoTickLabels);
        if (showTimeAxis) {
            ImPlot::SetupAxisFormat(ImAxis_X1, formatTimestepsInThousands, nullptr);
        }
        ImPlot::SetupAxis(ImAxis_Y1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxisFormat(ImAxis_Y1, "");
        auto seriesIndex = 0;
        for (auto const* lineage : plottedLineages) {
            auto count = toInt(lineage->timePoints.size());
            auto const& series = lineage->series.at(metricIndex);

            //rate series start with NaN samples until a sufficiently old rate reference exists; skip them
            auto offset = 0;
            while (offset < count && std::isnan(series.at(offset))) {
                ++offset;
            }
            count -= offset;
            if (count < 2) {
                ++seriesIndex;
                continue;
            }
            ImGui::PushID(toInt(lineage->id) + 1);
            auto color = getPlotColor(metricIndex, seriesIndex, toInt(plottedLineages.size()));
            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            ImPlot::PushStyleColor(ImPlotCol_Fill, (ImU32)color);
            ImPlot::PlotLine("##line", lineage->timePoints.data() + offset, series.data() + offset, count);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##shaded", lineage->timePoints.data() + offset, series.data() + offset, count);
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor(2);
            if (ImGui::GetStyle().Alpha == 1.0f) {
                ImPlot::Annotation(
                    lineage->timePoints.back(),
                    series.back(),
                    color,
                    ImVec2(-10.0f, 10.0f),
                    true,
                    "%s",
                    formatMetricValue(series.back(), Metrics[metricIndex].plotDecimals).c_str());
            }
            ImGui::PopID();
            ++seriesIndex;
        }
        if (ImGui::GetStyle().Alpha == 1.0f && ImPlot::IsPlotHovered() && plottedLineages.size() == 1 && plottedLineages.front()->timePoints.size() >= 2) {
            auto const* lineage = plottedLineages.front();
            drawValuesAtMouseCursor(
                lineage->series.at(metricIndex),
                lineage->timePoints,
                lineage->systemClockPoints,
                minTime,
                maxTime,
                upperBound + NEAR_ZERO,
                Metrics[metricIndex].plotDecimals);
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void EvolutionDashboardWindow::drawValuesAtMouseCursor(
    std::vector<double> const& series,
    std::vector<double> const& timePoints,
    std::vector<double> const& systemClockPoints,
    double startTime,
    double endTime,
    double upperBound,
    int fracPartDecimals)
{
    auto count = toInt(timePoints.size());
    auto hasSystemClock = !systemClockPoints.empty();

    auto mousePos = ImPlot::GetPlotMousePos();
    mousePos.x = std::max(startTime, std::min(endTime, mousePos.x));
    mousePos.y = series.at(0);
    auto systemClockEntry = hasSystemClock ? systemClockPoints.at(0) : 0.0;
    for (int i = 1; i < count; ++i) {
        if (timePoints.at(i) > mousePos.x) {
            mousePos.y = series.at(i);
            if (hasSystemClock) {
                systemClockEntry = systemClockPoints.at(i);
            }
            break;
        }
    }
    auto valueAtCursor = mousePos.y;  //may be NaN for rate series without a sufficiently old rate reference
    mousePos.y = std::isnan(mousePos.y) ? 0.0 : std::max(0.0, std::min(upperBound, mousePos.y));

    ImPlot::PushStyleColor(ImPlotCol_InlayText, ImColor::HSV(0.0f, 0.0f, 1.0f).Value);
    ImPlot::PlotText(ICON_FA_GENDERLESS, mousePos.x, mousePos.y, {scale(1.0f), scale(2.0f)});
    ImPlot::PopStyleColor();

    ImPlot::PushStyleColor(ImPlotCol_Line, ImColor::HSV(0.0f, 0.0f, 1.0f).Value);
    ImPlot::PlotInfLines("", &mousePos.x, 1);
    ImPlot::PopStyleColor();

    char label[256];
    auto leftSideFactor = mousePos.x > (startTime + endTime) / 2 ? -1.0f : 1.0f;
    if (hasSystemClock) {
        auto dateTimeString = systemClockEntry != 0 ? convertSystemClockToString(systemClockEntry) : std::string("-");
        snprintf(
            label,
            sizeof(label),
            "Time step: %s\nTimestamp: %s\nValue: %s",
            StringHelper::format(toFloat(mousePos.x), 0).c_str(),
            dateTimeString.c_str(),
            formatMetricValue(valueAtCursor, fracPartDecimals).c_str());
    } else {
        snprintf(
            label,
            sizeof(label),
            "Relative time: %s\nValue: %s",
            StringHelper::format(toFloat(mousePos.x), 0).c_str(),
            formatMetricValue(valueAtCursor, fracPartDecimals).c_str());
    }
    ImPlot::PlotText(label, mousePos.x, upperBound, {leftSideFactor * (scale(5.0f) + ImGui::CalcTextSize(label).x / 2), scale(28.0f)});
}

void EvolutionDashboardWindow::validateAndCorrect()
{
    _timelinesHeight = std::max(scale(100.0f), _timelinesHeight);
    _timelineMode = std::clamp(_timelineMode, static_cast<TimelineMode>(TimelineMode_RealTime), static_cast<TimelineMode>(TimelineMode_EntireHistory));
    _lastSteps = std::clamp(_lastSteps, 1000, TimelineLiveStatistics::MaxLiveSteps);
    _timeHorizon = std::clamp(_timeHorizon, 1.0f, TimelineLiveStatistics::MaxLiveHistory);
    _plotHeight = std::clamp(_plotHeight, MinPlotHeight, MaxPlotHeight);
    _sortColumnIndex = std::clamp(_sortColumnIndex, 0, NumMetrics);
    _colorFilter &= (1 << MAX_COLORS) - 1;
    if (_colorFilter == 0) {
        _colorFilter = (1 << MAX_COLORS) - 1;
    }
}
