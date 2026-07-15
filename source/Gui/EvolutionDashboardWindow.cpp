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

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr LiveStatisticsDeltaTime = 50;  //in millisec
    auto constexpr RateAveragingInterval = 30.0;  //in seconds

    auto constexpr ColorChipSize = 22.0f;
    auto constexpr SwatchSize = 11.0f;
    auto constexpr SwatchGap = 3.0f;
    auto constexpr PlotLabelColumnWidth = 150.0f;
    auto constexpr PlotHeight = 60.0f;
    auto constexpr PlotHeightWithAxis = 80.0f;
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

    //the last two metrics are rates: per second in the table, per 1K time steps in the timeline plots
    MetricDef const Metrics[EvolutionDashboardWindow::NumMetrics] = {
        {"Creatures", "Creatures", 0, 0},
        {"Avg cells", "Avg cells", 1, 1},
        {"Avg nodes", "Avg nodes", 1, 1},
        {"Internal energy", "Internal energy", 0, 0},
        {"Avg mut. rate", "Avg mutation rate", 4, 4},
        {"Avg age (gen)", "Avg age (generations)", 0, 0},
        {"Created /s", "Created creatures / 1K steps", 2, 2},
        {"Mutations /s", "Mutations / 1K steps", 4, 4},
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
    //or one per selected lineage if multiple lineages are plotted together
    ImColor getPlotColor(int metricIndex, int seriesIndex, int numSeries)
    {
        auto index = numSeries > 1 ? seriesIndex : metricIndex;
        return ImColor(ImPlot::GetColormapColor((index % 21) <= 10 ? (index % 21) : 20 - (index % 21), ImPlotColormap_Cool));
    }

    std::string formatMetricValue(double value, int decimals)
    {
        if (std::isinf(value)) {
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

    double getGlobalMetricValue(DataPointCollection const& dataPoints, DataPointCollection const* lastDataPoints, bool ratePerKSteps, int metricIndex)
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
            auto delta = ratePerKSteps ? (dataPoints.timestep - lastDataPoints->timestep) / 1000.0 : dataPoints.time - lastDataPoints->time;
            if (metricIndex == 6) {
                return calcRate(dataPoints.overall.accumCreatedCreatures, lastDataPoints->overall.accumCreatedCreatures, delta);
            } else {
                return calcRate(dataPoints.overall.accumMutations, lastDataPoints->overall.accumMutations, delta);
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
            return lastEntry ? calcRate(entry.numCreatedCreatures, lastEntry->numCreatedCreatures, rateDelta) : 0.0;
        case 7:
            return lastEntry ? calcRate(entry.totalMutations, lastEntry->totalMutations, rateDelta) : 0.0;
        default:
            return 0.0;
        }
    }

    LineageDataPoint const* findLineageEntry(DataPointCollection const& sample, uint32_t lineageId)
    {
        auto it = sample.lineages.find(lineageId);
        return it != sample.lineages.end() ? &it->second : nullptr;
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

    //apply selection changes from the table immediately; otherwise the plots show "No data" for one
    //frame, which collapses the plot child window and resets its scroll position
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

    //table rows from the latest live sample (rates per second)
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
        auto referenceTime = 0.0;
        for (auto sampleIndex = referenceIndex; sampleIndex + 1 < liveHistory.size(); ++sampleIndex) {  //young lineages: use their oldest sample
            if (auto const* candidate = findLineageEntry(liveHistory.at(sampleIndex), lineageId)) {
                referenceEntry = candidate;
                referenceTime = liveHistory.at(sampleIndex).time;
                break;
            }
        }
        for (int i = 0; i < NumMetrics; ++i) {
            lineage.currentValues.at(i) = getLineageMetricValue(entry, referenceEntry, lastSample.time - referenceTime, i);
        }
        _lineages.emplace_back(std::move(lineage));
    }
    std::sort(_lineages.begin(), _lineages.end(), [](auto const& lhs, auto const& rhs) { return lhs.currentValues.at(0) > rhs.currentValues.at(0); });

    //current values for the table summary row
    DataPointCollection const* referenceDataPoints = liveHistory.size() >= 2 ? &liveHistory.at(referenceIndex) : nullptr;
    for (int i = 0; i < NumMetrics; ++i) {
        _allLineages.currentValues.at(i) = getGlobalMetricValue(lastSample, referenceDataPoints, false, i);
    }
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

    //reference samples for the rate metrics are selected via a trailing ~30s window; the live buffer carries the
    //time since simulation start while the long-term history only provides the system clock
    auto getRateWindowClock = [&](DataPointCollection const& sample) { return _timelineMode == TimelineMode_EntireHistory ? sample.systemClock : sample.time; };

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
    }

    //"all lineages" series (the current values are maintained by updateTableData)
    _allLineages.id = -1;
    _allLineages.series = {};
    _allLineages.timePoints.clear();
    _allLineages.systemClockPoints.clear();
    auto globalReferenceIndex = firstIndex;
    for (auto sampleIndex = firstIndex; sampleIndex < source.size(); ++sampleIndex) {
        auto const& dataPoints = source.at(sampleIndex);
        auto rateWindowClock = getRateWindowClock(dataPoints);
        while (globalReferenceIndex + 1 < sampleIndex && getRateWindowClock(source.at(globalReferenceIndex + 1)) + RateAveragingInterval <= rateWindowClock) {
            ++globalReferenceIndex;
        }
        auto const* referenceDataPoints = sampleIndex > firstIndex ? &source.at(globalReferenceIndex) : nullptr;
        _allLineages.timePoints.emplace_back(getX(dataPoints));
        if (!useTimeAsX) {
            _allLineages.systemClockPoints.emplace_back(dataPoints.systemClock);
        }
        for (int i = 0; i < NumMetrics; ++i) {
            _allLineages.series.at(i).emplace_back(getGlobalMetricValue(dataPoints, referenceDataPoints, true, i));
        }
    }

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
        std::vector<RateReference> rateReferences;  //already visited samples containing the lineage
        size_t rateReferenceIndex = 0;
        for (auto sampleIndex = firstIndex; sampleIndex < source.size(); ++sampleIndex) {
            auto const& sample = source.at(sampleIndex);
            auto const* entry = findLineageEntry(sample, toUInt32(selectedId));
            if (!entry) {
                continue;
            }
            lineage.colorBitset = toInt(entry->colorBitset);
            lineage.timePoints.emplace_back(getX(sample));
            if (!useTimeAsX) {
                lineage.systemClockPoints.emplace_back(sample.systemClock);
            }
            auto windowClock = getRateWindowClock(sample);
            while (rateReferenceIndex + 1 < rateReferences.size()
                   && rateReferences.at(rateReferenceIndex + 1).windowClock + RateAveragingInterval <= windowClock) {
                ++rateReferenceIndex;
            }
            auto const* lastEntry = !rateReferences.empty() ? rateReferences.at(rateReferenceIndex).entry : nullptr;
            auto deltaKSteps = !rateReferences.empty() ? (sample.timestep - rateReferences.at(rateReferenceIndex).timestep) / 1000.0 : 0.0;
            for (int i = 0; i < NumMetrics; ++i) {
                lineage.series.at(i).emplace_back(getLineageMetricValue(*entry, lastEntry, deltaKSteps, i));
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
    auto energyParticles = lastDataPoints ? lastDataPoints->overall.numEnergyParticles : 0.0;
    processCard(
        "Entities",
        formatMetricValue(solids + fluids + cells + energyParticles, 0),
        {{"Solids", formatMetricValue(solids, 0)},
         {"Fluid particles", formatMetricValue(fluids, 0)},
         {"Cells", formatMetricValue(cells, 0)},
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
        {{">= 5% creatures", formatMetricValue(toDouble(numLineagesAbove5Percent), 0)},
         {">= 1% creatures", formatMetricValue(toDouble(numLineagesAbove1Percent), 0)}},
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

        for (auto const& [subLabel, subValue] : subValues) {
            ImGui::BeginGroup();
            ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
            AlienGui::Text(subLabel);
            ImGui::PopStyleColor();
            AlienGui::Text(subValue);
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
            rightAlignedText(formatMetricValue(_allLineages.currentValues.at(i), Metrics[i].tableDecimals));
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
    if (ImGui::BeginChild("##timelinePlots", {0, 0})) {
        processTimelinePlots();
    }
    ImGui::EndChild();
}

void EvolutionDashboardWindow::processTimelineHeader()
{
    ImGui::Spacing();
    std::vector<std::string> modeValues{"Real-time", "Last X steps", "Entire history"};
    AlienGui::Switcher(AlienGui::SwitcherParameters().name("Mode").width(260.0f).textWidth(45.0f).values(modeValues), &_timelineMode);
    if (_timelineMode == TimelineMode_RealTime) {
        ImGui::SameLine(0, scale(20.0f));
        if (ImGui::BeginChild("##timeHorizon", {scale(320.0f), ImGui::GetFrameHeight()})) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Time horizon").min(1.0f).max(TimelineLiveStatistics::MaxLiveHistory).format("%.1f s").textWidth(100.0f),
                &_timeHorizon);
            validateAndCorrect();
        }
        ImGui::EndChild();
    }
    if (_timelineMode == TimelineMode_LastSteps) {
        ImGui::SameLine(0, scale(20.0f));
        if (ImGui::BeginChild("##lastSteps", {scale(320.0f), ImGui::GetFrameHeight()})) {
            AlienGui::SliderInt(
                AlienGui::SliderIntParameters().name("Steps").min(1000).max(TimelineLiveStatistics::MaxLiveSteps).logarithmic(true).textWidth(100.0f),
                &_lastSteps);
            validateAndCorrect();
        }
        ImGui::EndChild();
    }

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

    if (ImGui::BeginTable("##plots", 2, ImGuiTableFlags_None)) {
        ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed, scale(PlotLabelColumnWidth));
        ImGui::TableSetupColumn("##plot");
        for (int i = 0; i < NumMetrics; ++i) {
            auto showTimeAxis = i == NumMetrics - 1 && _timelineMode == TimelineMode_EntireHistory;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
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
        auto seriesIndex = 0;
        for (auto const* lineage : plottedLineages) {
            auto count = toInt(lineage->timePoints.size());
            if (count < 2) {
                ++seriesIndex;
                continue;
            }
            ImGui::PushID(toInt(lineage->id) + 1);
            auto color = getPlotColor(metricIndex, seriesIndex, toInt(plottedLineages.size()));
            auto const& series = lineage->series.at(metricIndex);
            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            ImPlot::PushStyleColor(ImPlotCol_Fill, (ImU32)color);
            ImPlot::PlotLine("##line", lineage->timePoints.data(), series.data(), count);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##shaded", lineage->timePoints.data(), series.data(), count);
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
    mousePos.y = std::max(0.0, std::min(upperBound, mousePos.y));

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
            formatMetricValue(mousePos.y, fracPartDecimals).c_str());
    } else {
        snprintf(
            label,
            sizeof(label),
            "Relative time: %s\nValue: %s",
            StringHelper::format(toFloat(mousePos.x), 0).c_str(),
            formatMetricValue(mousePos.y, fracPartDecimals).c_str());
    }
    ImPlot::PlotText(label, mousePos.x, upperBound, {leftSideFactor * (scale(5.0f) + ImGui::CalcTextSize(label).x / 2), scale(28.0f)});
}

void EvolutionDashboardWindow::validateAndCorrect()
{
    _timelinesHeight = std::max(scale(100.0f), _timelinesHeight);
    _timelineMode = std::clamp(_timelineMode, static_cast<TimelineMode>(TimelineMode_RealTime), static_cast<TimelineMode>(TimelineMode_EntireHistory));
    _lastSteps = std::clamp(_lastSteps, 1000, TimelineLiveStatistics::MaxLiveSteps);
    _timeHorizon = std::clamp(_timeHorizon, 1.0f, TimelineLiveStatistics::MaxLiveHistory);
    _colorFilter &= (1 << MAX_COLORS) - 1;
    if (_colorFilter == 0) {
        _colorFilter = (1 << MAX_COLORS) - 1;
    }
}
