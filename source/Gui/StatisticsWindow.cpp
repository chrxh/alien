#include "StatisticsWindow.h"

#include <fstream>
#include <cmath>

#include <boost/algorithm/string.hpp>

#include <ImFileDialog.h>
#include <imgui.h>
#include <implot.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/GlobalSettings.h"
#include "Base/StringHelper.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/StatisticsHistory.h"
#include "EngineInterface/SerializerService.h"

#include "StyleRepository.h"
#include "AlienImGui.h"
#include "GenericFileDialogs.h"
#include "GenericMessageDialog.h"

namespace
{
    auto constexpr RightColumnWidth = 175.0f;
    auto constexpr RightColumnWidthTimeline = 150.0f;
    auto constexpr RightColumnWidthTable = 200.0f;
    auto constexpr LiveStatisticsDeltaTime = 50;  //in millisec
}

void StatisticsWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;

    auto path = std::filesystem::current_path();
    if (path.has_parent_path()) {
        path = path.parent_path();
    }
    _startingPath = GlobalSettings::get().getString("windows.statistics.starting path", path.string());
    _settingsOpen = GlobalSettings::get().getBool("windows.statistics.settings.open", _settingsOpen);
    _settingsHeight = GlobalSettings::get().getFloat("windows.statistics.settings.height", _settingsHeight);
    _plotHeight = GlobalSettings::get().getFloat("windows.statistics.plot height", _plotHeight);
    _plotMode = GlobalSettings::get().getInt("windows.statistics.mode", _plotMode);
    _timeHorizonForLiveStatistics = GlobalSettings::get().getFloat("windows.statistics.live horizon", _timeHorizonForLiveStatistics);
    _timeHorizonForLongtermStatistics = GlobalSettings::get().getFloat("windows.statistics.long term horizon", _timeHorizonForLongtermStatistics);
    _plotType = GlobalSettings::get().getInt("windows.statistics.plot type", _plotType);
    _plotScale = GlobalSettings::get().getInt("windows.statistics.plot scale", _plotScale);
    auto collapsedPlotIndexJoinedString = GlobalSettings::get().getString("windows.statistics.collapsed plot indices", "");
    
    if (!collapsedPlotIndexJoinedString.empty()) {
        std::vector<std::string> collapsedPlotIndexStrings;
        boost::split(collapsedPlotIndexStrings, collapsedPlotIndexJoinedString, boost::is_any_of(" "));
        for (auto const& s : collapsedPlotIndexStrings) {
            _collapsedPlotIndices.emplace(std::stoi(s));
        }
    }
}

StatisticsWindow::StatisticsWindow()
    : AlienWindow("Statistics", "windows.statistics", false)
{
}

void StatisticsWindow::shutdownIntern()
{
    GlobalSettings::get().setString("windows.statistics.starting path", _startingPath);
    GlobalSettings::get().setBool("windows.statistics.settings.open", _settingsOpen);
    GlobalSettings::get().setFloat("windows.statistics.settings.height", _settingsHeight);
    GlobalSettings::get().setFloat("windows.statistics.plot height", _plotHeight);
    GlobalSettings::get().setInt("windows.statistics.mode", _plotMode);
    GlobalSettings::get().setFloat("windows.statistics.live horizon", _timeHorizonForLiveStatistics);
    GlobalSettings::get().setFloat("windows.statistics.long term horizon", _timeHorizonForLongtermStatistics);
    GlobalSettings::get().setInt("windows.statistics.plot type", _plotType);
    GlobalSettings::get().setInt("windows.statistics.plot scale", _plotScale);

    std::vector<std::string> collapsedPlotIndexStrings;
    for (auto const& index : _collapsedPlotIndices) {
        collapsedPlotIndexStrings.emplace_back(std::to_string(index));
    }
    GlobalSettings::get().setString("windows.statistics.collapsed plot indices", boost::join(collapsedPlotIndexStrings, " "));
}

void StatisticsWindow::processIntern()
{
    if (ImGui::BeginChild("##statistics", {0, _settingsOpen ? -scale(_settingsHeight) : -scale(50.0f)})) {
        if (ImGui::BeginTabBar("##Statistics", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

            if (ImGui::BeginTabItem("Timelines")) {
                if (ImGui::BeginChild("##timelines", ImVec2(0, 0), false)) {
                    processTimelinesTab();
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Histograms")) {
                if (ImGui::BeginChild("##histograms", ImVec2(0, 0), false)) {
                    processHistogramsTab();
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Throughput")) {
                if (ImGui::BeginChild("##throughput", ImVec2(0, 0), false)) {
                    processTablesTab();
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::EndChild();
    processSettings();
}

void StatisticsWindow::processTimelinesTab()
{
    ImGui::Spacing();

    AlienImGui::Switcher(
        AlienImGui::SwitcherParameters()
            .name("Mode")
            .textWidth(RightColumnWidth)
            .values(
                {"Real-time plots", "Entire history plots"}),
        _plotMode);

    AlienImGui::Switcher(
        AlienImGui::SwitcherParameters()
            .name("Plot type")
            .textWidth(RightColumnWidth)
            .values(
            {"Accumulate values for all colors", "Break down by color", "Color #0", "Color #1", "Color #2", "Color #3", "Color #4", "Color #5", "Color #6"}),
        _plotType);
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();

    if (ImGui::BeginChild("##plots", ImVec2(0, 0), false)) {
        processTimelineStatistics();
    }
    ImGui::EndChild();
}

void StatisticsWindow::processHistogramsTab()
{
    if (!_histogramLiveStatistics.isDataAvailable()) {
        return;
    }
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha * 0.5 * Const::WindowAlpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha * 0.5 * Const::WindowAlpha));
    auto const& histogramData = _histogramLiveStatistics.getData();
    auto maxNumObjects = 0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_HISTOGRAM_SLOTS; ++j) {
            auto value = histogramData->numCellsByColorBySlot[i][j];
            maxNumObjects = std::max(maxNumObjects, value);
        }
    }

    //round maxNumObjects
    if (!_histogramUpperBound || toFloat(maxNumObjects) > *_histogramUpperBound * 0.9f || toFloat(maxNumObjects) < *_histogramUpperBound * 0.5f) {
        _histogramUpperBound = toFloat(maxNumObjects) * 1.3f;
    }

    ImPlot::SetNextAxesLimits(0, toFloat(MAX_HISTOGRAM_SLOTS), 0, *_histogramUpperBound, ImGuiCond_Always);

    auto getLabelString = [](int value) {
        if (value >= 1000) {
            return std::to_string(value / 1000) + "K";
        } else {
            return std::to_string(value);
        }
    };

    //y-ticks
    char const* labelsY[6];
    double positionsY[6];
    for (int i = 0; i < 5; ++i) {
        labelsY[i] = "";
        positionsY[i] = *_histogramUpperBound / 5 * i;
    }
    auto temp = getLabelString(maxNumObjects);
    labelsY[5] = temp.c_str();
    positionsY[5] = toFloat(maxNumObjects);

    //x-ticks
    char const* labelsX[5];
    std::string labelsX_temp[5];
    double positionsX[5];

    auto slotAge = histogramData->maxValue / MAX_HISTOGRAM_SLOTS;
    for (int i = 0; i < 5; ++i) {
        labelsX_temp[i] = getLabelString(slotAge * ((MAX_HISTOGRAM_SLOTS - 1) / 4) * i);
        labelsX[i] = labelsX_temp[i].c_str();
        positionsX[i] = toFloat(((MAX_HISTOGRAM_SLOTS - 1) / 4) * i);
    }


    //plot histogram
    if (ImPlot::BeginPlot("##Histograms", ImVec2(-1, -1))) {
        ImPlot::SetupAxisTicks(ImAxis_Y1, positionsY, 5, labelsY);
        ImPlot::SetupAxisTicks(ImAxis_X1, positionsX, 5, labelsX);
        ImPlot::SetupAxes("Age", "Cell count");
        ImPlot::SetupAxisFormat(ImAxis_X1, "");
        auto const width = 1.0f / MAX_COLORS;
        for (int i = 0; i < MAX_COLORS; ++i) {
            float h, s, v;
            AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[i], h, s, v);
            ImPlot::PushStyleColor(ImPlotCol_Fill, (ImVec4)ImColor::HSV(h, s /** 3 / 4*/, v /** 3 / 4*/, ImGui::GetStyle().Alpha));
            ImPlot::PlotBars((" ##" + std::to_string(i)).c_str(), histogramData->numCellsByColorBySlot[i], MAX_HISTOGRAM_SLOTS, width, width * i);
            ImPlot::PopStyleColor(1);
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleColor(2);
}

void StatisticsWindow::processTablesTab()
{
    if (!_tableLiveStatistics.isDataAvailable()) {
        return;
    }

    ImGui::PushID(3);
    if (ImGui::BeginTable(
            "##throughput",
            2,
            ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_BordersOuterV,
            ImVec2(-1, 0))) {

        ImGui::TableSetupColumn("##");
        ImGui::TableSetupColumn("##", ImGuiTableColumnFlags_WidthFixed, scale(RightColumnWidthTable));

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text(StringHelper::format(_tableLiveStatistics.getCreatedCellsPerSecond()));

        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Created cells / sec");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        AlienImGui::Text(StringHelper::format(_tableLiveStatistics.getCreatedReplicatorsPerSecond()));

        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Created self-replicators / sec");

        ImGui::EndTable();
    }
    ImGui::PopID();
}

void StatisticsWindow::processSettings()
{
    if (_settingsOpen) {
        ImGui::Spacing();
        ImGui::Spacing();
        AlienImGui::MovableSeparator(_settingsHeight);
    } else {
        AlienImGui::Separator();
    }

    _settingsOpen = AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().text("Settings").highlighted(true).defaultOpen(_settingsOpen));
    if (_settingsOpen) {
        if (ImGui::BeginChild("##addons", {scale(0), 0})) {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scale(RightColumnWidth));
            if (_plotMode == 0) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name("Time horizon")
                        .min(1.0f)
                        .max(TimelineLiveStatistics::MaxLiveHistory)
                        .format("%.1f s")
                        .textWidth(RightColumnWidth),
                    &_timeHorizonForLiveStatistics);
            }
            if (_plotMode == 1) {
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters().name("Time horizon").min(1.0f).max(100.0f).format("%.0f percent").textWidth(RightColumnWidth),
                    &_timeHorizonForLongtermStatistics);
            }

            AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters().name("Plot height").min(MinPlotHeight).max(1000.0f).format("%.0f").textWidth(RightColumnWidth),
                &_plotHeight);
            AlienImGui::Switcher(AlienImGui::SwitcherParameters().name("Scale").textWidth(RightColumnWidth).values({"Linear", "Logarithmic"}), _plotScale);
        }
        ImGui::EndChild();
        AlienImGui::EndTreeNode();
    }
}

void StatisticsWindow::processTimelineStatistics()
{
    ImGui::Spacing();
    AlienImGui::Group("Time step data");
    ImGui::PushID(1);
    int row = 0;
    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_BordersInnerH, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("##");
        ImGui::TableSetupColumn("##", ImGuiTableColumnFlags_WidthFixed, scale(RightColumnWidthTimeline));

        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numCells);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Cells");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numConnections);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Cell connections");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numParticles);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Energy particles");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::totalEnergy);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Contained energy");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numSelfReplicators);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Self-replicators");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numColonies);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Diversity");
        ImGui::SameLine();
        AlienImGui::HelpMarker("The number of colonies is displayed. A colony is a set of at least 20 same mutants.");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::averageGenomeCells, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Num genotype\ncells average");
        ImGui::SameLine();
        AlienImGui::HelpMarker("The average number of encoded cells in the genomes is displayed.");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::averageGenomeComplexity, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Genome complexity\naverage");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::varianceGenomeComplexity, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Genome complexity\nvariance");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::maxGenomeComplexityOfColonies, 2);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Genome complexity\nmaximum");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numViruses);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Viruses");

        ImPlot::PopColormap();

        ImGui::EndTable();
    }
    ImGui::PopID();

    ImGui::Spacing();
    AlienImGui::Group("Processes per time step and cell");
    ImGui::PushID(2);
    if (ImGui::BeginTable("##", 2, ImGuiTableFlags_BordersInnerH, ImVec2(-1, 0))) {
        ImGui::TableSetupColumn("##");
        ImGui::TableSetupColumn("##", ImGuiTableColumnFlags_WidthFixed, scale(RightColumnWidthTimeline));
        ImPlot::PushColormap(ImPlotColormap_Cool);

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numCreatedCells, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Created cells");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numAttacks, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Attacks");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numMuscleActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Muscle activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numTransmitterActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Transmitter activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numDefenderActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Defender activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numNervePulses, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Nerve pulses");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numNeuronActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Neural activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numSensorActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Sensor activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numSensorMatches, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Sensor matches");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numInjectionActivities, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Injection activities");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numCompletedInjections, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Completed injections");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numReconnectorCreated, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Reconnector creations");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numReconnectorRemoved, 6);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Reconnector deletions");

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        processPlot(row++, &DataPointCollection::numDetonations, 8);
        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text("Detonations");

        ImPlot::PopColormap();
        ImGui::EndTable();
    }
    ImGui::PopID();
}

void StatisticsWindow::processPlot(int row, DataPoint DataPointCollection::*valuesPtr, int fracPartDecimals)
{
    auto isCollapsed = _collapsedPlotIndices.contains(row);
    ImGui::PushID(row);
    if (AlienImGui::CollapseButton(isCollapsed)) {
        if (isCollapsed) {
            _collapsedPlotIndices.erase(row);
        } else {
            _collapsedPlotIndices.insert(row);
        }
    }
    ImGui::PopID();
    ImGui::SameLine();

    auto const& statisticsHistory = _simulationFacade->getStatisticsHistory();

    std::lock_guard lock(statisticsHistory.getMutex());
    auto longtermStatistics = &statisticsHistory.getDataRef();

    //create dummy history if empty
    std::vector dummy = {DataPointCollection()};
    if (longtermStatistics->empty()) {
        longtermStatistics = &dummy;
    }

    auto const& dataPointCollectionHistory = _timelineLiveStatistics.getDataPointCollectionHistory();
    auto count = _plotMode == 0 ? toInt(dataPointCollectionHistory.size()) : toInt(longtermStatistics->size());
    auto startTime = _plotMode == 0 ? dataPointCollectionHistory.back().time - toDouble(_timeHorizonForLiveStatistics)
        : longtermStatistics->back().time - (longtermStatistics->back().time - longtermStatistics->front().time) * toDouble(_timeHorizonForLongtermStatistics) / 100;
    auto endTime = _plotMode == 0 ? dataPointCollectionHistory.back().time : longtermStatistics->back().time;
    auto values = _plotMode == 0 ? &(dataPointCollectionHistory[0].*valuesPtr) : &((*longtermStatistics)[0].*valuesPtr);
    auto timePoints = _plotMode == 0 ? &dataPointCollectionHistory[0].time : &(*longtermStatistics)[0].time;
    auto systemClock = _plotMode == 0 ? nullptr : &(*longtermStatistics)[0].systemClock;

    switch (_plotType) {
    case 0:
        plotSumColorsIntern(row, values, timePoints, systemClock, count, startTime, endTime, fracPartDecimals);
        break;
    case 1:
        plotByColorIntern(row, values, timePoints, count, startTime, endTime, fracPartDecimals);
        break;
    default:
        plotForColorIntern(row, values, _plotType - 2, timePoints, systemClock, count, startTime, endTime, fracPartDecimals);
        break;
    }
    ImGui::Spacing();
}

void StatisticsWindow::processBackground()
{
    auto timepoint = std::chrono::steady_clock::now();
    auto duration = _lastTimepoint.has_value() ? static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_lastTimepoint).count()) : 0;
    if(!_lastTimepoint || duration > LiveStatisticsDeltaTime) {
        _lastTimepoint = timepoint;
        auto rawStatistics = _simulationFacade->getRawStatistics();
        _histogramLiveStatistics.update(rawStatistics.histogram);
        _timelineLiveStatistics.update(rawStatistics.timeline, _simulationFacade->getCurrentTimestep());
        _tableLiveStatistics.update(rawStatistics.timeline);
    }
}

namespace
{
    double getMaxWithDataPointStride(double const* data, double const* timePoints, double startTime, int count)
    {
        auto constexpr strideDouble = sizeof(DataPointCollection) / sizeof(double);

        auto result = 0.0;
        for (int i = count / 20; i < count; ++i) {
            if (timePoints[i * strideDouble] >= startTime - NEAR_ZERO) {
                result = std::max(result, data[i * strideDouble]);
            }
        }
        return result;
    }
}

void StatisticsWindow::plotSumColorsIntern(
    int row,
    DataPoint const* dataPoints,
    double const* timePoints,
    double const* systemClock,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto constexpr strideBytes = sizeof(DataPointCollection);
    auto constexpr strideDouble = sizeof(DataPointCollection) / sizeof(double);

    double const* plotDataY = reinterpret_cast<double const*>(dataPoints) + MAX_COLORS;
    double upperBound = getMaxWithDataPointStride(plotDataY, timePoints, startTime, count);
    double endValue = count > 0 ? plotDataY[(count - 1) * strideDouble] : 0.0;
    upperBound = getUpperBound(upperBound);

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextAxesLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);

    if (ImPlot::BeginPlot("##", ImVec2(-1, scale(calcPlotHeight(row))), ImPlotFlags_NoMouseText)) {
        ImPlot::SetupAxis(ImAxis_X1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxisFormat(ImAxis_X1, "");
        ImPlot::SetupAxisFormat(ImAxis_Y1, "");
        setPlotScale();
        auto color = ImPlot::GetColormapColor((row % 21) <= 10 ? (row % 21) : 20 - (row % 21));
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::Annotation(
                endTime, endValue, ImPlot::GetLastItemColor(), ImVec2(-10.0f, 15.0f), true, "%s", StringHelper::format(toFloat(endValue), fracPartDecimals).c_str());
        }
        if (count > 0) {
            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("##", timePoints, plotDataY, count, 0, 0, strideBytes);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", timePoints, plotDataY, count, 0, 0, 0, strideBytes);
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
        }
        if (ImGui::GetStyle().Alpha == 1.0f && ImPlot::IsPlotHovered() && count > 0) {
            drawValuesAtMouseCursor(plotDataY, timePoints, systemClock, count, startTime, endTime, upperBound, fracPartDecimals);
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void StatisticsWindow::plotByColorIntern(
    int row,
    DataPoint const* values,
    double const* timePoints,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto upperBound = 0.0;
    for (int i = 0; i < MAX_COLORS; ++i) {
        upperBound = std::max(upperBound, getMaxWithDataPointStride(reinterpret_cast<double const*>(values) + i, timePoints, startTime, count));
    }
    upperBound = getUpperBound(upperBound);

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
    ImPlot::SetNextAxesLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);

    auto isCollapsed = _collapsedPlotIndices.contains(row);
    auto flags = _plotHeight > 159.0f && !isCollapsed ? ImPlotFlags_None : ImPlotFlags_NoLegend;
    if (ImPlot::BeginPlot("##", ImVec2(-1, scale(calcPlotHeight(row))), flags)) {
        ImPlot::SetupAxis(ImAxis_X1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxisFormat(ImAxis_X1, "");
        ImPlot::SetupAxisFormat(ImAxis_Y1, "");
        setPlotScale();
        for (int i = 0; i < MAX_COLORS; ++i) {
            ImGui::PushID(i);
            auto colorRaw = Const::IndividualCellColors[i];
            ImColor color(toInt((colorRaw >> 16) & 0xff), toInt((colorRaw >> 8) & 0xff), toInt(colorRaw & 0xff));

            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            auto endValue = count > 0 ? *(reinterpret_cast<double const*>(reinterpret_cast<DataPointCollection const*>(values) + (count - 1)) + i) : 0.0f;
            auto labelId = StringHelper::format(toFloat(endValue), fracPartDecimals);
            ImPlot::PlotLine(labelId.c_str(), timePoints, reinterpret_cast<double const*>(values) + i, count, 0, 0, sizeof(DataPointCollection));
            ImPlot::PopStyleColor();
            ImGui::PopID();
        }

        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar(2);
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void StatisticsWindow::plotForColorIntern(
    int row,
    DataPoint const* values,
    int colorIndex,
    double const* timePoints,
    double const* systemClock,
    int count,
    double startTime,
    double endTime,
    int fracPartDecimals)
{
    auto constexpr strideBytes = sizeof(DataPointCollection);
    auto constexpr strideDouble = sizeof(DataPointCollection) / sizeof(double);

    auto valuesForColor = reinterpret_cast<double const*>(values) + colorIndex;
    auto upperBound = getMaxWithDataPointStride(valuesForColor, timePoints, startTime, count);
    upperBound = getUpperBound(upperBound);
    auto endValue = count > 0 ? valuesForColor[(count - 1) * strideDouble] : 0.0;

    ImGui::PushID(row);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextAxesLimits(startTime, endTime, 0, upperBound, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##", ImVec2(-1, scale(calcPlotHeight(row))))) {
        ImPlot::SetupAxis(ImAxis_X1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, "", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxisFormat(ImAxis_X1, "");
        ImPlot::SetupAxisFormat(ImAxis_Y1, "");
        setPlotScale();

        float h, s, v;
        AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[colorIndex], h, s, v);
        auto color = static_cast<ImVec4>(ImColor::HSV(h, s, v));
        if (ImGui::GetStyle().Alpha == 1.0f) {
            ImPlot::Annotation(
                endTime,
                endValue,
                ImPlot::GetLastItemColor(),
                ImVec2(-10.0f, 10.0f),
                true,
                "%s",
                StringHelper::format(toFloat(endValue), fracPartDecimals).c_str());
        }
        if (count > 0) {
            ImPlot::PushStyleColor(ImPlotCol_Line, color);
            ImPlot::PlotLine("##", timePoints, valuesForColor, count, 0, 0, strideBytes);
            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.5f * ImGui::GetStyle().Alpha);
            ImPlot::PlotShaded("##", timePoints, valuesForColor, count, 0, 0, 0, strideBytes);
            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
            if (ImGui::GetStyle().Alpha == 1.0f && ImPlot::IsPlotHovered()) {
                drawValuesAtMouseCursor(valuesForColor, timePoints, systemClock, count, startTime, endTime, upperBound, fracPartDecimals);
            }
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void StatisticsWindow::setPlotScale()
{
    if (_plotScale == PlotScale_Linear) {
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);
        return;
    }
    if (_plotScale == PlotScale_Logarithmic) {
        ImPlot::SetupAxisScale(
            ImAxis_Y1,
            [](double value, void* user_data) {
                return log(value * 1000 + 1.0) / log(2.0);
            },
            [](double value, void* user_data) {
                return (pow(2.0f, value) - 1.0) / 1000;
            });
        return;
    }
    THROW_NOT_IMPLEMENTED();
}

double StatisticsWindow::getUpperBound(double maxValue)
{
    if (_plotScale == PlotScale_Linear) {
        return maxValue * 1.5;
    }
    if (_plotScale == PlotScale_Logarithmic) {
        if (maxValue > 10) {
            return maxValue * pow(maxValue, 0.5);
        }
        return maxValue * 1.5;
    }
    THROW_NOT_IMPLEMENTED();
}

namespace
{
    std::string convertSystemClockToString(double systemClock)
    {
        auto time_t = static_cast<std::time_t>(systemClock);
        std::tm* tm = std::localtime(&time_t);

        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm);
        return std::string(buffer);

    }
}

void StatisticsWindow::drawValuesAtMouseCursor(
    double const* dataPoints,
    double const* timePoints,
    double const* systemClock,
    int count,
    double startTime,
    double endTime,
    double upperBound,
    int fracPartDecimals)
{
    auto constexpr stride = sizeof(DataPointCollection) / sizeof(double);

    auto mousePos = ImPlot::GetPlotMousePos();
    mousePos.x = std::max(startTime, std::min(endTime, mousePos.x));
    mousePos.y = dataPoints[0];

    auto dateTimeString =
        [&] {
        if (systemClock == nullptr) {
            for (int i = 1; i < count; ++i) {
                if (timePoints[i * stride] > mousePos.x) {
                    mousePos.y = dataPoints[i * stride];
                    break;
                }
            }
            return std::string();
        }
        auto systemClockEntry = systemClock[0];
        for (int i = 1; i < count; ++i) {
            if (timePoints[i * stride] > mousePos.x) {
                mousePos.y = dataPoints[i * stride];
                systemClockEntry = systemClock[i * stride];
                break;
            }
        }
        mousePos.y = std::max(0.0, std::min(upperBound, mousePos.y));

        return systemClockEntry != 0 ? convertSystemClockToString(systemClockEntry) : std::string("-");
    }();

    ImPlot::PushStyleColor(ImPlotCol_InlayText, ImColor::HSV(0.0f, 0.0f, 1.0f).Value);
    ImPlot::PlotText(ICON_FA_GENDERLESS, mousePos.x, mousePos.y, {scale(1.0f), scale(2.0f)});
    ImPlot::PopStyleColor();

    ImPlot::PushStyleColor(ImPlotCol_Line, ImColor::HSV(0.0f, 0.0f, 1.0f).Value);
    ImPlot::PlotInfLines("", &mousePos.x, 1);
    ImPlot::PopStyleColor();

    char label[256];
    auto leftSideFactor = mousePos.x > (startTime + endTime) / 2 ? -1.0f : 1.0f;
    if (!dateTimeString.empty()) {
        snprintf(
            label,
            sizeof(label),
            "Time step: %s\nTimestamp: %s\nValue: %s",
            StringHelper::format(mousePos.x, 0).c_str(),
            dateTimeString.c_str(),
            StringHelper::format(mousePos.y, fracPartDecimals).c_str());
    } else {
        snprintf(
            label,
            sizeof(label),
            "Relative time: %s\nValue: %s",
            StringHelper::format(mousePos.x, 0).c_str(),
            StringHelper::format(mousePos.y, fracPartDecimals).c_str());
    }
    ImPlot::PlotText(label, mousePos.x, upperBound,  {leftSideFactor * (scale(5.0f) + ImGui::CalcTextSize(label).x / 2), scale(28.0f)});
}

void StatisticsWindow::validationAndCorrection()
{
    _timeHorizonForLiveStatistics = std::max(1.0f, std::min(TimelineLiveStatistics::MaxLiveHistory, _timeHorizonForLiveStatistics));
    _timeHorizonForLongtermStatistics = std::max(1.0f, std::min(100.0f, _timeHorizonForLongtermStatistics));
}

float StatisticsWindow::calcPlotHeight(int row) const
{
    auto isCollapsed = _collapsedPlotIndices.contains(row);
    return isCollapsed ? 25.0f : _plotHeight;
}
