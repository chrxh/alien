#include "EvolutionDashboardWindow.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <random>
#include <cstdio>

#include <imgui.h>
#include <implot.h>

#include <Base/GlobalSettings.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr SparklineWidth = 90.0f;
    auto constexpr SparklineHeight = 26.0f;
    auto constexpr ColorChipSize = 22.0f;
    auto constexpr SwatchSize = 11.0f;
    auto constexpr SwatchGap = 3.0f;
    auto constexpr PlotLabelColumnWidth = 150.0f;
    auto constexpr PlotHeight = 60.0f;
    auto constexpr PlotHeightWithAxis = 80.0f;
    auto constexpr DefaultTimelinesHeight = 380.0f;
    auto constexpr TimeSpan = 1250000.0;

    ImColor const CardBackgroundColor = ImColor(0.095f, 0.117f, 0.165f, 1.0f);
    ImColor const CardBorderColor = ImColor(0.165f, 0.196f, 0.270f, 1.0f);
    ImColor const PositiveDeltaColor = ImColor(0.19f, 0.77f, 0.55f, 1.0f);
    ImColor const NegativeDeltaColor = ImColor(0.94f, 0.32f, 0.32f, 1.0f);
    ImColor const SumSeriesColor = ImColor(0.78f, 0.78f, 0.78f, 1.0f);

    struct MetricDef
    {
        char const* tableHeader;
        char const* plotName;
        double baseValue;
        int decimals;  //-1: scientific notation
    };

    MetricDef const Metrics[EvolutionDashboardWindow::NumMetrics] = {
        {"Creatures", "Creatures", 1500.0, 0},
        {"Avg size", "Avg creature size", 15.0, 1},
        {"Avg cells", "Avg cells", 40.0, 1},
        {"Avg nodes", "Avg nodes", 60.0, 1},
        {"Sum energy", "Sum energy", 150000.0, 0},
        {"Avg mut. rate", "Avg mutation rate", 2.0e-4, -1},
        {"Avg age (gen)", "Avg age (generations)", 200.0, 0},
        {"Created /s", "Created creatures / s", 3.0, 1},
        {"Mutations /s", "Mutations / s", 1.0, 1},
    };

    float const MetricDeltas[EvolutionDashboardWindow::NumMetrics] = {4.1f, 0.6f, 1.2f, 0.0f, -0.4f, -2.3f, 5.0f, 1.1f, -0.8f};

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

    std::vector<double> createRandomWalk(std::mt19937& rng, int count, double baseValue, double volatility, double trend)
    {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::vector<double> result;
        result.reserve(count);
        auto value = baseValue * (0.4 + 0.4 * (distribution(rng) + 1.0) / 2);
        for (int i = 0; i < count; ++i) {
            value += baseValue * (distribution(rng) * volatility + trend);
            value = std::max(baseValue * 0.05, value);
            result.emplace_back(value);
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
    _plotScale = GlobalSettings::get().getValue("windows.evolution dashboard.plot scale", _plotScale);
    _lastSteps = GlobalSettings::get().getValue("windows.evolution dashboard.last steps", _lastSteps);
    _colorFilter = GlobalSettings::get().getValue("windows.evolution dashboard.color filter", _colorFilter);
    validateAndCorrect();

    generateDummyData();
}

void EvolutionDashboardWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("windows.evolution dashboard.timelines height", _timelinesHeight);
    GlobalSettings::get().setValue("windows.evolution dashboard.timeline mode", _timelineMode);
    GlobalSettings::get().setValue("windows.evolution dashboard.plot scale", _plotScale);
    GlobalSettings::get().setValue("windows.evolution dashboard.last steps", _lastSteps);
    GlobalSettings::get().setValue("windows.evolution dashboard.color filter", _colorFilter);
}

void EvolutionDashboardWindow::processIntern()
{
    updateCellColors();
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

void EvolutionDashboardWindow::processHeader()
{
    auto const& style = ImGui::GetStyle();
    auto cardWidth = (ImGui::GetContentRegionAvail().x - 4 * style.ItemSpacing.x) / 6;
    auto cardHeight =
        style.WindowPadding.y * 2 + ImGui::GetTextLineHeight() * 3 + StyleRepository::get().getLargeFont()->FontSize + style.ItemSpacing.y * 3 + scale(6.0f);

    processEntitiesCard(cardWidth * 2, cardHeight);
    ImGui::SameLine();
    processKpiCard("SUM ENERGY", "1.24 M", 0.8f, 0, cardWidth, cardHeight);
    ImGui::SameLine();
    processKpiCard("EXTERNAL ENERGY", "356 K", -1.2f, 1, cardWidth, cardHeight);
    ImGui::SameLine();
    processKpiCard("LINEAGES", StringHelper::format(toFloat(_lineages.size()), 0), 2.4f, 2, cardWidth, cardHeight);
    ImGui::SameLine();
    processKpiCard("CREATURES", formatMetricValue(_allLineages.currentValues.at(0), 0), 0.5f, 3, cardWidth, cardHeight);
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
        ImGui::SetCursorPos({width - scale(SparklineWidth + 12.0f), height - scale(SparklineHeight + 12.0f)});
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
        ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0, 0, 0, 0));
        ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0, 0, 0, 0));
        ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0, 0, 0, 0));
        if (ImPlot::BeginPlot(("##spark" + label).c_str(), {scale(SparklineWidth), scale(SparklineHeight)}, ImPlotFlags_CanvasOnly | ImPlotFlags_NoInputs)) {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
            auto [minIt, maxIt] = std::minmax_element(sparkline.begin(), sparkline.end());
            ImPlot::SetupAxesLimits(0, toDouble(sparkline.size() - 1), *minIt * 0.9, *maxIt * 1.1, ImGuiCond_Always);
            ImPlot::PushStyleColor(ImPlotCol_Line, delta >= 0 ? (ImU32)PositiveDeltaColor : (ImU32)NegativeDeltaColor);
            ImPlot::PlotLine("##", sparkline.data(), toInt(sparkline.size()));
            ImPlot::PopStyleColor();
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleColor(3);
        ImPlot::PopStyleVar();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
}

void EvolutionDashboardWindow::processEntitiesCard(float width, float height)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, scale(6.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, (ImU32)CardBackgroundColor);
    ImGui::PushStyleColor(ImGuiCol_Border, (ImU32)CardBorderColor);
    if (ImGui::BeginChild("##cardEntities", {width, height}, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TextDecentColor);
        AlienGui::Text("ENTITIES");
        ImGui::PopStyleColor();

        ImGui::PushFont(StyleRepository::get().getLargeFont());
        AlienGui::Text("486,120");
        ImGui::PopFont();

        std::pair<char const*, char const*> const subValues[] = {
            {"Solids", "12,410"},
            {"Fluid particles", "213,880"},
            {"Cells", "245,230"},
            {"Energy particles", "14,600"},
        };
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
            ImGui::PushID(lineage.id);
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
        for (auto const& lineage : _lineages) {
            if (!_selectedLineageIds.contains(lineage.id)) {
                continue;
            }
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
            snprintf(deltaText, sizeof(deltaText), "%+.1f %%", MetricDeltas[i]);
            ImGui::PushStyleColor(ImGuiCol_Text, MetricDeltas[i] >= 0 ? (ImU32)PositiveDeltaColor : (ImU32)NegativeDeltaColor);
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
    auto count = toInt(_timePoints.size());
    auto offset = 0;
    if (_timelineMode == TimelineMode_RealTime) {
        offset = count * 9 / 10;
    } else if (_timelineMode == TimelineMode_LastSteps) {
        auto numPoints = toInt(std::lround(toDouble(_lastSteps) / TimeSpan * count));
        offset = std::max(0, count - std::max(2, numPoints));
    }
    count -= offset;

    std::vector<DummyLineage const*> plottedLineages;
    if (_selectedLineageIds.empty()) {
        plottedLineages.emplace_back(&_allLineages);
    } else {
        for (auto const& lineage : _lineages) {
            if (_selectedLineageIds.contains(lineage.id)) {
                plottedLineages.emplace_back(&lineage);
            }
        }
    }

    auto upperBound = 0.0;
    for (auto const* lineage : plottedLineages) {
        auto const& series = lineage->series.at(metricIndex);
        for (int i = offset; i < offset + count; ++i) {
            upperBound = std::max(upperBound, series.at(i));
        }
    }
    upperBound *= 1.35;

    ImGui::PushID(metricIndex);
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBg, (ImU32)ImColor(0.0f, 0.0f, 0.0f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleColor(ImPlotCol_PlotBorder, (ImU32)ImColor(0.3f, 0.3f, 0.3f, ImGui::GetStyle().Alpha));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextAxesLimits(_timePoints.at(offset), _timePoints.at(offset + count - 1), 0, upperBound, ImGuiCond_Always);

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
            ImGui::PushID(lineage->id + 1);
            auto color = lineage->id == -1 ? SumSeriesColor : toImColor(_cellColors.at(getFirstColor(lineage->colorBitset)));
            auto const& series = lineage->series.at(metricIndex);
            ImPlot::PushStyleColor(ImPlotCol_Line, (ImU32)color);
            ImPlot::PlotLine("##line", _timePoints.data() + offset, series.data() + offset, count);
            if (_plotScale == PlotScale_Linear) {
                ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.15f * ImGui::GetStyle().Alpha);
                ImPlot::PlotShaded("##shaded", _timePoints.data() + offset, series.data() + offset, count);
                ImPlot::PopStyleVar();
            }
            ImPlot::PopStyleColor();
            if (ImGui::GetStyle().Alpha == 1.0f) {
                ImPlot::Annotation(
                    _timePoints.at(offset + count - 1),
                    series.at(offset + count - 1),
                    color,
                    ImVec2(-10.0f, 10.0f),
                    true,
                    "%s",
                    formatMetricValue(series.at(offset + count - 1), Metrics[metricIndex].decimals).c_str());
            }
            ImGui::PopID();
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor(3);
    ImGui::PopID();
}

void EvolutionDashboardWindow::generateDummyData()
{
    std::mt19937 rng(20260711);
    std::uniform_int_distribution<int> colorDistribution(0, MAX_COLORS - 1);
    std::uniform_int_distribution<int> numColorsDistribution(1, 3);
    std::uniform_real_distribution<double> trendDistribution(-0.002, 0.004);

    _timePoints.clear();
    for (int i = 0; i < NumTimePoints; ++i) {
        _timePoints.emplace_back(TimeSpan * i / (NumTimePoints - 1));
    }

    _lineages.clear();
    int const lineageIds[] = {3, 7, 12, 18, 21, 24, 29, 33, 38, 42, 47, 51};
    for (auto id : lineageIds) {
        DummyLineage lineage;
        lineage.id = id;
        auto numColors = numColorsDistribution(rng);
        while (std::popcount(static_cast<unsigned>(lineage.colorBitset)) < numColors) {
            lineage.colorBitset |= 1 << colorDistribution(rng);
        }
        for (int i = 0; i < NumMetrics; ++i) {
            lineage.series.at(i) = createRandomWalk(rng, NumTimePoints, Metrics[i].baseValue, 0.015, trendDistribution(rng));
            lineage.currentValues.at(i) = lineage.series.at(i).back();
        }
        _lineages.emplace_back(lineage);
    }

    _allLineages.id = -1;
    for (int i = 0; i < NumMetrics; ++i) {
        auto& sumSeries = _allLineages.series.at(i);
        sumSeries.assign(NumTimePoints, 0.0);
        auto isSummable = i == 0 || i == 4 || i == 7 || i == 8;  //creatures, sum energy, created/s, mutations/s
        for (auto const& lineage : _lineages) {
            for (int j = 0; j < NumTimePoints; ++j) {
                sumSeries.at(j) += lineage.series.at(i).at(j);
            }
        }
        if (!isSummable) {
            for (auto& value : sumSeries) {
                value /= toDouble(_lineages.size());
            }
        }
        _allLineages.currentValues.at(i) = sumSeries.back();
    }

    for (auto& sparkline : _headerSparklines) {
        sparkline = createRandomWalk(rng, 40, 1.0, 0.03, trendDistribution(rng));
    }
}

void EvolutionDashboardWindow::validateAndCorrect()
{
    _timelinesHeight = std::max(scale(100.0f), _timelinesHeight);
    _timelineMode = std::clamp(_timelineMode, static_cast<TimelineMode>(TimelineMode_RealTime), static_cast<TimelineMode>(TimelineMode_LastSteps));
    _plotScale = std::clamp(_plotScale, static_cast<PlotScale>(PlotScale_Linear), static_cast<PlotScale>(PlotScale_Logarithmic));
    _lastSteps = std::clamp(_lastSteps, 1000, toInt(TimeSpan));
    _colorFilter &= (1 << MAX_COLORS) - 1;
    if (_colorFilter == 0) {
        _colorFilter = (1 << MAX_COLORS) - 1;
    }
}
