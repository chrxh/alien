#include "GenomeEditorWindow.h"

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "AlienImGui.h"

namespace
{
    auto const MaxContentTextWidth = 260.0f;
}

_GenomeEditorWindow ::_GenomeEditorWindow()
    : _AlienWindow("Genome editor", "windows.genome editor", false)
{
    _tabDatas = {TabData()};
}

_GenomeEditorWindow::~_GenomeEditorWindow()
{
}

void _GenomeEditorWindow::processIntern()
{
    if (AlienImGui::ToolbarButton(ICON_FA_PLUS)) {
    }
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_MINUS)) {
    }
    AlienImGui::Separator();

    if (ImGui::BeginTabBar("##Flow", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {

        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            _tabDatas.emplace_back(TabData());
        }
        AlienImGui::Tooltip("New gene");

        for (auto const& [index, tabData] : _tabDatas | boost::adaptors::indexed(1)) {
            if (ImGui::BeginTabItem(("Gene " + std::to_string(index)).c_str(), NULL, ImGuiTabItemFlags_None)) {
                ImGui::EndTabItem();
            }
        }

        ImGui::EndTabBar();
    }
}

