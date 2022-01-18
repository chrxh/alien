#include "MultiplierWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "AlienImGui.h"

namespace
{
    auto const ModeText = std::unordered_map<MultiplierMode, std::string>{
        {MultiplierMode::Grid, "Grid multiplier"},
        {MultiplierMode::Random, "Random multiplier"},
    };
}

_MultiplierWindow::_MultiplierWindow(SimulationController const& simController, Viewport const& viewport)
    : _AlienWindow("Multiplier", "editor.multiplier", false)
    , _simController(simController)
    , _viewport(viewport)
{}

void _MultiplierWindow::processIntern()
{
    if (AlienImGui::BeginToolbarButton(ICON_GRID)) {
        _mode = MultiplierMode::Grid;
    }
    AlienImGui::EndToolbarButton();

    ImGui::SameLine();
    if (AlienImGui::BeginToolbarButton(ICON_RANDOM)) {
        _mode = MultiplierMode::Random;
    }
    AlienImGui::EndToolbarButton();

    AlienImGui::Group(ModeText.at(_mode));
}

