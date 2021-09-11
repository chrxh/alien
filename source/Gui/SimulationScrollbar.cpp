#include "SimulationScrollbar.h"

#include <algorithm>

#include <glad/glad.h>
#include "imgui.h"

#include "EngineImpl/SimulationController.h"
#include "Viewport.h"

_SimulationScrollbar::_SimulationScrollbar(
    std::string const& id,
    Orientation orientation,
    SimulationController const& simController,
    Viewport const& viewport)
    : _id(id)
    , _orientation(orientation)
    , _simController(simController)
    , _viewport(viewport)
{}

void _SimulationScrollbar::draw(RealVector2D const& topLeft, RealVector2D const& size2d)
{
    auto worldSize =
        Orientation::Horizontal == _orientation ? _simController->getWorldSize().x : _simController->getWorldSize().y;
    auto size = Orientation::Horizontal == _orientation ? size2d.x : size2d.y;

    ImGui::SetNextWindowPos(ImVec2(topLeft.x, topLeft.y));
    ImGui::SetNextWindowSize(ImVec2(size2d.x, size2d.y));
    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDecoration;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin(_id.c_str(), NULL, windowFlags);
    
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.3f, 0.3f, 0.3f, 1.0f});
    auto worldRect = _viewport->getVisibleWorldRect();
    auto startWorldPos = Orientation::Horizontal == _orientation ? worldRect.topLeft.x : worldRect.topLeft.y;
    auto endWorldPos = Orientation::Horizontal == _orientation ? worldRect.bottomRight.x : worldRect.bottomRight.y;

    auto sliderBarStartPos = std::min(std::max(startWorldPos / worldSize * size, 0.0f), size);
    auto sliderBarEndPos = std::min(std::max(endWorldPos / worldSize * size, 0.0f), size);
    if (sliderBarEndPos < sliderBarStartPos) {
        sliderBarEndPos = sliderBarStartPos;
    }
    auto sliderBarPos =
        Orientation::Horizontal == _orientation ? ImVec2{4 + sliderBarStartPos, 4} : ImVec2{4, 4 + sliderBarStartPos};
    ImGui::SetCursorPos(sliderBarPos);
    auto sliderBarSize = Orientation::Horizontal == _orientation ? ImVec2{sliderBarEndPos - sliderBarStartPos - 8, 10}
                                                                 : ImVec2{10, sliderBarEndPos - sliderBarStartPos - 8};
    ImGui::BeginChild((_id + "child").c_str(), sliderBarSize);
    ImGui::EndChild();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    
    ImGui::End();
    ImGui::PopStyleVar();
}
