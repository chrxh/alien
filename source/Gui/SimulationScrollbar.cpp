#include "SimulationScrollbar.h"

#include <algorithm>

#include <glad/glad.h>
#include "imgui.h"

#include "EngineImpl/SimulationController.h"

_SimulationScrollbar::_SimulationScrollbar(
    std::string const& id,
    Orientation orientation,
    SimulationController const& simController)
    : _id(id)
    , _orientation(orientation)
    , _simController(simController)
{}

void _SimulationScrollbar::setVisibleWorldSection(float startWorldPos, float endWorldPos)
{
    _startWorldPos = startWorldPos;
    _endWorldPos = endWorldPos;
}

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
    auto sliderBarStartPos = std::min(std::max(_startWorldPos / worldSize * size, 0.0f), size);
    auto sliderBarEndPos = std::min(std::max(_endWorldPos / worldSize * size, 0.0f), size);
    if (sliderBarEndPos < sliderBarStartPos) {
        sliderBarEndPos = sliderBarStartPos;
    }
    auto sliderBarPos =
        Orientation::Horizontal == _orientation ? ImVec2{sliderBarStartPos, 4} : ImVec2{4, sliderBarStartPos};
    ImGui::SetCursorPos(sliderBarPos);
    auto sliderBarSize = Orientation::Horizontal == _orientation ? ImVec2{sliderBarEndPos - sliderBarStartPos, 10}
                                                                 : ImVec2{10, sliderBarEndPos - sliderBarStartPos};
    ImGui::BeginChild((_id + "child").c_str(), sliderBarSize);
    ImGui::EndChild();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    
    ImGui::End();
    ImGui::PopStyleVar();
}
