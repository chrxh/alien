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

void _SimulationScrollbar::processEvents()
{
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
    }
}

void _SimulationScrollbar::draw(RealVector2D const& topLeft, RealVector2D const& size2d)
{
    auto worldSize =
        Orientation::Horizontal == _orientation ? _simController->getWorldSize().x : _simController->getWorldSize().y;
    auto size = Orientation::Horizontal == _orientation ? size2d.x : size2d.y;
   
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
    auto sliderBarSize = Orientation::Horizontal == _orientation ? ImVec2{sliderBarEndPos - sliderBarStartPos - 8, 10}
                                                                 : ImVec2{10, sliderBarEndPos - sliderBarStartPos - 8};

    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDecoration;

    ImGui::SetNextWindowPos(ImVec2(topLeft.x, topLeft.y));
    ImGui::SetNextWindowSize(ImVec2(size2d.x, size2d.y));
    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::Begin(_id.c_str(), NULL, windowFlags);

    ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
    ImColor sliderColor;
    if (mousePositionAbsolute.x >= topLeft.x + sliderBarPos.x - 3
        && mousePositionAbsolute.x <= topLeft.x + sliderBarPos.x + sliderBarSize.x + 3
        && mousePositionAbsolute.y >= topLeft.y + sliderBarPos.y - 3
        && mousePositionAbsolute.y <= topLeft.y + sliderBarPos.y + sliderBarSize.y + 3) {
        sliderColor = ImColor{0.6f, 0.6f, 0.6f, 1.0f};
    } else {
        sliderColor = ImColor{0.3f, 0.3f, 0.3f, 1.0f};
    }

    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(topLeft.x + sliderBarPos.x, topLeft.y + sliderBarPos.y),
        ImVec2(topLeft.x + sliderBarPos.x + sliderBarSize.x, topLeft.y + sliderBarPos.y + sliderBarSize.y),
        sliderColor,
        5.0f);

    ImGui::End();
    ImGui::PopStyleVar();

/*
    ImGui::SetNextWindowPos(ImVec2(/ *topLeft.x, topLeft.y* /0, 150));
//    printf("%f, %f\n", topLeft.x, topLeft.y);
    ImGui::SetNextWindowSize(ImVec2(size2d.x, size2d.y));
//    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGui::Begin((_id + "1").c_str(), NULL, windowFlags);
    ImGui::GetWindowDrawList()->AddRectFilled({0, 0}, {200, 200}, IM_COL32(255, 0, 255, 255));
    ImGui::End();
*/
}
