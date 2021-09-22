#include "ModeWindow.h"

#include "imgui.h"

#include "OpenGLHelper.h"

_ModeWindow::_ModeWindow()
{
    _navigationOn = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\navigation on.png");
    _navigationOff = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\navigation off.png");
    _actionOn = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\action on.png");
    _actionOff = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\action off.png");
}

void _ModeWindow::process()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x /*+ viewport->Size.x - 200*/ + 00, viewport->Pos.y /*+ viewport->Size.y - 100*/+ 20));
    ImGui::SetNextWindowSize(ImVec2(130, 70));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("TOOLBAR", NULL, windowFlags);

    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor());
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor());
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor());

    auto navTexture = Mode::Navigation == _mode ? _navigationOn.textureId : _navigationOff.textureId;
    if (ImGui::ImageButton((void*)(intptr_t)navTexture, {48.0f, 48.0f}, {0, 0}, {1.0f, 1.0f})) {
        _mode = Mode::Navigation;
    }
    ImGui::SameLine();
    auto actionTexture = Mode::Action == _mode ? _actionOn.textureId : _actionOff.textureId;
    if (ImGui::ImageButton((void*)(intptr_t)actionTexture, {48.0f, 48.0f}, {0, 0}, {1.0f, 1.0f})) {
        _mode = Mode::Action;
    }

    ImGui::PopStyleColor(3);

    /*
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PopStyleVar();
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    ImGui::SameLine();
    ImGui::Button("Zoom in", ImVec2(0, 37));
*/

    ImGui::End();
}
