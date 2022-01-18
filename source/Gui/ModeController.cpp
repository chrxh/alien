#include "ModeController.h"

#include <imgui.h>

#include "OpenGLHelper.h"
#include "Resources.h"
#include "EditorController.h"

_ModeController::_ModeController(EditorController const& editorController)
    : _editorController(editorController)
{
    _navigationOn = OpenGLHelper::loadTexture(Const::NavigationOnFilename);
    _navigationOff = OpenGLHelper::loadTexture(Const::NavigationOffFilename);
    _actionOn = OpenGLHelper::loadTexture(Const::ActionOnFilename);
    _actionOff = OpenGLHelper::loadTexture(Const::ActionOffFilename);
}

void _ModeController::process()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x /*+ viewport->Size.x - 200*/ + 00, viewport->Pos.y + viewport->Size.y - 100));
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
        _editorController->setOn(false);
    }
    ImGui::SameLine();
    auto actionTexture = Mode::Action == _mode ? _actionOn.textureId : _actionOff.textureId;
    if (ImGui::ImageButton((void*)(intptr_t)actionTexture, {48.0f, 48.0f}, {0, 0}, {1.0f, 1.0f})) {
        _mode = Mode::Action;
        _editorController->setOn(true);
    }

    ImGui::PopStyleColor(3);
    ImGui::End();
}

auto _ModeController::getMode() const -> Mode
{
    return _mode;
}

void _ModeController::setMode(Mode value)
{
    _mode = value;
    _editorController->setOn(_mode == Mode::Action);
}
