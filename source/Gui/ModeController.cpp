#include "ModeController.h"

#include <imgui.h>

#include "Base/Resources.h"

#include "OpenGLHelper.h"
#include "EditorController.h"
#include "StyleRepository.h"

_ModeController::_ModeController(EditorController const& editorController)
    : _editorController(editorController)
{
    _editorOn = OpenGLHelper::loadTexture(Const::EditorOnFilename);
    _editorOff = OpenGLHelper::loadTexture(Const::EditorOffFilename);
}

void _ModeController::process()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y + viewport->Size.y - contentScale(120.0f)));
    ImGui::SetNextWindowSize(ImVec2(contentScale(160.0f), contentScale(100.0f)));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("TOOLBAR", NULL, windowFlags);

    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor());
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor());
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor());

    auto actionTexture = Mode::Editor == _mode ? _editorOn.textureId : _editorOff.textureId;
    if (ImGui::ImageButton((void*)(intptr_t)actionTexture, {contentScale(80.0f), contentScale(80.0f)}, {0, 0}, {1.0f, 1.0f})) {
        _mode = _mode == Mode::Editor ? Mode::Navigation : Mode::Editor;
        _editorController->setOn(!_editorController->isOn());
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
    _editorController->setOn(_mode == Mode::Editor);
}
