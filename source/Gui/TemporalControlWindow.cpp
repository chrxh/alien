#include "TemporalControlWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"

#include "Style.h"
#include "StyleRepository.h"
#include "OpenGLHelper.h"

_TemporalControlWindow::_TemporalControlWindow(StyleRepository const& styleRepository)
    : _styleRepository(styleRepository)
{
    _runTexture = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\run.png");
    _pauseTexture = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\pause.png");
    _stepBackwardTexture = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\step backward.png");
    _stepForwardTexture = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\step forward.png");
    _snapshotTexture = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\snapshot.png");
    _restoreTexture = OpenGLHelper::loadTexture("d:\\temp\\alien-imgui\\source\\Gui\\Resources\\restore.png");
}

void _TemporalControlWindow::process()
{
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("Temporal control");

    ImGui::Text("Time steps per second");

    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, 0xff909090);
    ImGui::Text("100");
    ImGui::PopFont();
    ImGui::PopStyleColor();

    ImGui::Text("Total time steps");
    
    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, 0xff909090);
    ImGui::Text("112,323");
    ImGui::PopFont();
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Spacing();
    static int slowDown = false;
    ImGui::CheckboxFlags("Slow down", &slowDown, ImGuiSliderFlags_AlwaysClamp);
    ImGui::SameLine();
    static int restrictTps = 30;
    if (!slowDown) {
        ImGui::BeginDisabled();
    }
    ImGui::SliderInt("", &restrictTps, 0, 100, "%d TPS");
    if (!slowDown) {
        ImGui::EndDisabled();
    }
    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::ImageButton(
        (void*)(intptr_t)_runTexture.textureId,
        {toFloat(_runTexture.width), toFloat(_runTexture.height)},
        {0, 0},
        {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton(
        (void*)(intptr_t)_pauseTexture.textureId,
        {toFloat(_pauseTexture.width), toFloat(_pauseTexture.height)},
        {0, 0},
        {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton(
        (void*)(intptr_t)_stepBackwardTexture.textureId,
        {toFloat(_stepBackwardTexture.width), toFloat(_stepBackwardTexture.height)},
        {0, 0},
        {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton(
        (void*)(intptr_t)_stepForwardTexture.textureId,
        {toFloat(_stepForwardTexture.width), toFloat(_stepForwardTexture.height)},
        {0, 0},
        {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton(
        (void*)(intptr_t)_snapshotTexture.textureId,
        {toFloat(_snapshotTexture.width), toFloat(_snapshotTexture.height)},
        {0, 0},
        {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton(
        (void*)(intptr_t)_restoreTexture.textureId,
        {toFloat(_restoreTexture.width), toFloat(_restoreTexture.height)},
        {0, 0},
        {1.0f, 1.0f});

    ImGui::End();
}
