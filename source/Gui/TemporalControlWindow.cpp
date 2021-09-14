#include "TemporalControlWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "EngineImpl/SimulationController.h"

#include "Style.h"
#include "StyleRepository.h"
#include "OpenGLHelper.h"

_TemporalControlWindow::_TemporalControlWindow(
    SimulationController const& simController,
    StyleRepository const& styleRepository)
    : _simController(simController)
    , _styleRepository(styleRepository)
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
    ImGui::Text(std::to_string(_simController->getTps()).c_str());
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
    static int tpsRestriction = 30;
    if (!slowDown) {
        ImGui::BeginDisabled();
    }
    ImGui::SliderInt("", &tpsRestriction, 1, 200, "%d TPS");
    if (!slowDown) {
        ImGui::EndDisabled();
    }
    if (slowDown) {
        _simController->setTpsRestriction(tpsRestriction);
    } else {
        _simController->setTpsRestriction(boost::none);
    }
    ImGui::Spacing();
    ImGui::Spacing();

    auto isRunning = _simController->isSimulationRunning();
    if (isRunning) {
        ImGui::BeginDisabled();
    }
    if (ImGui::ImageButton((void*)(intptr_t)_runTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _simController->runSimulation();
    }
    if (isRunning) {
        ImGui::EndDisabled(); 
    }

    if (!isRunning) {
        ImGui::BeginDisabled();
    }
    ImGui::SameLine();
    if (ImGui::ImageButton((void*)(intptr_t)_pauseTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _simController->pauseSimulation();
    }
    if (!isRunning) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    ImGui::ImageButton((void*)(intptr_t)_stepBackwardTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton((void*)(intptr_t)_stepForwardTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton((void*)(intptr_t)_snapshotTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f});

    ImGui::SameLine();
    ImGui::ImageButton((void*)(intptr_t)_restoreTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f});

    ImGui::End();
}
