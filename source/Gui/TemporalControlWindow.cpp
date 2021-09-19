#include "TemporalControlWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "Base/StringFormatter.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineImpl/SimulationController.h"

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

    processRunButton();
    ImGui::SameLine();
    processPauseButton();
    ImGui::SameLine();
    processStepBackwardButton();
    ImGui::SameLine();
    processStepForwardButton();
    ImGui::SameLine();
    processSnapshotButton();
    ImGui::SameLine();
    processRestoreButton();

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    processTpsInfo();
    processTotalTimestepsInfo();

    ImGui::Spacing();
    ImGui::Spacing();
    processTpsRestriction();

    ImGui::End();
}

void _TemporalControlWindow::processTpsInfo()
{
    ImGui::Text("Time steps per second");

    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::Text(StringFormatter::format(_simController->getTps()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::Text("Total time steps");

    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::Text(StringFormatter::format(_simController->getCurrentTimestep()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTpsRestriction()
{
    static int slowDown = 0;
    ImGui::CheckboxFlags("Slow down", &slowDown, ImGuiSliderFlags_AlwaysClamp);
    ImGui::SameLine();
    static int tpsRestriction = 30;
    ImGui::BeginDisabled(!slowDown);
    ImGui::SliderInt("", &tpsRestriction, 1, 200, "%d TPS");
    ImGui::EndDisabled();
    if (slowDown) {
        _simController->setTpsRestriction(tpsRestriction);
    } else {
        _simController->setTpsRestriction(boost::none);
    }
}

void _TemporalControlWindow::processRunButton()
{
    ImGui::BeginDisabled(_simController->isSimulationRunning());
    if (ImGui::ImageButton((void*)(intptr_t)_runTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _history.clear();
        _simController->runSimulation();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processPauseButton()
{
    ImGui::BeginDisabled(!_simController->isSimulationRunning());
    if (ImGui::ImageButton((void*)(intptr_t)_pauseTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _simController->pauseSimulation();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepBackwardButton()
{
    ImGui::BeginDisabled(_history.empty() || _simController->isSimulationRunning());
    if (ImGui::ImageButton((void*)(intptr_t)_stepBackwardTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        auto const& snapshot = _history.back();
        _simController->clear();
        _simController->setCurrentTimestep(snapshot.timestep);
        _simController->updateData(snapshot.data);
        _history.pop_back();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processStepForwardButton()
{
    ImGui::BeginDisabled(_simController->isSimulationRunning());
    if (ImGui::ImageButton((void*)(intptr_t)_stepForwardTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        Snapshot newSnapshot;
        newSnapshot.timestep = _simController->getCurrentTimestep();
        auto size = _simController->getWorldSize();
        newSnapshot.data = _simController->getSimulationData({0, 0}, size);
        _history.emplace_back(newSnapshot);

        _simController->calcSingleTimestep();
    }
    ImGui::EndDisabled();
}

void _TemporalControlWindow::processSnapshotButton()
{
    if (ImGui::ImageButton((void*)(intptr_t)_snapshotTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        Snapshot newSnapshot;
        newSnapshot.timestep = _simController->getCurrentTimestep();
        auto size = _simController->getWorldSize();
        newSnapshot.data = _simController->getSimulationData({0, 0}, size);
        _snapshot = newSnapshot;
    }
}

void _TemporalControlWindow::processRestoreButton()
{
    ImGui::BeginDisabled(!_snapshot);
    if (ImGui::ImageButton((void*)(intptr_t)_restoreTexture.textureId, {32.0f, 32.0f}, {0, 0}, {1.0f, 1.0f})) {
        _simController->clear();
        _simController->setCurrentTimestep(_snapshot->timestep);
        _simController->updateData(_snapshot->data);
    }
    ImGui::EndDisabled();
}
