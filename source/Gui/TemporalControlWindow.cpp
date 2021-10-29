#include "TemporalControlWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "Base/StringFormatter.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineImpl/SimulationController.h"

#include "StyleRepository.h"
#include "OpenGLHelper.h"
#include "Resources.h"
#include "StatisticsWindow.h"
#include "GlobalSettings.h"

_TemporalControlWindow::_TemporalControlWindow(
    SimulationController const& simController,
    StyleRepository const& styleRepository,
    StatisticsWindow const& statisticsWindow)
    : _simController(simController)
    , _styleRepository(styleRepository)
    , _statisticsWindow(statisticsWindow)
{
    _runTexture = OpenGLHelper::loadTexture(Const::RunFilename);
    _pauseTexture = OpenGLHelper::loadTexture(Const::PauseFilename);
    _stepBackwardTexture = OpenGLHelper::loadTexture(Const::StepBackwardFilename);
    _stepForwardTexture = OpenGLHelper::loadTexture(Const::StepForwardFilename);
    _snapshotTexture = OpenGLHelper::loadTexture(Const::SnapshotFilename);
    _restoreTexture = OpenGLHelper::loadTexture(Const::RestoreFilname);

    _on = GlobalSettings::getInstance().getBoolState("windows.temporal control.active", true);
}

_TemporalControlWindow::~_TemporalControlWindow()
{
    GlobalSettings::getInstance().setBoolState("windows.temporal control.active", _on);
}

void _TemporalControlWindow::process()
{
    if (!_on) {
        return;
    }

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::Begin("Temporal control", &_on);

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

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
        processTpsInfo();
        processTotalTimestepsInfo();

        ImGui::Spacing();
        ImGui::Spacing();
        processTpsRestriction();
        ImGui::EndChild();
    }

    ImGui::End();
}

bool _TemporalControlWindow::isOn() const
{
    return _on;
}

void _TemporalControlWindow::setOn(bool value)
{
    _on = value;
}

void _TemporalControlWindow::processTpsInfo()
{
    ImGui::Text("Time steps per second");

    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor /*0xffa07050*/);
    ImGui::Text(StringFormatter::format(_simController->getTps(), 1).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTotalTimestepsInfo()
{
    ImGui::Text("Total time steps");

    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor/*0xffa07050*/);
    ImGui::Text(StringFormatter::format(_simController->getCurrentTimestep()).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void _TemporalControlWindow::processTpsRestriction()
{
    ImGui::Checkbox("Slow down", &_slowDown);
    ImGui::SameLine();
    static int tpsRestriction = 30;
    ImGui::BeginDisabled(!_slowDown);
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::SliderInt("", &tpsRestriction, 1, 400, "%d TPS", ImGuiSliderFlags_Logarithmic);
    if (_slowDown) {
        _simController->setTpsRestriction(tpsRestriction);
    } else {
        _simController->setTpsRestriction(boost::none);
    }
    ImGui::PopItemWidth();
    ImGui::EndDisabled();
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
        _statisticsWindow->reset();
        _simController->clear();
        _simController->setCurrentTimestep(_snapshot->timestep);
        _simController->updateData(_snapshot->data);
    }
    ImGui::EndDisabled();
}
