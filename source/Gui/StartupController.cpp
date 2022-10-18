#include "StartupController.h"

#include <imgui.h>

#include "Base/Definitions.h"
#include "Base/Resources.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/SimulationController.h"
#include "OpenGLHelper.h"
#include "Viewport.h"
#include "StyleRepository.h"
#include "TemporalControlWindow.h"
#include "MessageDialog.h"

namespace
{
    std::chrono::milliseconds::rep const LogoDuration = 1500;
    std::chrono::milliseconds::rep const FadeOutDuration = 1500;
    std::chrono::milliseconds::rep const FadeInDuration = 500;
}

_StartupController::_StartupController(SimulationController const& simController, TemporalControlWindow const& temporalControlWindow, Viewport const& viewport)
    : _simController(simController)
    , _temporalControlWindow(temporalControlWindow)
    , _viewport(viewport)
{
    _logo = OpenGLHelper::loadTexture(Const::LogoFilename);
    _startupTimepoint = std::chrono::steady_clock::now();
}

void _StartupController::process()
{
    if (_state == State::Unintialized) {
        processWindow();
        auto now = std::chrono::steady_clock::now();
        auto millisecSinceStartup =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - *_startupTimepoint).count();
        if (millisecSinceStartup > LogoDuration) {
            activate();
        }
        return;
    }

    if (_state == State::RequestLoading) {
        DeserializedSimulation deserializedData;
        if (!Serializer::deserializeSimulationFromFiles(deserializedData, Const::AutosaveFile)) {
            MessageDialog::getInstance().show("Error", "The default simulation file could not be read. An empty simulation will be created.");
            deserializedData.settings.generalSettings.worldSizeX = 1000;
            deserializedData.settings.generalSettings.worldSizeY = 500;
            deserializedData.timestep = 0;
        }

        _simController->newSimulation(deserializedData.timestep, deserializedData.settings);
        _simController->setClusteredSimulationData(deserializedData.content);
        _viewport->setCenterInWorldPos(
            {toFloat(deserializedData.settings.generalSettings.worldSizeX) / 2,
             toFloat(deserializedData.settings.generalSettings.worldSizeY) / 2});
        _viewport->setZoomFactor(2.0f);
        _temporalControlWindow->onSnapshot();

        _lastActivationTimepoint = std::chrono::steady_clock::now();
        _state = State::LoadingSimulation;
        processWindow();
        return;
    }

    if (_state == State::LoadingSimulation) {
        auto now = std::chrono::steady_clock::now();
        auto millisecSinceActivation =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - *_lastActivationTimepoint).count();
        millisecSinceActivation = std::min(FadeOutDuration, millisecSinceActivation);
        auto alphaFactor = 1.0f - toFloat(millisecSinceActivation) / FadeOutDuration;

        ImGui::GetStyle().Alpha = alphaFactor;
        processWindow();

        if (alphaFactor == 0.0f) {
            _state = State::LoadingControls;
        }
        return;
    }

    if (_state == State::LoadingControls) {
        auto now = std::chrono::steady_clock::now();
        auto millisecSinceActivation =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - *_lastActivationTimepoint).count()
            - FadeOutDuration;
        millisecSinceActivation = std::min(FadeInDuration, millisecSinceActivation);
        auto alphaFactor = toFloat(millisecSinceActivation) / FadeInDuration;
        ImGui::GetStyle().Alpha = alphaFactor;
        if (alphaFactor == 1.0f) {
            _state = State::FinishedLoading;
        }
    }

    if (_state == State::FinishedLoading) {
        return;
    }
}

auto _StartupController::getState() -> State
{
    return _state;
}

void _StartupController::activate()
{
    _state = State::RequestLoading;
}

void _StartupController::processWindow()
{
    auto styleRep = StyleRepository::getInstance();
    auto center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    auto imageScale = styleRep.scaleContent(2.0f);
    ImGui::SetNextWindowSize(ImVec2(_logo.width * imageScale + 30.0f, _logo.height * imageScale + 30.0f));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("##startup", NULL, windowFlags);
    ImGui::Image((void*)(intptr_t)_logo.textureId, ImVec2(_logo.width * imageScale, _logo.height * imageScale));
    ImGui::End();

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();

    ImColor textColor = Const::ProgramVersionColor;
    textColor.Value.w = ImGui::GetStyle().Alpha;

    drawList->AddText(
        styleRep.getLargeFont(),
        styleRep.scaleContent(48.0f),
        {center.x - styleRep.scaleContent(250), center.y + styleRep.scaleContent(160)},
        textColor,
        "Artificial Life Environment");
    drawList->AddText(
        styleRep.getMediumFont(),
        styleRep.scaleContent(20.0f),
        {center.x - styleRep.scaleContent(50), center.y + styleRep.scaleContent(220)},
        textColor,
        ("Version " + Const::ProgramVersion).c_str());
}
