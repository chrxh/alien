#include "StartupWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineInterface/Serializer.h"
#include "EngineImpl/SimulationController.h"
#include "OpenGLHelper.h"
#include "Resources.h"
#include "Viewport.h"

namespace
{
    std::chrono::milliseconds::rep const LogoDuration = 1500;
    std::chrono::milliseconds::rep const FadeOutDuration = 1500;
    std::chrono::milliseconds::rep const FadeInDuration = 500;
}

_StartupWindow::_StartupWindow(SimulationController const& simController, Viewport const& viewport)
    : _simController(simController)
    , _viewport(viewport)
{
    _logo = OpenGLHelper::loadTexture(Const::LogoFilename);
    _startupTimepoint = std::chrono::steady_clock::now();
}

void _StartupWindow::process()
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
        Serializer serializer = boost::make_shared<_Serializer>();
        SerializedSimulation serializedData;
        serializer->loadSimulationDataFromFile(Const::AutosaveFile, serializedData);
        auto deserializedData = serializer->deserializeSimulation(serializedData);

        _simController->newSimulation(deserializedData.timestep, deserializedData.settings, deserializedData.symbolMap);
        _simController->setSimulationData(deserializedData.content);
        _viewport->setCenterInWorldPos(
            {toFloat(deserializedData.settings.generalSettings.worldSizeX) / 2,
             toFloat(deserializedData.settings.generalSettings.worldSizeY) / 2});
        _viewport->setZoomFactor(2.0f);

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

auto _StartupWindow::getState() -> State
{
    return _state;
}

void _StartupWindow::activate()
{
    _state = State::RequestLoading;
}

void _StartupWindow::processWindow()
{
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(1212, 776));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("##startup", NULL, windowFlags);
    ImGui::Image((void*)(intptr_t)_logo.textureId, ImVec2(1182, 746));
    ImGui::End();
}
