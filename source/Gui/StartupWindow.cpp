#include "StartupWindow.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "EngineInterface/ChangeDescriptions.h"
#include "EngineInterface/Serializer.h"
#include "EngineImpl/SimulationController.h"
#include "OpenGLHelper.h"
#include "Resources.h"
#include "Viewport.h"
#include "TemporalControlWindow.h"
#include "SpatialControlWindow.h"
#include "StatisticsWindow.h"

namespace
{
    auto const FadeOutDuration = 2000ll;
    auto const FadeInDuration = 500ll;
}

_StartupWindow::_StartupWindow(
    SimulationController const& simController,
    Viewport const& viewport,
    TemporalControlWindow const& temporalControlWindow,
    SpatialControlWindow const& spatialControlWindow,
    StatisticsWindow const& statisticsWindow)
    : _simController(simController)
    , _viewport(viewport)
    , _temporalControlWindow(temporalControlWindow)
    , _spatialControlWindow(spatialControlWindow)
    , _statisticsWindow(statisticsWindow)
{
    _logo = OpenGLHelper::loadTexture(Const::LogoFilename);
}

void _StartupWindow::process()
{
    if (_state == State::Unintialized) {
        processWindow();
        return;
    }

    if (_state == State::RequestLoading) {
        Serializer serializer = boost::make_shared<_Serializer>();
        SerializedSimulation serializedData;
        serializer->loadSimulationDataFromFile(Const::AutosaveFile, serializedData);
        auto deserializedData = serializer->deserializeSimulation(serializedData);

        _simController->newSimulation(
            deserializedData.timestep,
            deserializedData.generalSettings,
            deserializedData.simulationParameters,
            SymbolMap());
        _simController->updateData(deserializedData.content);

        _viewport->setCenterInWorldPos(
            {toFloat(deserializedData.generalSettings.worldSize.x) / 2,
             toFloat(deserializedData.generalSettings.worldSize.y) / 2});
        _viewport->setZoomFactor(4.0f);

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
//            ImGui::GetStyle().Alpha = 1.0f;
            _temporalControlWindow->setOn(true);
            _spatialControlWindow->setOn(true);
            _statisticsWindow->setOn(true);
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
            _state = State::Finished;
        }
    }

    if (_state == State::Finished) {
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
    ImGui::SetNextWindowSize(ImVec2(1220, 620));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("##startup", NULL, windowFlags);
    ImGui::Image((void*)(intptr_t)_logo.textureId, ImVec2(1200, 600));
    ImGui::End();
}
