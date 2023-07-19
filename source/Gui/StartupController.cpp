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
#include "OverlayMessageController.h"

namespace
{
    std::chrono::milliseconds::rep const LogoDuration = 2500;
    std::chrono::milliseconds::rep const FadeOutDuration = 1500;
    std::chrono::milliseconds::rep const FadeInDuration = 500;
}

_StartupController::_StartupController(
    SimulationController const& simController,
    TemporalControlWindow const& temporalControlWindow,
    Viewport const& viewport)
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
        DeserializedSimulation deserializedSim;
        if (!Serializer::deserializeSimulationFromFiles(deserializedSim, Const::AutosaveFile)) {
            MessageDialog::getInstance().show("Error", "The default simulation file could not be read. An empty simulation will be created.");
            deserializedSim.auxiliaryData.generalSettings.worldSizeX = 1000;
            deserializedSim.auxiliaryData.generalSettings.worldSizeY = 500;
            deserializedSim.auxiliaryData.timestep = 0;
            deserializedSim.auxiliaryData.zoom = 12.0f;
            deserializedSim.auxiliaryData.center = {500.0f, 250.0f};
            deserializedSim.mainData = ClusteredDataDescription();
        }

        _simController->newSimulation(
            deserializedSim.auxiliaryData.timestep, deserializedSim.auxiliaryData.generalSettings, deserializedSim.auxiliaryData.simulationParameters);
        _simController->setClusteredSimulationData(deserializedSim.mainData);
        _viewport->setCenterInWorldPos(deserializedSim.auxiliaryData.center);
        _viewport->setZoomFactor(deserializedSim.auxiliaryData.zoom);
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
        printOverlayMessage(Const::AutosaveFileWithoutPath);
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
    auto bottom = ImGui::GetMainViewport()->Pos.y + ImGui::GetMainViewport()->Size.y;
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    auto imageScale = scale(1.0f);
    ImGui::SetNextWindowSize(ImVec2(_logo.width * imageScale + 10.0f, _logo.height * imageScale + 10.0f));

    ImGuiWindowFlags windowFlags = 0 | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
    ImGui::Begin("##startup", NULL, windowFlags);
    ImGui::Image((void*)(intptr_t)_logo.textureId, ImVec2(_logo.width * imageScale, _logo.height * imageScale));
    ImGui::End();

    drawGrid();

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    ImColor textColor = Const::ProgramVersionColor;
    textColor.Value.w = ImGui::GetStyle().Alpha;
    drawList->AddText(styleRep.getReefLargeFont(), scale(48.0f), {center.x - scale(165), bottom - scale(200)}, textColor, "Artificial Life Environment");

    auto versionString = "Version " + Const::ProgramVersion;
    drawList->AddText(
        styleRep.getReefMediumFont(),
        scale(24.0f),
        {center.x - scale(toFloat(versionString.size()) * 3.0f), bottom - scale(140)},
        textColor,
        versionString.c_str());
}

namespace
{
    enum class Direction
    {
        Up,
        Down,
        Left,
        Right
    };
    void drawGridIntern(float lineDistance, float maxDistance, Direction const& direction, bool includeMainLine)
    {
        if (lineDistance > 10.0f) {
            drawGridIntern(lineDistance / 2, maxDistance, direction, false);
        }
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto alpha = std::min(1.0f, lineDistance / 20.0f) * ImGui::GetStyle().Alpha;
        float accumulatedDistance = 0.0f;

        if (!includeMainLine) {
            accumulatedDistance += lineDistance;
        }
        auto right = ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x;
        auto bottom = ImGui::GetMainViewport()->Pos.y + ImGui::GetMainViewport()->Size.y;
        while (accumulatedDistance < maxDistance) {
            ImU32 color = ImColor::HSV(0.0f, 0.8f, 0.45f, alpha * (maxDistance - accumulatedDistance) / maxDistance);
            switch (direction) {
            case Direction::Up:
                drawList->AddLine(ImVec2(0.0f, bottom / 2 - accumulatedDistance), ImVec2(right, bottom / 2 - accumulatedDistance), color);
                break;
            case Direction::Down:
                drawList->AddLine(ImVec2(0.0f, bottom / 2 + accumulatedDistance), ImVec2(right, bottom / 2 + accumulatedDistance), color);
                break;
            case Direction::Left:
                drawList->AddLine(ImVec2(right / 2 - accumulatedDistance, 0.0f), ImVec2(right / 2 - accumulatedDistance, bottom), color);
                break;
            case Direction::Right:
                drawList->AddLine(ImVec2(right / 2 + accumulatedDistance, 0.0f), ImVec2(right / 2 + accumulatedDistance, bottom), color);
                break;
            }
            accumulatedDistance += lineDistance;
        }
    }
}

void _StartupController::drawGrid()
{
    static float lineDistance = 10.0f;
    drawGridIntern(lineDistance, 300.0f, Direction::Up, true);
    drawGridIntern(lineDistance, 300.0f, Direction::Down, false);
    //drawGridIntern(lineDistance, 1000.0f, Direction::Left, true);
    //drawGridIntern(lineDistance, 1000.0f, Direction::Right, false);
    lineDistance *= 1.01f;
}
