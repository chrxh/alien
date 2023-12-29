#include "StartupController.h"

#include <imgui.h>

#include "Base/Definitions.h"
#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "Base/LoggingService.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"

#include "OpenGLHelper.h"
#include "Viewport.h"
#include "StyleRepository.h"
#include "TemporalControlWindow.h"
#include "MessageDialog.h"
#include "OverlayMessageController.h"

namespace
{
    std::chrono::milliseconds::rep const LogoDuration = 3000;
    std::chrono::milliseconds::rep const FadeOutDuration = 1500;
    std::chrono::milliseconds::rep const FadeInDuration = 500;

    auto constexpr InitialLineDistance = 15.0f;
}

_StartupController::_StartupController(
    SimulationController const& simController,
    TemporalControlWindow const& temporalControlWindow,
    Viewport const& viewport)
    : _simController(simController)
    , _temporalControlWindow(temporalControlWindow)
    , _viewport(viewport)
{
    log(Priority::Important, "starting ALIEN v" + Const::ProgramVersion);
    _logo = OpenGLHelper::loadTexture(Const::LogoFilename);
    _lineDistance = scale(InitialLineDistance);
    _startupTimepoint = std::chrono::steady_clock::now();
}

void _StartupController::process()
{
    if (_state == State::Unintialized) {
        processLoadingScreen();
        auto now = std::chrono::steady_clock::now();
        auto millisecSinceStartup =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - *_startupTimepoint).count();
        if (millisecSinceStartup > LogoDuration) {
            activate();
        }
        return;
    }

    if (_state == State::LoadSimulation) {
        DeserializedSimulation deserializedSim;
        if (!SerializerService::deserializeSimulationFromFiles(deserializedSim, Const::AutosaveFile)) {
            MessageDialog::getInstance().information("Error", "The default simulation file could not be read.\nAn empty simulation will be created.");
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
        _simController->setStatisticsHistory(deserializedSim.statistics);
        _viewport->setCenterInWorldPos(deserializedSim.auxiliaryData.center);
        _viewport->setZoomFactor(deserializedSim.auxiliaryData.zoom);
        _temporalControlWindow->onSnapshot();

        _lastActivationTimepoint = std::chrono::steady_clock::now();
        _state = State::FadeOutLoadingScreen;
        processLoadingScreen();
        return;
    }

    if (_state == State::FadeOutLoadingScreen) {
        auto now = std::chrono::steady_clock::now();
        auto millisecSinceActivation =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - *_lastActivationTimepoint).count();
        millisecSinceActivation = std::min(FadeOutDuration, millisecSinceActivation);
        auto alphaFactor = 1.0f - toFloat(millisecSinceActivation) / FadeOutDuration;

        ImGui::GetStyle().Alpha = alphaFactor;
        processLoadingScreen();

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
            _state = State::Ready;
        }
    }

    if (_state == State::Ready) {
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
    _state = State::LoadSimulation;
}

void _StartupController::processLoadingScreen()
{
    auto& styleRep = StyleRepository::getInstance();
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

    auto now = std::chrono::steady_clock::now();
    auto millisecSinceStartup = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_startupTimepoint).count();

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    ImColor textColor = Const::ProgramVersionTextColor;
    textColor.Value.w *= ImGui::GetStyle().Alpha;

    ImColor loadingTextColor = Const::ProgramVersionTextColor;
    loadingTextColor.Value.w *= ImGui::GetStyle().Alpha * 0.5f;

    //draw 'Initializing' text if it fits
    if (bottom - scale(230) > bottom / 2 + _logo.height * imageScale / 2) {
        drawGrid(bottom - scale(250), std::max(0.0f, 1.0f - toFloat(millisecSinceStartup) / LogoDuration));
        if (_state == State::Unintialized) {
            drawList->AddText(styleRep.getReefLargeFont(), scale(32.0), {center.x - scale(38), bottom - scale(270)}, loadingTextColor, "Initializing");
        }
    }
    drawList->AddText(styleRep.getReefLargeFont(), scale(48.0f), {center.x - scale(165), bottom - scale(200)}, textColor, "Artificial Life Environment");

    auto versionString = "Version " + Const::ProgramVersion;
    drawList->AddText(
        styleRep.getReefMediumFont(),
        scale(24.0f),
        {center.x - scale(toFloat(versionString.size()) * 2.8f), bottom - scale(140)},
        textColor,
        versionString.c_str());

    if (GlobalSettings::getInstance().isDebugMode()) {
        drawList->AddText(
            styleRep.getReefMediumFont(),
            scale(24.0f),
            {center.x - scale(12.0f),  bottom - scale(100)},
            textColor,
            "DEBUG");
    }
}

namespace
{
    enum class Direction
    {
        Up,
        Down,
    };
    void drawGridIntern(float yPos, float lineDistance, float maxDistance, Direction const& direction, bool includeMainLine, float alpha)
    {
        if (lineDistance > scale(InitialLineDistance)) {
            drawGridIntern(yPos, lineDistance / 2, maxDistance, direction, includeMainLine, alpha);
        }
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        alpha *= std::min(1.0f, lineDistance / scale(InitialLineDistance * 2)) * ImGui::GetStyle().Alpha;
        float accumulatedDistance = 0.0f;

        if (!includeMainLine) {
            accumulatedDistance += lineDistance;
        }
        auto right = ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x;
        auto bottom = ImGui::GetMainViewport()->Pos.y + ImGui::GetMainViewport()->Size.y;
        while (accumulatedDistance < maxDistance) {
            ImU32 color = ImColor::HSV(0.6f, 0.8f, 0.4f, alpha * (maxDistance - accumulatedDistance) / maxDistance);
            switch (direction) {
            case Direction::Up:
                drawList->AddLine(ImVec2(0.0f, yPos - accumulatedDistance), ImVec2(right, yPos - accumulatedDistance), color);
                break;
            case Direction::Down:
                drawList->AddLine(ImVec2(0.0f, yPos + accumulatedDistance), ImVec2(right, yPos + accumulatedDistance), color);
                break;
            }
            accumulatedDistance += lineDistance;
        }
    }
}

void _StartupController::drawGrid(float yPos, float alpha)
{
    drawGridIntern(yPos, _lineDistance, scale(300.0f), Direction::Up, true, alpha);
    drawGridIntern(yPos, _lineDistance, scale(300.0f), Direction::Down, false, alpha);
    _lineDistance *= 1.05f;
}
