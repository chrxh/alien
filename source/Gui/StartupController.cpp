#include "StartupController.h"

#include <imgui.h>

#include "Base/Definitions.h"
#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "Base/LoggingService.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationFacade.h"
#include "PersisterInterface/PersisterFacade.h"

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

    auto const StartupSenderId = "Startup";
}

void StartupController::init(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade)
{
    _simulationFacade = simulationFacade;
    _persisterFacade =persisterFacade;

    log(Priority::Important, "starting ALIEN v" + Const::ProgramVersion);
    _logo = OpenGLHelper::loadTexture(Const::LogoFilename);
}

void StartupController::process()
{
    if (_state == State::StartLoadSimulation) {
        auto senderInfo = SenderInfo{.senderId = SenderId{StartupSenderId}, .wishResultData = true, .wishErrorInfo = true};
        auto readData = ReadSimulationRequestData{Const::AutosaveFile};
        _startupSimRequestId = _persisterFacade->scheduleReadSimulationFromFile(senderInfo, readData);
        _startupTimepoint = std::chrono::steady_clock::now();
        _state = State::LoadingSimulation;
        return;
    }

    if (_state == State::LoadingSimulation) {
        processLoadingScreen();
        auto requestedSimState = _persisterFacade->getRequestState(_startupSimRequestId);
        if (requestedSimState == PersisterRequestState::Finished) {
            auto const& data = _persisterFacade->fetchReadSimulationData(_startupSimRequestId);
            auto const& deserializedSim = data.deserializedSimulation;
            _simulationFacade->newSimulation(
                data.simulationName,
                deserializedSim.auxiliaryData.timestep,
                deserializedSim.auxiliaryData.generalSettings,
                deserializedSim.auxiliaryData.simulationParameters);
            _simulationFacade->setClusteredSimulationData(deserializedSim.mainData);
            _simulationFacade->setStatisticsHistory(deserializedSim.statistics);
            _simulationFacade->setRealTime(deserializedSim.auxiliaryData.realTime);
            Viewport::get().setCenterInWorldPos(deserializedSim.auxiliaryData.center);
            Viewport::get().setZoomFactor(deserializedSim.auxiliaryData.zoom);
            TemporalControlWindow::get().onSnapshot();

            _lastActivationTimepoint = std::chrono::steady_clock::now();
            _state = State::FadeOutLoadingScreen;
        }
        if (requestedSimState == PersisterRequestState::Error) {
            MessageDialog::get().information("Error", "The default simulation file could not be read.\nAn empty simulation will be created.");

            DeserializedSimulation deserializedSim;
            deserializedSim.auxiliaryData.generalSettings.worldSizeX = 1000;
            deserializedSim.auxiliaryData.generalSettings.worldSizeY = 500;
            deserializedSim.auxiliaryData.timestep = 0;
            deserializedSim.auxiliaryData.zoom = 12.0f;
            deserializedSim.auxiliaryData.center = {500.0f, 250.0f};
            deserializedSim.auxiliaryData.realTime = std::chrono::milliseconds(0);

            _simulationFacade->newSimulation(
                "autosave",
                deserializedSim.auxiliaryData.timestep,
                deserializedSim.auxiliaryData.generalSettings,
                deserializedSim.auxiliaryData.simulationParameters);
            _simulationFacade->setClusteredSimulationData(deserializedSim.mainData);
            _simulationFacade->setStatisticsHistory(deserializedSim.statistics);
            _simulationFacade->setRealTime(deserializedSim.auxiliaryData.realTime);
            Viewport::get().setCenterInWorldPos(deserializedSim.auxiliaryData.center);
            Viewport::get().setZoomFactor(deserializedSim.auxiliaryData.zoom);
            TemporalControlWindow::get().onSnapshot();

            _lastActivationTimepoint = std::chrono::steady_clock::now();
            _state = State::FadeOutLoadingScreen;
        }
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
            _state = State::FadeInUI;
        }
        return;
    }

    if (_state == State::FadeInUI) {
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

auto StartupController::getState() -> State
{
    return _state;
}

void StartupController::activate()
{
    _state = State::StartLoadSimulation;
}

void StartupController::processLoadingScreen()
{
    auto& styleRep = StyleRepository::get();
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

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    ImColor textColor = Const::ProgramVersionTextColor;
    textColor.Value.w *= ImGui::GetStyle().Alpha;

    drawList->AddText(styleRep.getReefLargeFont(), scale(48.0f), {center.x - scale(175), bottom - scale(200)}, textColor, "Artificial Life Environment");

    auto versionString = "Version " + Const::ProgramVersion;
    drawList->AddText(
        styleRep.getReefMediumFont(),
        scale(24.0f),
        {center.x - scale(toFloat(versionString.size()) * 3.4f), bottom - scale(140)},
        textColor,
        versionString.c_str());

    if (GlobalSettings::get().isDebugMode()) {
        drawList->AddText(
            styleRep.getReefMediumFont(),
            scale(24.0f),
            {center.x - scale(12.0f),  bottom - scale(100)},
            textColor,
            "DEBUG");
    }
}
