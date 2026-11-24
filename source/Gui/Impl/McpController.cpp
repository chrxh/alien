#include "McpController.h"

#include <algorithm>
#include <stdexcept>
#include <format>

#include <Base/GlobalSettings.h>

#include <Engine/Interface/SimulationFacade.h>

#include <McpTools/Interface/McpToolsFacade.h>

#include <Network/McpServer.h>

#include <Rendering/Interface/PictureService.h>

#include "BrowserWindow.h"
#include "ConsoleModeController.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "MainLoopController.h"
#include "OverlayController.h"
#include "PictureGuiService.h"
#include "SimulationSessionGuiService.h"
#include "SimulationView.h"
#include "Viewport.h"

void McpController::process()
{
    if (!MainLoopController::get().isOperatingMode()) {
        return;
    }
    _McpToolsFacade::get()->processToolCalls();
}

bool McpController::isServerRunning() const
{
    return _McpToolsFacade::get()->isServerRunning();
}

void McpController::setServerRunning(bool value)
{
    if (value) {
        startServer();
        if (isServerRunning()) {
            _startServerAtNextLaunch = true;
            printOverlayMessage("MCP server running at " + getServerUrl());
        }
    } else if (isServerRunning()) {
        _McpToolsFacade::get()->stopServer();
        _startServerAtNextLaunch = false;
        printOverlayMessage("MCP server stopped");
    }
}

int McpController::getPort() const
{
    return _port;
}

void McpController::setPort(int value)
{
    auto port = std::clamp(value, McpServer::MinPort, McpServer::MaxPort);
    if (port == _port) {
        return;
    }
    _port = port;
    if (isServerRunning()) {
        _McpToolsFacade::get()->stopServer();
        setServerRunning(true);
    }
}

std::string McpController::getServerUrl() const
{
    return McpServer::createUrl(_port);
}

void McpController::init()
{
    _McpToolsFacade::get()->setup(*this);

    setPort(GlobalSettings::get().getValue("settings.mcp server.port", McpServer::DefaultPort));
    _startServerAtNextLaunch = GlobalSettings::get().getValue("settings.mcp server.enabled", false);
    if (_startServerAtNextLaunch) {
        startServer();
    }
}

void McpController::shutdown()
{
    GlobalSettings::get().setValue("settings.mcp server.port", _port);
    GlobalSettings::get().setValue("settings.mcp server.enabled", _startServerAtNextLaunch);
    _McpToolsFacade::get()->shutdown();
}

RealVector2D McpController::getVisibleAreaCenter() const
{
    return Viewport::get().getCenterInWorldPos();
}

RealVector2D McpController::getVisibleAreaSize() const
{
    auto viewSize = Viewport::get().getViewSize();
    auto zoom = Viewport::get().getZoomFactor();
    return {toFloat(viewSize.x) / zoom, toFloat(viewSize.y) / zoom};
}

float McpController::getZoomFactor() const
{
    return Viewport::get().getZoomFactor();
}

void McpController::setVisibleArea(RealVector2D const& center, float zoomFactor)
{
    Viewport::get().setCenterInWorldPos(center);
    Viewport::get().setZoomFactor(zoomFactor);
}

void McpController::createSimulation(std::string const& projectName, IntVector2D const& worldSize)
{
    SimulationSessionGuiService::get().createSimulation(SimulationSessionGuiService::Parameters()
                                                            .projectName(projectName)
                                                            .worldSize(worldSize)
                                                            .externalEnergy(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
}

void McpController::applySimulation(SimulationDesc const& simulation)
{
    if (auto errorMessage = SimulationSessionGuiService::get().loadSimulation(simulation)) {
        throw std::runtime_error(*errorMessage);
    }
}

std::string McpController::createPicture(IntVector2D const& resolution, McpPictureFormat format)
{
    auto picture = SimulationView::get().savePicture(resolution);
    return format == McpPictureFormat::Png ? PictureService::get().encodePng(picture) : PictureService::get().encodeJpg(picture);
}

std::optional<std::string> McpController::createSimulationPreviewJpg()
{
    return PictureGuiService::get().createSimulationPreviewJpg();
}

void McpController::onSelectionChanged()
{
    EditorModel::get().update();
}

void McpController::onSpeedLimitChanged()
{
    ConsoleModeController::get().onSpeedLimitChanged();
}

void McpController::onNetworkResourcesChanged()
{
    BrowserWindow::get().onRefresh();
}

void McpController::onCommandExecuted(McpCommandLogEntry const& entry)
{
    ConsoleModeController::get().onMcpCommandExecuted(entry);
}

void McpController::showMessage(std::string const& message)
{
    printOverlayMessage(message);
}

void McpController::startServer()
{
    if (!_McpToolsFacade::get()->startServer(_port)) {
        GenericMessageDialog::get().information("MCP server", std::format("The MCP server could not be started because port {} is not available.", _port));
    }
}
