#include "McpSession.h"

#include <stdexcept>
#include <format>

#include <Base/LoggingService.h>

#include <ConsoleUi/ConsoleLiveOutput.h>
#include <ConsoleUi/ConsoleWidgets.h>

#include <Engine/Interface/SimulationFacade.h>

#include <McpTools/Interface/McpToolsFacade.h>

#include <Persister/Interface/SimulationReplacementService.h>

#include <Rendering/Interface/PictureService.h>
#include <Rendering/Interface/RenderingFacade.h>

namespace
{
    auto constexpr ViewSize = IntVector2D{1920, 1080};
    auto constexpr InitialZoomFactor = 4.0f;
}

McpSession::McpSession(ConsoleLiveOutput& liveOutput, int port)
    : _liveOutput(liveOutput)
{
    _McpToolsFacade::get()->setup(*this);
    if (!_McpToolsFacade::get()->startServer(port)) {
        _errorMessage = std::format("The MCP server could not be started because port {} is not available.", port);
    }
}

McpSession::~McpSession()
{
    _McpToolsFacade::get()->shutdown();
}

std::optional<std::string> const& McpSession::getErrorMessage() const
{
    return _errorMessage;
}

RealVector2D McpSession::getVisibleAreaCenter() const
{
    return _center;
}

RealVector2D McpSession::getVisibleAreaSize() const
{
    return toRealVector2D(ViewSize) / _zoomFactor;
}

float McpSession::getZoomFactor() const
{
    return _zoomFactor;
}

void McpSession::setVisibleArea(RealVector2D const& center, float zoomFactor)
{
    _center = center;
    _zoomFactor = zoomFactor;
}

void McpSession::createSimulation(std::string const& projectName, IntVector2D const& worldSize)
{
    SimulationReplacementService::get().createSimulation(SimulationReplacementService::NewSimulationParameters()
                                                             .projectName(projectName)
                                                             .worldSize(worldSize)
                                                             .externalEnergy(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
    setVisibleArea(toRealVector2D(worldSize) / 2, InitialZoomFactor);
}

void McpSession::applySimulation(SimulationDesc const& simulation)
{
    auto errorMessage = SimulationReplacementService::get().loadSimulation(simulation);
    setVisibleArea(simulation._center, simulation._zoom);
    if (errorMessage) {
        throw std::runtime_error(*errorMessage);
    }
}

std::string McpSession::createPicture(IntVector2D const& resolution, McpPictureFormat format)
{
    auto picture = renderPicture(resolution);
    return format == McpPictureFormat::Png ? PictureService::get().encodePng(picture) : PictureService::get().encodeJpg(picture);
}

std::optional<std::string> McpSession::createSimulationPreviewJpg()
{
    try {
        return PictureService::get().createPreviewJpg(renderPicture(PictureService::get().calcPreviewRenderResolution(ViewSize.x)));
    } catch (std::exception const& exception) {
        log(Priority::Important, std::string("preview picture could not be created: ") + exception.what());
        return std::string();
    }
}

void McpSession::onCommandExecuted(McpCommandLogEntry const& entry)
{
    _liveOutput.printMessage(ConsoleWidgets::createCommandMessage(entry.time, entry.toolName, entry.isError ? entry.result : entry.arguments, entry.isError));
}

PictureData McpSession::renderPicture(IntVector2D const& resolution)
{
    setupRenderingIfNecessary();
    return _RenderingFacade::get()->renderSimulationPicture(createRenderView(resolution));
}

void McpSession::setupRenderingIfNecessary()
{
    if (_renderingInitialized) {
        return;
    }
    if (!_renderingErrorMessage) {
        try {
            _RenderingFacade::get()->setupForPicturesWithoutWindow();
            _renderingInitialized = true;
            return;
        } catch (std::exception const& exception) {
            _renderingErrorMessage = std::string("Pictures cannot be rendered on this system. ") + exception.what();
        }
    }
    throw std::runtime_error(*_renderingErrorMessage);
}

RenderView McpSession::createRenderView(IntVector2D const& resolution) const
{
    auto renderScale = toFloat(resolution.x) / toFloat(ViewSize.x);
    auto zoomFactor = _zoomFactor * renderScale;
    auto topLeft = _center - toRealVector2D(resolution) / (2 * zoomFactor);
    auto bottomRight = topLeft + toRealVector2D({resolution.x - 1, resolution.y - 1}) / zoomFactor;
    return RenderView{
        .viewSize = resolution,
        .zoomFactor = zoomFactor,
        .renderScale = renderScale,
        .visibleWorldRect = {topLeft, bottomRight},
    };
}
