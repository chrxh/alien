#pragma once

#include <optional>
#include <string>

#include <McpTools/Interface/McpToolContext.h>

#include <Rendering/Interface/PictureData.h>
#include <Rendering/Interface/RenderView.h>

class ConsoleLiveOutput;

class McpSession : public McpToolContext
{
public:
    McpSession(ConsoleLiveOutput& liveOutput, int port);
    ~McpSession() override;

    std::optional<std::string> const& getErrorMessage() const;

    RealVector2D getVisibleAreaCenter() const override;
    RealVector2D getVisibleAreaSize() const override;
    float getZoomFactor() const override;
    void setVisibleArea(RealVector2D const& center, float zoomFactor) override;

private:
    void createSimulation(std::string const& projectName, IntVector2D const& worldSize) override;
    void applySimulation(SimulationDesc const& simulation) override;
    std::string createPicture(IntVector2D const& resolution, McpPictureFormat format) override;
    std::optional<std::string> createSimulationPreviewJpg() override;
    void onCommandExecuted(McpCommandLogEntry const& entry) override;

    PictureData renderPicture(IntVector2D const& resolution);
    void setupRenderingIfNecessary();
    RenderView createRenderView(IntVector2D const& resolution) const;

    ConsoleLiveOutput& _liveOutput;
    std::optional<std::string> _errorMessage;

    RealVector2D _center;
    float _zoomFactor = 4.0f;

    bool _renderingInitialized = false;
    std::optional<std::string> _renderingErrorMessage;
};
