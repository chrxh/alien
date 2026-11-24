#pragma once

#include <optional>
#include <string>

#include <Base/Singleton.h>

#include <McpTools/Interface/McpToolContext.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class McpController
    : public MainLoopEntity
    , public McpToolContext
{
    MAKE_SINGLETON(McpController);

public:
    void process() override;

    bool isServerRunning() const;
    void setServerRunning(bool value);

    int getPort() const;
    void setPort(int value);
    std::string getServerUrl() const;

private:
    void init() override;
    void shutdown() override;

    RealVector2D getVisibleAreaCenter() const override;
    RealVector2D getVisibleAreaSize() const override;
    float getZoomFactor() const override;
    void setVisibleArea(RealVector2D const& center, float zoomFactor) override;
    void createSimulation(std::string const& projectName, IntVector2D const& worldSize) override;
    void applySimulation(SimulationDesc const& simulation) override;
    std::string createPicture(IntVector2D const& resolution, McpPictureFormat format) override;
    std::optional<std::string> createSimulationPreviewJpg() override;
    void onSelectionChanged() override;
    void onSpeedLimitChanged() override;
    void onNetworkResourcesChanged() override;
    void onCommandExecuted(McpCommandLogEntry const& entry) override;
    void showMessage(std::string const& message) override;

    void startServer();

    int _port = 0;
    bool _startServerAtNextLaunch = false;
};
