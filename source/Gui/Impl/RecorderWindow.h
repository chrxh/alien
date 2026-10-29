#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <Video/CameraPath.h>

#include "AlienWindow.h"
#include "Definitions.h"

class RecorderWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(RecorderWindow);

private:
    RecorderWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;

    void processToolbar();
    void processWaypointTable();
    void processSettings();
    void processStatusBar();

    void onOpenFlight();
    void onSaveFlight();
    void onAddWaypoint();
    void onDeleteWaypoint();
    void onRecordVideo();

    void updateVideoInfo();

    std::optional<int> _selectedWaypoint;
    bool _settingsOpen = true;
    float _settingsHeight = 0;
    std::string _flightDirectory;
    std::string _videoDirectory;

    // The schedule of a long video takes a while to calculate, so its key figures are only updated when the camera path changes
    struct VideoInfo
    {
        std::vector<CameraWaypoint> waypoints;
        float smoothing = 0;
        int fps = 0;
        int timestepsPerSecond = 0;

        int numFrames = 0;
        uint64_t numTimesteps = 0;
    };
    std::optional<VideoInfo> _videoInfo;
};
