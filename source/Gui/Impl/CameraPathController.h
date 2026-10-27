#pragma once

#include <chrono>
#include <filesystem>
#include <optional>
#include <vector>

#include <Base/Singleton.h>

#include <Video/CameraPath.h>
#include <Video/Definitions.h>
#include <Video/RecordingSchedule.h>

#include "Definitions.h"
#include "MainLoopEntity.h"
#include "PictureGuiService.h"

// Flies the camera along a path through waypoints and records the flight as a video. During the recording, the controller calculates the
// time steps of the simulation itself, so that each frame of the video shows the simulation as many time steps later as the time-lapse
// prescribes, independent of how fast the frames are rendered.
class CameraPathController : public MainLoopEntity
{
    MAKE_SINGLETON(CameraPathController);

public:
    static auto constexpr MaxDuration = 3600.0f;
    static auto constexpr MinTimeLapse = 0.01f;
    static auto constexpr MaxTimeLapse = 100000.0f;

    static auto constexpr DefaultSmoothing = 1.0f;
    static auto constexpr DefaultFps = 30;
    static auto constexpr DefaultBitRate = 30;
    static auto constexpr DefaultTimestepsPerSecond = 60;
    static auto constexpr DefaultTimeLapseShown = true;

    std::vector<CameraWaypoint> const& getWaypoints() const;
    void setWaypoints(std::vector<CameraWaypoint> const& waypoints);

    // Standard deviation of the Gaussian kernel in seconds
    float getSmoothing() const;
    void setSmoothing(float value);

    int getFps() const;
    void setFps(int value);

    // In MBit per second
    int getBitRate() const;
    void setBitRate(int value);

    // Speed of the simulation in the video at a time-lapse of 1
    int getTimestepsPerSecond() const;
    void setTimestepsPerSecond(int value);

    // The current time-lapse is shown in the lower right corner of the view and the video
    bool isTimeLapseShown() const;
    void setTimeLapseShown(bool value);

    CameraWaypoint createWaypointFromView() const;
    void moveViewToWaypoint(CameraWaypoint const& waypoint) const;

    // Requires at least one waypoint
    CameraPath createPath() const;
    RecordingSchedule createRecordingSchedule() const;

    bool isPlaying() const;
    bool isRecording() const;

    // The playback shows the camera flight in real time while the simulation keeps its speed
    void startPlayback();

    // The resolution of the video is the current size of the view. Throws an AlienException if the video cannot be created.
    void startRecording(std::filesystem::path const& filename);

    // A stopped recording keeps the frames that have been recorded so far
    void stop();

    // Seconds of the camera flight shown in the current frame
    std::optional<float> getPlaybackTime() const;

    struct RecordingProgress
    {
        int frame = 0;
        int numFrames = 0;
        IntVector2D resolution;
    };
    std::optional<RecordingProgress> getRecordingProgress() const;

    // Has to be called before the simulation is drawn, so that the frame shows the current camera
    void processCamera();

private:
    void init() override;
    void process() override {}
    void shutdown() override;

    void processPlayback();
    void processRecording();
    void calcTimestepsForRecording();
    void finishRecording(bool reportResult);

    void applyCameraState(CameraState const& state) const;
    void drawTimeLapse(float timeLapse) const;
    void drawTimeLapse(PictureData& frame);

    std::vector<CameraWaypoint> _waypoints;
    float _smoothing = DefaultSmoothing;
    int _fps = DefaultFps;
    int _bitRate = DefaultBitRate;
    int _timestepsPerSecond = DefaultTimestepsPerSecond;
    bool _timeLapseShown = DefaultTimeLapseShown;

    std::optional<std::chrono::steady_clock::time_point> _playbackStart;
    std::optional<float> _playbackTime;

    struct Recording
    {
        RecordingSchedule schedule;
        VideoWriter videoWriter;
        std::filesystem::path filename;
        IntVector2D resolution;

        int frame = 0;
        uint64_t pendingTimesteps = 0;
        int numViewRenderings = 0;
        bool captureRequested = false;
        uint64_t maxTimestepsPerFrame = 1;

        // The label of the time-lapse is only rendered again when its text changes
        std::string timeLapseText;
        std::optional<PictureGuiService::OverlayText> timeLapseLabel;

        // Settings of the simulation that the recording overrides
        bool syncSimulationWithRendering = false;
        int syncSimulationWithRenderingRatio = 0;
    };
    std::optional<Recording> _recording;
};
