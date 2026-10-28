#include "RecorderController.h"

#include <algorithm>
#include <format>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>

#include <Engine/Interface/SimulationFacade.h>

#include <Rendering/Interface/RenderingFacade.h>

#include <Video/VideoWriter.h>

#include "GenericMessageDialog.h"
#include "OverlayController.h"
#include "PictureGuiService.h"
#include "Viewport.h"

namespace
{
    // At high time-lapse factors, the time steps between two frames of a video are distributed over several rendered frames, so that the user
    // interface stays responsive
    auto constexpr TargetTimestepDurationPerFrame = 0.1;

    // The rendering can lag a few frames behind a sudden change of the view, e.g. when the camera jumps to the start of the path. A frame of the
    // video is therefore only captured once its view has been rendered before, at the start several times.
    auto constexpr NumViewRenderingsAtStart = 10;

    // Like the overlay messages, the time-lapse is shown in the monospace font, here in the lower right corner of the view and the video. Its
    // size is relative to their height.
    auto constexpr TimeLapseFontSize = 0.045f;
    auto constexpr TimeLapseMargin = 0.025f;

    std::string formatTimeLapse(float timeLapse)
    {
        auto result = timeLapse < 10.0f ? std::format("{:.1f}x", timeLapse) : std::format("{:.0f}x", timeLapse);

        // Only a simulation that runs faster than normal gets the forward icon
        if (timeLapse > 1.0f && result != "1.0x") {
            result = ICON_FA_FORWARD " " + result;
        }
        return result;
    }

    // The resolution of the video is fixed when the recording starts, while the view might have changed since then
    PictureData fitToResolution(PictureData&& picture, IntVector2D const& resolution)
    {
        auto isCroppable = picture.resolution.x >= resolution.x && picture.resolution.y >= resolution.y && picture.resolution.x <= resolution.x + 1
            && picture.resolution.y <= resolution.y + 1;
        if (!isCroppable) {
            picture = PictureGuiService::get().scale(picture, resolution);
        }
        if (picture.resolution == resolution) {
            return std::move(picture);
        }

        // An odd view size loses its last column or row since the video needs an even width and height
        PictureData result{.resolution = resolution};
        auto rowSize = static_cast<size_t>(resolution.x) * PictureData::NumChannels;
        auto pictureRowSize = static_cast<size_t>(picture.resolution.x) * PictureData::NumChannels;
        result.pixels.reserve(rowSize * resolution.y);
        for (int row = 0; row < resolution.y; ++row) {
            auto rowStart = picture.pixels.begin() + row * pictureRowSize;
            result.pixels.insert(result.pixels.end(), rowStart, rowStart + rowSize);
        }
        return result;
    }
}

std::vector<CameraWaypoint> const& RecorderController::getWaypoints() const
{
    return _waypoints;
}

void RecorderController::setWaypoints(std::vector<CameraWaypoint> const& waypoints)
{
    _waypoints = waypoints;
    for (auto& waypoint : _waypoints) {
        waypoint.zoom = std::clamp(waypoint.zoom, Viewport::MinZoomFactor, Viewport::MaxZoomFactor);
        waypoint.duration = std::clamp(waypoint.duration, CameraPath::MinDuration, MaxDuration);
        waypoint.timeLapse = std::clamp(waypoint.timeLapse, MinTimeLapse, MaxTimeLapse);
    }

    // The playback needs a path, while the recording works with its own copy of it
    if (_waypoints.size() < 2) {
        _playbackStart.reset();
        _playbackTime.reset();
    }
}

float RecorderController::getSmoothing() const
{
    return _smoothing;
}

void RecorderController::setSmoothing(float value)
{
    _smoothing = std::max(0.0f, value);
}

int RecorderController::getFps() const
{
    return _fps;
}

void RecorderController::setFps(int value)
{
    _fps = std::clamp(value, 1, 240);
}

int RecorderController::getBitRate() const
{
    return _bitRate;
}

void RecorderController::setBitRate(int value)
{
    _bitRate = std::clamp(value, 1, 1000);
}

int RecorderController::getTimestepsPerSecond() const
{
    return _timestepsPerSecond;
}

void RecorderController::setTimestepsPerSecond(int value)
{
    _timestepsPerSecond = std::clamp(value, 1, 100000);
}

bool RecorderController::isTimeLapseShown() const
{
    return _timeLapseShown;
}

void RecorderController::setTimeLapseShown(bool value)
{
    _timeLapseShown = value;
}

bool RecorderController::isSimulationStartedOnRecording() const
{
    return _simulationStartedOnRecording;
}

void RecorderController::setSimulationStartedOnRecording(bool value)
{
    _simulationStartedOnRecording = value;
}

CameraWaypoint RecorderController::createWaypointFromView() const
{
    return CameraWaypoint{.center = Viewport::get().getCenterInWorldPos(), .zoom = Viewport::get().getZoomFactor()};
}

void RecorderController::moveViewToWaypoint(CameraWaypoint const& waypoint) const
{
    applyCameraState(CameraState{.center = waypoint.center, .zoom = waypoint.zoom});
}

CameraPath RecorderController::createPath() const
{
    return CameraPath(_waypoints, _smoothing);
}

RecordingSchedule RecorderController::createRecordingSchedule() const
{
    return RecordingSchedule(createPath(), _fps, toDouble(_timestepsPerSecond));
}

bool RecorderController::isPlaying() const
{
    return _playbackStart.has_value();
}

bool RecorderController::isRecording() const
{
    return _recording.has_value();
}

void RecorderController::startPlayback()
{
    stop();
    if (_waypoints.size() < 2) {
        return;
    }
    _playbackStart = std::chrono::steady_clock::now();
}

void RecorderController::startRecording(std::filesystem::path const& filename)
{
    stop();
    if (_waypoints.size() < 2) {
        throw AlienException("The camera path needs at least two waypoints.");
    }

    // H.264 requires an even width and height
    auto viewSize = Viewport::get().getViewSize();
    IntVector2D resolution{viewSize.x / 2 * 2, viewSize.y / 2 * 2};
    auto videoWriter = std::make_shared<_VideoWriter>(filename, resolution, _fps, _bitRate * 1000000);

    auto simulationFacade = _SimulationFacade::get();
    _recording = Recording{
        .schedule = createRecordingSchedule(),
        .videoWriter = videoWriter,
        .filename = filename,
        .resolution = resolution,
        .syncSimulationWithRendering = simulationFacade->isSyncSimulationWithRendering(),
        .syncSimulationWithRenderingRatio = simulationFacade->getSyncSimulationWithRenderingRatio(),
        .simulationStarted = _simulationStartedOnRecording && !simulationFacade->isSimulationRunning(),
    };

    // The simulation only advances by the time steps between the frames of the video
    simulationFacade->setSyncSimulationWithRendering(true);
    simulationFacade->setSyncSimulationWithRenderingRatio(0);
    if (_recording->simulationStarted) {
        simulationFacade->runSimulation();
    }
    OverlayController::get().setStatus("Recording...");
    log(Priority::Important, "recording of video " + filename.string() + " started");
}

void RecorderController::stop()
{
    _playbackStart.reset();
    _playbackTime.reset();
    if (_recording) {
        finishRecording(true);
    }
}

std::optional<float> RecorderController::getPlaybackTime() const
{
    return _playbackTime;
}

std::optional<RecorderController::RecordingProgress> RecorderController::getRecordingProgress() const
{
    if (!_recording) {
        return std::nullopt;
    }
    return RecordingProgress{.frame = _recording->frame, .numFrames = _recording->schedule.getNumFrames(), .resolution = _recording->resolution};
}

void RecorderController::processCamera()
{
    if (_recording) {
        processRecording();
    } else if (_playbackStart) {
        processPlayback();
    }

    if (_timeLapseShown) {
        if (_recording) {
            drawTimeLapse(_recording->schedule.getCameraState(_recording->frame).timeLapse);
        } else if (_playbackTime) {
            drawTimeLapse(createPath().getState(*_playbackTime).timeLapse);
        }
    }
}

void RecorderController::init()
{
    auto& settings = GlobalSettings::get();
    setSmoothing(settings.getValue("controllers.recorder.smoothing", _smoothing));
    setFps(settings.getValue("controllers.recorder.fps", _fps));
    setBitRate(settings.getValue("controllers.recorder.bit rate", _bitRate));
    setTimestepsPerSecond(settings.getValue("controllers.recorder.time steps per second", _timestepsPerSecond));
    setTimeLapseShown(settings.getValue("controllers.recorder.show time-lapse", _timeLapseShown));
    setSimulationStartedOnRecording(settings.getValue("controllers.recorder.start simulation", _simulationStartedOnRecording));

    std::vector<CameraWaypoint> waypoints;
    auto numWaypoints = settings.getValue("controllers.recorder.number of waypoints", 0);
    for (int index = 0; index < numWaypoints; ++index) {
        auto key = "controllers.recorder.waypoint " + std::to_string(index);
        CameraWaypoint waypoint;
        waypoint.center.x = settings.getValue(key + ".center x", waypoint.center.x);
        waypoint.center.y = settings.getValue(key + ".center y", waypoint.center.y);
        waypoint.zoom = settings.getValue(key + ".zoom", waypoint.zoom);
        waypoint.duration = settings.getValue(key + ".duration", waypoint.duration);
        waypoint.timeLapse = settings.getValue(key + ".time-lapse", waypoint.timeLapse);
        waypoints.emplace_back(waypoint);
    }
    setWaypoints(waypoints);
}

void RecorderController::shutdown()
{
    if (_recording) {
        finishRecording(false);
    }

    auto& settings = GlobalSettings::get();
    settings.setValue("controllers.recorder.smoothing", _smoothing);
    settings.setValue("controllers.recorder.fps", _fps);
    settings.setValue("controllers.recorder.bit rate", _bitRate);
    settings.setValue("controllers.recorder.time steps per second", _timestepsPerSecond);
    settings.setValue("controllers.recorder.show time-lapse", _timeLapseShown);
    settings.setValue("controllers.recorder.start simulation", _simulationStartedOnRecording);

    settings.setValue("controllers.recorder.number of waypoints", toInt(_waypoints.size()));
    for (int index = 0; index < toInt(_waypoints.size()); ++index) {
        auto key = "controllers.recorder.waypoint " + std::to_string(index);
        auto const& waypoint = _waypoints.at(index);
        settings.setValue(key + ".center x", waypoint.center.x);
        settings.setValue(key + ".center y", waypoint.center.y);
        settings.setValue(key + ".zoom", waypoint.zoom);
        settings.setValue(key + ".duration", waypoint.duration);
        settings.setValue(key + ".time-lapse", waypoint.timeLapse);
    }
}

void RecorderController::processPlayback()
{
    if (_waypoints.size() < 2) {
        stop();
        return;
    }
    auto path = createPath();
    auto time = std::chrono::duration<float>(std::chrono::steady_clock::now() - *_playbackStart).count();
    if (time >= path.getDuration()) {
        applyCameraState(path.getState(path.getDuration()));
        stop();
        return;
    }
    _playbackTime = time;
    applyCameraState(path.getState(time));
}

void RecorderController::processRecording()
{
    auto& recording = *_recording;
    if (recording.videoWriter->getError()) {
        finishRecording(true);
        return;
    }

    if (recording.captureRequested) {
        recording.captureRequested = false;

        // A frame that has not been shown, e.g. due to a minimized window, is captured again
        if (auto picture = _RenderingFacade::get()->fetchCapturedSimulationFrame()) {
            auto frame = fitToResolution(std::move(*picture), recording.resolution);
            if (_timeLapseShown) {
                drawTimeLapse(frame);
            }
            recording.videoWriter->addFrame(RgbImage{.width = frame.resolution.x, .height = frame.resolution.y, .pixels = std::move(frame.pixels)});
            recording.pendingTimesteps = recording.schedule.getTimestepsAfterFrame(recording.frame);
            recording.numViewRenderings = 0;
            ++recording.frame;
            if (recording.frame == recording.schedule.getNumFrames()) {
                finishRecording(true);
                return;
            }
        }
    }

    // The simulation only advances by the time steps between the frames of the video
    auto simulationFacade = _SimulationFacade::get();
    simulationFacade->setSyncSimulationWithRendering(true);
    simulationFacade->setSyncSimulationWithRenderingRatio(0);

    calcTimestepsForRecording();
    if (recording.pendingTimesteps > 0) {
        return;
    }

    applyCameraState(recording.schedule.getCameraState(recording.frame));
    auto requiredViewRenderings = recording.frame == 0 ? NumViewRenderingsAtStart : 1;
    if (recording.numViewRenderings >= requiredViewRenderings) {
        _RenderingFacade::get()->captureSimulationFrame();
        recording.captureRequested = true;
    }
    ++recording.numViewRenderings;
}

void RecorderController::calcTimestepsForRecording()
{
    auto& recording = *_recording;
    if (recording.pendingTimesteps == 0) {
        return;
    }

    // A paused simulation stays as it is in the video
    auto simulationFacade = _SimulationFacade::get();
    if (!simulationFacade->isSimulationRunning()) {
        recording.pendingTimesteps = 0;
        return;
    }

    auto timesteps = std::min(recording.pendingTimesteps, recording.maxTimestepsPerFrame);
    auto startTimepoint = std::chrono::steady_clock::now();
    simulationFacade->calcTimesteps(timesteps);
    auto duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTimepoint).count();
    recording.pendingTimesteps -= timesteps;

    if (timesteps == recording.maxTimestepsPerFrame) {
        auto suitableTimesteps = TargetTimestepDurationPerFrame * toDouble(timesteps) / std::max(duration, 1.0e-6);
        auto limitedTimesteps = std::clamp(suitableTimesteps, toDouble(timesteps) / 2, toDouble(timesteps) * 2);
        recording.maxTimestepsPerFrame = std::max(uint64_t{1}, static_cast<uint64_t>(limitedTimesteps));
    }
}

void RecorderController::finishRecording(bool reportResult)
{
    auto recording = std::move(*_recording);
    _recording.reset();
    OverlayController::get().setStatus(std::nullopt);

    // The simulation is paused before the synchronization is restored, so that it does not advance beyond the last frame
    auto simulationFacade = _SimulationFacade::get();
    if (recording.simulationStarted) {
        simulationFacade->pauseSimulation();
    }
    simulationFacade->setSyncSimulationWithRendering(recording.syncSimulationWithRendering);
    simulationFacade->setSyncSimulationWithRenderingRatio(recording.syncSimulationWithRenderingRatio);

    if (recording.frame == 0) {
        recording.videoWriter->cancel();
        log(Priority::Important, "recording of video " + recording.filename.string() + " stopped before its first frame");
        return;
    }
    try {
        recording.videoWriter->finish();
        log(Priority::Important, "video " + recording.filename.string() + " recorded with " + std::to_string(recording.frame) + " frames");
        if (reportResult) {
            printOverlayMessage(recording.filename.filename().string());
        }
    } catch (AlienException const& exception) {
        log(Priority::Important, std::string("video could not be recorded: ") + exception.what());
        if (reportResult) {
            GenericMessageDialog::get().information("Record video", std::string("The video could not be written.\n\n") + exception.what());
        }
    }
}

void RecorderController::applyCameraState(CameraState const& state) const
{
    Viewport::get().setCenterInWorldPos(state.center);
    Viewport::get().setZoomFactor(state.zoom);
}

void RecorderController::drawTimeLapse(float timeLapse) const
{
    auto viewport = ImGui::GetMainViewport();
    auto margin = viewport->Size.y * TimeLapseMargin;
    PictureGuiService::get().drawOverlayText(
        ImGui::GetBackgroundDrawList(),
        formatTimeLapse(timeLapse),
        viewport->Size.y * TimeLapseFontSize,
        {viewport->Pos.x + viewport->Size.x - margin, viewport->Pos.y + viewport->Size.y - margin});
}

void RecorderController::drawTimeLapse(PictureData& frame)
{
    auto& recording = *_recording;
    auto text = formatTimeLapse(recording.schedule.getCameraState(recording.frame).timeLapse);
    auto height = toFloat(frame.resolution.y);
    if (!recording.timeLapseLabel || recording.timeLapseText != text) {
        recording.timeLapseLabel = PictureGuiService::get().renderOverlayText(text, height * TimeLapseFontSize);
        recording.timeLapseText = text;
    }
    auto margin = toInt(height * TimeLapseMargin);
    PictureGuiService::get().drawOverlayText(frame, *recording.timeLapseLabel, {frame.resolution.x - margin, frame.resolution.y - margin});
}
