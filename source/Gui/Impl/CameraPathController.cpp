#include "CameraPathController.h"

#include <algorithm>

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

    // The resolution of the video is fixed when the recording starts, while the view might have changed since then
    RgbImage toVideoFrame(PictureData&& picture, IntVector2D const& resolution)
    {
        auto isCroppable = picture.resolution.x >= resolution.x && picture.resolution.y >= resolution.y && picture.resolution.x <= resolution.x + 1
            && picture.resolution.y <= resolution.y + 1;
        if (!isCroppable) {
            picture = PictureGuiService::get().scale(picture, resolution);
        }
        if (picture.resolution == resolution) {
            return RgbImage{.width = resolution.x, .height = resolution.y, .pixels = std::move(picture.pixels)};
        }

        // An odd view size loses its last column or row since the video needs an even width and height
        RgbImage result{.width = resolution.x, .height = resolution.y};
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

std::vector<CameraWaypoint> const& CameraPathController::getWaypoints() const
{
    return _waypoints;
}

void CameraPathController::setWaypoints(std::vector<CameraWaypoint> const& waypoints)
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

float CameraPathController::getSmoothing() const
{
    return _smoothing;
}

void CameraPathController::setSmoothing(float value)
{
    _smoothing = std::max(0.0f, value);
}

int CameraPathController::getFps() const
{
    return _fps;
}

void CameraPathController::setFps(int value)
{
    _fps = std::clamp(value, 1, 240);
}

int CameraPathController::getBitRate() const
{
    return _bitRate;
}

void CameraPathController::setBitRate(int value)
{
    _bitRate = std::clamp(value, 1, 1000);
}

int CameraPathController::getTimestepsPerSecond() const
{
    return _timestepsPerSecond;
}

void CameraPathController::setTimestepsPerSecond(int value)
{
    _timestepsPerSecond = std::clamp(value, 1, 100000);
}

CameraWaypoint CameraPathController::createWaypointFromView() const
{
    return CameraWaypoint{.center = Viewport::get().getCenterInWorldPos(), .zoom = Viewport::get().getZoomFactor()};
}

void CameraPathController::moveViewToWaypoint(CameraWaypoint const& waypoint) const
{
    applyCameraState(CameraState{.center = waypoint.center, .zoom = waypoint.zoom});
}

CameraPath CameraPathController::createPath() const
{
    return CameraPath(_waypoints, _smoothing);
}

RecordingSchedule CameraPathController::createRecordingSchedule() const
{
    return RecordingSchedule(createPath(), _fps, toDouble(_timestepsPerSecond));
}

bool CameraPathController::isPlaying() const
{
    return _playbackStart.has_value();
}

bool CameraPathController::isRecording() const
{
    return _recording.has_value();
}

void CameraPathController::startPlayback()
{
    stop();
    if (_waypoints.size() < 2) {
        return;
    }
    _playbackStart = std::chrono::steady_clock::now();
}

void CameraPathController::startRecording(std::filesystem::path const& filename)
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
    };
    log(Priority::Important, "recording of video " + filename.string() + " started");
}

void CameraPathController::stop()
{
    _playbackStart.reset();
    _playbackTime.reset();
    if (_recording) {
        finishRecording(true);
    }
}

std::optional<float> CameraPathController::getPlaybackTime() const
{
    return _playbackTime;
}

std::optional<CameraPathController::RecordingProgress> CameraPathController::getRecordingProgress() const
{
    if (!_recording) {
        return std::nullopt;
    }
    return RecordingProgress{.frame = _recording->frame, .numFrames = _recording->schedule.getNumFrames(), .resolution = _recording->resolution};
}

void CameraPathController::processCamera()
{
    if (_recording) {
        processRecording();
    } else if (_playbackStart) {
        processPlayback();
    }
}

void CameraPathController::init()
{
    auto& settings = GlobalSettings::get();
    setSmoothing(settings.getValue("controllers.camera path.smoothing", _smoothing));
    setFps(settings.getValue("controllers.camera path.fps", _fps));
    setBitRate(settings.getValue("controllers.camera path.bit rate", _bitRate));
    setTimestepsPerSecond(settings.getValue("controllers.camera path.time steps per second", _timestepsPerSecond));

    std::vector<CameraWaypoint> waypoints;
    auto numWaypoints = settings.getValue("controllers.camera path.number of waypoints", 0);
    for (int index = 0; index < numWaypoints; ++index) {
        auto key = "controllers.camera path.waypoint " + std::to_string(index);
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

void CameraPathController::shutdown()
{
    if (_recording) {
        finishRecording(false);
    }

    auto& settings = GlobalSettings::get();
    settings.setValue("controllers.camera path.smoothing", _smoothing);
    settings.setValue("controllers.camera path.fps", _fps);
    settings.setValue("controllers.camera path.bit rate", _bitRate);
    settings.setValue("controllers.camera path.time steps per second", _timestepsPerSecond);

    settings.setValue("controllers.camera path.number of waypoints", toInt(_waypoints.size()));
    for (int index = 0; index < toInt(_waypoints.size()); ++index) {
        auto key = "controllers.camera path.waypoint " + std::to_string(index);
        auto const& waypoint = _waypoints.at(index);
        settings.setValue(key + ".center x", waypoint.center.x);
        settings.setValue(key + ".center y", waypoint.center.y);
        settings.setValue(key + ".zoom", waypoint.zoom);
        settings.setValue(key + ".duration", waypoint.duration);
        settings.setValue(key + ".time-lapse", waypoint.timeLapse);
    }
}

void CameraPathController::processPlayback()
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

void CameraPathController::processRecording()
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
            recording.videoWriter->addFrame(toVideoFrame(std::move(*picture), recording.resolution));
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

void CameraPathController::calcTimestepsForRecording()
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

void CameraPathController::finishRecording(bool reportResult)
{
    auto recording = std::move(*_recording);
    _recording.reset();

    auto simulationFacade = _SimulationFacade::get();
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

void CameraPathController::applyCameraState(CameraState const& state) const
{
    Viewport::get().setCenterInWorldPos(state.center);
    Viewport::get().setZoomFactor(state.zoom);
}
