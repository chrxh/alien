#pragma once

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <Base/Definitions.h>
#include <Base/RgbImage.h>

#include "Definitions.h"

// Encodes images into a video file on an own thread. On Windows, Media Foundation writes an MP4 file with H.264. Other systems pipe the
// images to ffmpeg, which needs to be installed.
class _VideoWriter
{
public:
    // Throws an AlienException if the video cannot be created. The width and height must be even, the bit rate is given in bits per second.
    _VideoWriter(std::filesystem::path const& filename, IntVector2D const& resolution, int fps, int bitRate);
    ~_VideoWriter();

    // The image needs the resolution of the video and rows that are ordered from top to bottom. Waits while the encoding lags behind.
    void addFrame(RgbImage&& image);

    // Writes the remaining frames and closes the file. Throws an AlienException if the video could not be written.
    void finish();

    // Discards the video including its file, e.g. if it has no frames
    void cancel();

    std::optional<std::string> getError() const;

private:
    std::optional<RgbImage> waitForFrame();
    bool isCancelled() const;
    void setError(std::string const& error);
    void stopEncoding();

    std::filesystem::path _filename;
    std::thread _thread;
    mutable std::mutex _mutex;
    std::condition_variable _condition;
    std::deque<RgbImage> _frames;
    bool _finishing = false;
    bool _cancelled = false;
    std::optional<std::string> _error;
};
