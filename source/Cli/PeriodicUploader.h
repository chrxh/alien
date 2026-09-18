#pragma once

#include <chrono>
#include <string>

#include <PersisterInterface/Definitions.h>
#include <PersisterInterface/DownloadCache.h>

class ConsoleLiveOutput;

class PeriodicUploader
{
public:
    PeriodicUploader(ConsoleLiveOutput& liveOutput, std::string const& baseName, std::chrono::minutes const& interval);

    void process();
    void waitForPendingUpload();

private:
    void scheduleUpload();
    void printMessage(std::string const& text) const;
    void appendResult(std::string const& text, bool isError) const;

    ConsoleLiveOutput& _liveOutput;
    std::string _baseName;
    std::chrono::minutes _interval;
    int _uploadNumber = 0;
    std::chrono::steady_clock::time_point _lastUploadTimepoint;

    TaskProcessor _taskProcessor;
    DownloadCache _downloadCache;
};
