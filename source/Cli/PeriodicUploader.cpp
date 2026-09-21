#include "PeriodicUploader.h"

#include <thread>

#include <Base/StringHelper.h>

#include <ConsoleUi/ConsoleLiveOutput.h>
#include <ConsoleUi/ConsoleWidgets.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/TaskProcessor.h>

namespace
{
    auto constexpr PollInterval = std::chrono::milliseconds(50);
    auto constexpr UploadDescription = "Periodically uploaded by the ALIEN command line.";
}

PeriodicUploader::PeriodicUploader(ConsoleLiveOutput& liveOutput, std::string const& baseName, std::chrono::minutes const& interval)
    : _liveOutput(liveOutput)
    , _baseName(baseName)
    , _interval(interval)
    , _lastUploadTimepoint(std::chrono::steady_clock::now())
    , _taskProcessor(_TaskProcessor::createTaskProcessor(_PersisterFacade::get()))
    , _downloadCache(std::make_shared<_DownloadCache>())
{}

void PeriodicUploader::process()
{
    auto now = std::chrono::steady_clock::now();
    if (!_taskProcessor->pendingTasks() && now - _lastUploadTimepoint >= _interval) {
        _lastUploadTimepoint = now;
        scheduleUpload();
    }
    _taskProcessor->process();
}

void PeriodicUploader::waitForPendingUpload()
{
    while (_taskProcessor->pendingTasks()) {
        _taskProcessor->process();
        std::this_thread::sleep_for(PollInterval);
    }
}

void PeriodicUploader::scheduleUpload()
{
    ++_uploadNumber;
    auto resourceName = _baseName + "_" + std::to_string(_uploadNumber);

    auto worldSize = _SimulationFacade::get()->getWorldSize();
    UploadNetworkResourceRequestData requestData{
        .resourceWithoutFolderName = resourceName,
        .resourceDescription = UploadDescription,
        .workspaceType = WorkspaceType_Private,
        .downloadCache = _downloadCache,
        .data = UploadNetworkResourceRequestData::SimulationData{.center = {toFloat(worldSize.x) / 2, toFloat(worldSize.y) / 2}}};

    printMessage("Uploading '" + resourceName + "' ...");
    _taskProcessor->executeTask(
        [&](auto const& senderId) {
            return _PersisterFacade::get()->scheduleUploadNetworkResource(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [this](auto const& requestId) {
            _PersisterFacade::get()->fetchUploadNetworkResourcesData(requestId);
            appendResult("done", false);
        },
        [this](auto const& errors) {
            auto const& message = errors.empty() ? std::string() : errors.front().message;
            appendResult(message.empty() ? "failed" : message.substr(0, message.find('\n')), true);
        });
}

void PeriodicUploader::printMessage(std::string const& text) const
{
    auto timestamp = "[" + StringHelper::format(std::chrono::system_clock::now()) + "]";
    _liveOutput.printMessage(
        "  " + ConsoleWidgets::createText(timestamp, ConsolePalette::Label) + " " + ConsoleWidgets::createText(text, ConsolePalette::Success));
}

void PeriodicUploader::appendResult(std::string const& text, bool isError) const
{
    _liveOutput.appendToLastMessage(" " + ConsoleWidgets::createText(text, isError ? ConsolePalette::Error : ConsolePalette::Success));
}
