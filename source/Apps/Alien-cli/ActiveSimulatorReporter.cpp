#include "ActiveSimulatorReporter.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

#include <Network/NetworkService.h>

namespace
{
    // Short runs, e.g. benchmarks, end before the first report and therefore never contact the server
    auto constexpr FirstReportDelay = std::chrono::minutes(1);
    auto constexpr ReportInterval = std::chrono::minutes(20);
}

ActiveSimulatorReporter::ActiveSimulatorReporter()
    : _reportThread([](std::stop_token const& stopToken) {
        std::mutex mutex;
        std::condition_variable_any conditionVariable;
        std::unique_lock lock(mutex);
        auto delay = FirstReportDelay;
        while (!conditionVariable.wait_for(lock, stopToken, delay, [&] { return stopToken.stop_requested(); })) {
            NetworkService::get().reportActiveSimulator();
            delay = ReportInterval;
        }
    })
{}
