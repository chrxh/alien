#include "KernelTracer.h"

#include <iostream>
#include <ranges>
#include <format>

#include "AlienExceptions.h"
#include "StringHelper.h"

namespace
{
    auto constexpr NumEntries = 100;
    auto constexpr RecordSize = 128;
    auto constexpr TimestepReportInterval = 100;

    std::string toRecord(std::string const& text)
    {
        auto result = text.substr(0, RecordSize - 1);
        result.resize(RecordSize - 1, ' ');
        result += '\n';
        return result;
    }
}

KernelTracer::~KernelTracer()
{
    if (_file) {
        std::fclose(_file);
    }
}

void KernelTracer::init(std::string const& filename)
{
    _file = std::fopen(filename.c_str(), "w+b");
    if (!_file) {
        throw AlienException("Could not open kernel trace file '" + filename + "'.");
    }
    std::setvbuf(_file, nullptr, _IONBF, 0);

    auto content =
        toRecord("ALIEN kernel trace: ring buffer of the last " + std::to_string(NumEntries) + " kernel calls. The newest entry has the highest call number.");
    auto blank = toRecord({});
    for ([[maybe_unused]] auto slot : std::views::iota(0, NumEntries)) {
        content += blank;
    }
    std::fwrite(content.data(), 1, content.size(), _file);
}

void KernelTracer::setTimestep(uint64_t value)
{
    if (!_file) {
        return;
    }
    std::lock_guard lock(_mutex);
    _timestep = value;
    if (value % TimestepReportInterval == 0) {
        reportProgress(value);
    }
}

void KernelTracer::traceBegin(char const* name)
{
    if (!_file) {
        return;
    }
    std::lock_guard lock(_mutex);
    ++_callIndex;
    _pendingName = name;
    writeRecord("running");
}

void KernelTracer::traceEnd(std::chrono::steady_clock::duration duration)
{
    if (!_file) {
        return;
    }
    auto milliseconds = std::chrono::duration<double, std::milli>(duration).count();

    std::lock_guard lock(_mutex);
    writeRecord(std::format("done {:>10.3f} ms", milliseconds));
}

void KernelTracer::reportProgress(uint64_t timestep)
{
    auto now = std::chrono::steady_clock::now();

    std::cout << "Timestep " << StringHelper::format(timestep);
    if (_lastReport) {
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - _lastReport->timepoint).count();
        std::cout << " (" << StringHelper::format(timestep - _lastReport->timestep) << " steps in " << StringHelper::format(milliseconds) << " ms)";
    }
    std::cout << std::endl;

    _lastReport = Report{now, timestep};
}

void KernelTracer::writeRecord(std::string const& status)
{
    auto record = toRecord(std::format("[{:>10} | timestep {:>12}] {:<56}{}", _callIndex, _timestep, _pendingName, status));

    // The header occupies the first record
    auto slot = (_callIndex - 1) % NumEntries;
    std::fseek(_file, static_cast<long>((slot + 1) * RecordSize), SEEK_SET);
    std::fwrite(record.data(), 1, record.size(), _file);
}
