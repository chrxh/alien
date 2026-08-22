#include "KernelTracer.h"

#include <ranges>
#include <format>

#include "AlienExceptions.h"

namespace
{
    auto constexpr NumEntries = 10;
    auto constexpr RecordSize = 160;

    // Record layout of the trace file
    auto constexpr HeaderRecord = 0;
    auto constexpr CurrentRecord = 1;
    auto constexpr SeparatorRecord = 2;
    auto constexpr FirstHistoryRecord = 3;

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
    close();
}

void KernelTracer::init(std::filesystem::path const& filename)
{
    std::lock_guard lock(_mutex);
    if (_file) {
        std::fclose(_file);
    }
    _callIndex = 0;
    _pendingName = nullptr;
    _file = std::fopen(filename.string().c_str(), "w+b");
    if (!_file) {
        throw AlienException("Could not open kernel trace file '" + filename.string() + "'.");
    }
    std::setvbuf(_file, nullptr, _IONBF, 0);

    auto content = toRecord("ALIEN kernel trace. The next line is the most recent kernel call: if it is marked as running, that kernel did not return.");
    content += toRecord({});
    content += toRecord("--- The last " + std::to_string(NumEntries) + " calls follow as a ring, oldest first from the line stated above ---");
    auto blank = toRecord({});
    for ([[maybe_unused]] auto slot : std::views::iota(0, NumEntries)) {
        content += blank;
    }
    std::fwrite(content.data(), 1, content.size(), _file);
}

void KernelTracer::close()
{
    std::lock_guard lock(_mutex);
    if (_file) {
        std::fclose(_file);
        _file = nullptr;
    }
}

void KernelTracer::setTimestep(uint64_t value)
{
    if (!_file) {
        return;
    }
    std::lock_guard lock(_mutex);
    _timestep = value;
}

void KernelTracer::traceBegin(char const* name)
{
    if (!_file) {
        return;
    }
    std::lock_guard lock(_mutex);
    ++_callIndex;
    _pendingName = name;
    writeEntry("running", false);
}

void KernelTracer::traceEnd(std::chrono::steady_clock::duration duration)
{
    if (!_file) {
        return;
    }
    auto milliseconds = std::chrono::duration<double, std::milli>(duration).count();

    std::lock_guard lock(_mutex);
    writeEntry(std::format("done {:>10.3f} ms", milliseconds), true);
}

// The record at the top of the file always holds the most recent call, the ring below keeps the calls that completed
// before it. Rewriting the whole file to keep it in chronological order is not an option, a write of that size costs
// three orders of magnitude more than a single record.
void KernelTracer::writeEntry(std::string const& status, bool intoHistory)
{
    auto entry = std::format("[{:>10} | timestep {:>12}] {:<56}{}", _callIndex, _timestep, _pendingName, status);

    // A call that is still running is not part of the history, hence it does not shift the oldest entry yet
    auto numHistoryEntries = intoHistory ? _callIndex : _callIndex - 1;
    auto oldestSlot = numHistoryEntries > NumEntries ? numHistoryEntries % NumEntries : 0;
    auto oldestLine = FirstHistoryRecord + static_cast<int>(oldestSlot) + 1;
    writeRecord(CurrentRecord, entry + std::format("  (history starts at line {})", oldestLine));

    if (intoHistory) {
        auto slot = (_callIndex - 1) % NumEntries;
        writeRecord(FirstHistoryRecord + static_cast<int>(slot), entry);
    }
}

void KernelTracer::writeRecord(int recordIndex, std::string const& text)
{
    auto record = toRecord(text);
    std::fseek(_file, static_cast<long>(recordIndex) * RecordSize, SEEK_SET);
    std::fwrite(record.data(), 1, record.size(), _file);
}
