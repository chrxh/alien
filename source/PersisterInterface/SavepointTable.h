#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <string>

#include "Definitions.h"

using SavepointState = int;
enum SavepointState_
{
    SavepointState_InQueue,
    SavepointState_InProgress,
    SavepointState_Persisted,
    SavepointState_Error
};

struct SavepointEntry
{
    std::filesystem::path filename;
    SavepointState state = SavepointState_InQueue;
    std::string timestamp;
    std::string name;
    uint64_t timestep = 0;

    std::string requestId;  // transient
};

class SavepointTable
{
    friend SavepointTableService;

public:
    SavepointEntry const& at(int index) const { return _entries.at(index); }
    int getSize() const { return toInt(_entries.size()); }
    std::string const& getFilename() const { return _filename; }
    int const& getSequenceNumber() const { return _sequenceNumber; }

private:
    SavepointTable(std::string const& filename, std::deque<SavepointEntry> const& entries)
        : _filename(filename)
        , _entries(entries)
    {}

    std::string _filename;
    int _sequenceNumber = 0;
    std::deque<SavepointEntry> _entries;
};

