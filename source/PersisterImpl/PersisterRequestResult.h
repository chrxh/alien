#pragma once

#include "EngineInterface/DeserializedSimulation.h"
#include "PersisterInterface/PersisterRequestId.h"

class _PersisterRequestResult
{
public:
    PersisterRequestId const& getRequestId() const;

protected:
    _PersisterRequestResult(PersisterRequestId const& requestId);
    virtual ~_PersisterRequestResult() = default;

    PersisterRequestId _requestId;
};
using PersisterRequestResult = std::shared_ptr<_PersisterRequestResult>;

class _SaveToFileJobResult : public _PersisterRequestResult
{
public:
    _SaveToFileJobResult(
        PersisterRequestId const& requestId,
        std::string const& simulationName,
        uint64_t const& timestep,
        std::chrono::system_clock::time_point const& timestamp);

    std::string const& getSimulationName() const;
    uint64_t const& getTimestep() const;
    std::chrono::system_clock::time_point const& getTimestamp();

private:
    std::string _simulationName;
    uint64_t _timestep = 0;
    std::chrono::system_clock::time_point _timestamp;
};
using SaveToFileJobResult = std::shared_ptr<_SaveToFileJobResult>;
