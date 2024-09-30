#include "PersisterRequestResult.h"


PersisterRequestId const& _PersisterRequestResult::getRequestId() const
{
    return _requestId;
}

_PersisterRequestResult::_PersisterRequestResult(PersisterRequestId const& requestId)
    : _requestId(requestId)
{}

_SaveToFileJobResult::_SaveToFileJobResult(
    PersisterRequestId const& requestId,
    std::string const& simulationName,
    uint64_t const& timestep,
    std::chrono::system_clock::time_point const& timestamp)
    : _PersisterRequestResult(requestId)
    , _simulationName(simulationName)
    , _timestep(timestep)
    , _timestamp(timestamp)
{}

std::string const& _SaveToFileJobResult::getSimulationName() const
{
    return _simulationName;
}

uint64_t const& _SaveToFileJobResult::getTimestep() const
{
    return _timestep;
}

std::chrono::system_clock::time_point const& _SaveToFileJobResult::getTimestamp()
{
    return _timestamp;
}
