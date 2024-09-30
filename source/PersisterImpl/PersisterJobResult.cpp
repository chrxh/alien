#include "PersisterJobResult.h"


PersisterJobId const& _PersisterJobResult::getId() const
{
    return _id;
}

_PersisterJobResult::_PersisterJobResult(PersisterJobId const& id)
    : _id(id)
{}

_SaveToFileJobResult::_SaveToFileJobResult(
    PersisterJobId const& id,
    std::string const& simulationName,
    uint64_t const& timestep,
    std::chrono::system_clock::time_point const& timestamp)
    : _PersisterJobResult(id)
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
