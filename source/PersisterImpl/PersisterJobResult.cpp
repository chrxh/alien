#include "PersisterJobResult.h"


PersisterJobId const& _PersisterJobResult::getId() const
{
    return _id;
}

_PersisterJobResult::_PersisterJobResult(PersisterJobId const& id)
    : _id(id)
{}

_SaveToDiscJobResult::_SaveToDiscJobResult(PersisterJobId const& id, uint64_t const& timestep, std::chrono::milliseconds const& realtime)
    : _PersisterJobResult(id)
    , _timestep(timestep)
    , _realtime(realtime)
{}

uint64_t const& _SaveToDiscJobResult::getTimestep() const
{
    return _timestep;
}

std::chrono::milliseconds const& _SaveToDiscJobResult::getRealtime()
{
    return _realtime;
}
