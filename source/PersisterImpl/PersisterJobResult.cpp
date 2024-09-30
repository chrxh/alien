#include "PersisterJobResult.h"


PersisterJobId _PersisterJobResult::getId() const
{
    return _id;
}

_PersisterJobResult::_PersisterJobResult(PersisterJobId const& id)
    : _id(id)
{}

_SaveToDiscJobResult::_SaveToDiscJobResult(PersisterJobId const& id)
    : _PersisterJobResult(id)
{}
