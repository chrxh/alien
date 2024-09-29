#include "PersisterJob.h"

int _PersisterJob::getId()
{
    return _id;
}

_PersisterJob::_PersisterJob(int id)
    : _id(id)
{}

_SaveToDiscJob::_SaveToDiscJob(int id, std::string const& filename)
    : _PersisterJob(id)
    , _filename(filename)
{}

std::string const& _SaveToDiscJob::getFilename() const
{
    return _filename;
}

_PersisterJobResult::_PersisterJobResult(int id)
    : _id(id)
{}

_SaveToDiscJobResult::_SaveToDiscJobResult(int id)
    : _PersisterJobResult(id)
{}
