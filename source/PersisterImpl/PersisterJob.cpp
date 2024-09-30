#include "PersisterJob.h"

int _PersisterJob::getId()
{
    return _id;
}

_PersisterJob::_PersisterJob(int id)
    : _id(id)
{}

_SaveToDiscJob::_SaveToDiscJob(int id, std::string const& filename, float const& zoom, RealVector2D const& center)
    : _PersisterJob(id)
    , _filename(filename)
    , _zoom(zoom)
    , _center(center)
{}

std::string const& _SaveToDiscJob::getFilename() const
{
    return _filename;
}

float const& _SaveToDiscJob::getZoom() const
{
    return _zoom;
}

RealVector2D const& _SaveToDiscJob::getCenter() const
{
    return _center;
}

_PersisterJobResult::_PersisterJobResult(int id)
    : _id(id)
{}

_SaveToDiscJobResult::_SaveToDiscJobResult(int id)
    : _PersisterJobResult(id)
{}
