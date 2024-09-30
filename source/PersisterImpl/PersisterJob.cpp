#include "PersisterJob.h"

int _PersisterJob::getId() const
{
    return _id;
}

_PersisterJob::_PersisterJob(PersisterJobId const& id)
    : _id(id)
{}

_SaveToDiscJob::_SaveToDiscJob(PersisterJobId const& id, std::string const& filename, float const& zoom, RealVector2D const& center)
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
