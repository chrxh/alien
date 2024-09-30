#include "PersisterJob.h"

PersisterJobId const& _PersisterJob::getId() const
{
    return _id;
}

bool _PersisterJob::isCritical() const
{
    return _critical;
}

_PersisterJob::_PersisterJob(PersisterJobId const& id, bool critical)
    : _id(id)
    , _critical(critical)
{}

_SaveToDiscJob::_SaveToDiscJob(PersisterJobId const& id, bool critical, std::string const& filename, float const& zoom, RealVector2D const& center)
    : _PersisterJob(id, critical)
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
