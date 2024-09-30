#include "PersisterRequest.h"

PersisterRequestId const& _PersisterRequest::getId() const
{
    return _requestId;
}

SenderInfo const& _PersisterRequest::getSenderInfo() const
{
    return _senderInfo;
}

_PersisterRequest::_PersisterRequest(PersisterRequestId const& requestId, SenderInfo const& senderInfo)
    : _requestId(requestId)
    , _senderInfo(senderInfo)
{}

_SaveToFileJob::_SaveToFileJob(
    PersisterRequestId const& requestId,
    SenderInfo const& senderInfo,
    std::string const& filename,
    float const& zoom,
    RealVector2D const& center)
    : _PersisterRequest(requestId, senderInfo)
    , _filename(filename)
    , _zoom(zoom)
    , _center(center)
{}

std::string const& _SaveToFileJob::getFilename() const
{
    return _filename;
}

float const& _SaveToFileJob::getZoom() const
{
    return _zoom;
}

RealVector2D const& _SaveToFileJob::getCenter() const
{
    return _center;
}

_LoadFromFileJob::_LoadFromFileJob(PersisterRequestId const& requestId, SenderInfo const& senderInfo, std::string const& filename)
    : _PersisterRequest(requestId, senderInfo)
    , _filename(filename)
{}

std::string const& _LoadFromFileJob::getFilename() const
{
    return _filename;
}
