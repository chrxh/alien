#pragma once

#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/SenderInfo.h"

#include "Definitions.h"

class _PersisterRequest
{
public:
    PersisterRequestId const& getId() const;
    SenderInfo const& getSenderInfo() const;

protected:
    _PersisterRequest(PersisterRequestId const& requestId, SenderInfo const& senderInfo);
    virtual ~_PersisterRequest() = default;

    PersisterRequestId _requestId;
    SenderInfo _senderInfo;
};
using PersisterRequest = std::shared_ptr<_PersisterRequest>;

class _SaveToFileJob : public _PersisterRequest
{
public:
    _SaveToFileJob(PersisterRequestId const& requestId, SenderInfo const& senderInfo, std::string const& filename, float const& zoom, RealVector2D const& center);

    std::string const& getFilename() const;
    float const& getZoom() const;
    RealVector2D const& getCenter() const;

private:
    std::string _filename;
    float _zoom = 0;
    RealVector2D _center;
};
using SaveToFileJob = std::shared_ptr<_SaveToFileJob>;

class _LoadFromFileJob : public _PersisterRequest
{
public:
    _LoadFromFileJob(PersisterRequestId const& requestId, SenderInfo const& senderInfo, std::string const& filename);

    std::string const& getFilename() const;

private:
    std::string _filename;
};
using LoadFromFileJob = std::shared_ptr<_LoadFromFileJob>;
