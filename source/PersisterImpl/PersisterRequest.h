#pragma once

#include "PersisterInterface/DownloadNetworkResourceRequestData.h"
#include "PersisterInterface/GetNetworkResourcesRequestData.h"
#include "PersisterInterface/LoginRequestData.h"
#include "PersisterInterface/ReadSimulationRequestData.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/SenderInfo.h"
#include "PersisterInterface/SaveSimulationRequestData.h"

class _PersisterRequest
{
public:
    PersisterRequestId const& getRequestId() const { return _requestId; }
    SenderInfo const& getSenderInfo() const { return _senderInfo; }

protected:
    _PersisterRequest(PersisterRequestId const& requestId, SenderInfo const& senderInfo)
        : _requestId(requestId)
        , _senderInfo(senderInfo) {}

    virtual ~_PersisterRequest() = default;

private:
    PersisterRequestId _requestId;
    SenderInfo _senderInfo;
};

using PersisterRequest = std::shared_ptr<_PersisterRequest>;


template <typename Data_t>
class _ConcreteRequest : public _PersisterRequest
{
public:
    Data_t const& getData() const { return _data; }

    _ConcreteRequest(PersisterRequestId const& requestId, SenderInfo const& senderInfo, Data_t const& data)
        : _PersisterRequest(requestId, senderInfo)
        , _data(data)
    {}

    virtual ~_ConcreteRequest() = default;

private:
    Data_t _data;
};

template<typename Data_t>
using ConcreteRequest = std::shared_ptr<_ConcreteRequest<Data_t>>;

using _SaveToFileRequest = _ConcreteRequest<SaveSimulationRequestData>;
using SaveToFileRequest = std::shared_ptr<_SaveToFileRequest>;

using _ReadFromFileRequest = _ConcreteRequest<ReadSimulationRequestData>;
using ReadFromFileRequest = std::shared_ptr<_ReadFromFileRequest>;

using _LoginRequest = _ConcreteRequest<LoginRequestData>;
using LoginRequest = std::shared_ptr<_LoginRequest>;

using _GetNetworkResourcesRequest = _ConcreteRequest<GetNetworkResourcesRequestData>;
using GetNetworkResourcesRequest = std::shared_ptr<_GetNetworkResourcesRequest>;

using _DownloadNetworkResourceRequest = _ConcreteRequest<DownloadNetworkResourceRequestData>;
using DownloadNetworkResourceRequest = std::shared_ptr<_DownloadNetworkResourceRequest>;
