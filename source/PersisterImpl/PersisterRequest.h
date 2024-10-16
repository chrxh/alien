#pragma once

#include "PersisterInterface/DownloadNetworkResourceRequestData.h"
#include "PersisterInterface/GetNetworkResourcesRequestData.h"
#include "PersisterInterface/GetUserNamesForEmojiRequestData.h"
#include "PersisterInterface/LoginRequestData.h"
#include "PersisterInterface/ReadSimulationRequestData.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/ReplaceNetworkResourceRequestData.h"
#include "PersisterInterface/SenderInfo.h"
#include "PersisterInterface/SaveSimulationRequestData.h"
#include "PersisterInterface/UploadNetworkResourceRequestData.h"

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

using _SaveSimulationRequest = _ConcreteRequest<SaveSimulationRequestData>;
using SaveSimulationRequest = std::shared_ptr<_SaveSimulationRequest>;

using _ReadSimulationRequest = _ConcreteRequest<ReadSimulationRequestData>;
using ReadSimulationRequest = std::shared_ptr<_ReadSimulationRequest>;

using _LoginRequest = _ConcreteRequest<LoginRequestData>;
using LoginRequest = std::shared_ptr<_LoginRequest>;

using _GetNetworkResourcesRequest = _ConcreteRequest<GetNetworkResourcesRequestData>;
using GetNetworkResourcesRequest = std::shared_ptr<_GetNetworkResourcesRequest>;

using _DownloadNetworkResourceRequest = _ConcreteRequest<DownloadNetworkResourceRequestData>;
using DownloadNetworkResourceRequest = std::shared_ptr<_DownloadNetworkResourceRequest>;

using _UploadNetworkResourceRequest = _ConcreteRequest<UploadNetworkResourceRequestData>;
using UploadNetworkResourceRequest = std::shared_ptr<_UploadNetworkResourceRequest>;

using _ReplaceNetworkResourceRequest = _ConcreteRequest<ReplaceNetworkResourceRequestData>;
using ReplaceNetworkResourceRequest = std::shared_ptr<_ReplaceNetworkResourceRequest>;

using _GetUserNamesForEmojiRequest = _ConcreteRequest<GetUserNamesForEmojiRequestData>;
using GetUserNamesForEmojiRequest = std::shared_ptr<_GetUserNamesForEmojiRequest>;
