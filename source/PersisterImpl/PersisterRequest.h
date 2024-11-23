#pragma once

#include "PersisterInterface/DeleteNetworkResourceRequestData.h"
#include "PersisterInterface/DownloadNetworkResourceRequestData.h"
#include "PersisterInterface/EditNetworkResourceRequestData.h"
#include "PersisterInterface/GetNetworkResourcesRequestData.h"
#include "PersisterInterface/GetPeakSimulationRequestData.h"
#include "PersisterInterface/GetUserNamesForReactionRequestData.h"
#include "PersisterInterface/LoginRequestData.h"
#include "PersisterInterface/MoveNetworkResourceRequestData.h"
#include "PersisterInterface/ReadSimulationRequestData.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/ReplaceNetworkResourceRequestData.h"
#include "PersisterInterface/SaveDeserializedSimulationRequestData.h"
#include "PersisterInterface/SaveSimulationRequestData.h"
#include "PersisterInterface/SenderInfo.h"
#include "PersisterInterface/ToggleReactionNetworkResourceRequestData.h"
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

using _GetUserNamesForEmojiRequest = _ConcreteRequest<GetUserNamesForReactionRequestData>;
using GetUserNamesForEmojiRequest = std::shared_ptr<_GetUserNamesForEmojiRequest>;

using _DeleteNetworkResourceRequest = _ConcreteRequest<DeleteNetworkResourceRequestData>;
using DeleteNetworkResourceRequest = std::shared_ptr<_DeleteNetworkResourceRequest>;

using _EditNetworkResourceRequest = _ConcreteRequest<EditNetworkResourceRequestData>;
using EditNetworkResourceRequest = std::shared_ptr<_EditNetworkResourceRequest>;

using _MoveNetworkResourceRequest = _ConcreteRequest<MoveNetworkResourceRequestData>;
using MoveNetworkResourceRequest = std::shared_ptr<_MoveNetworkResourceRequest>;

using _ToggleReactionNetworkResourceRequest = _ConcreteRequest<ToggleReactionNetworkResourceRequestData>;
using ToggleReactionNetworkResourceRequest = std::shared_ptr<_ToggleReactionNetworkResourceRequest>;

using _GetPeakSimulationRequest = _ConcreteRequest<GetPeakSimulationRequestData>;
using GetPeakSimulationRequest = std::shared_ptr<_GetPeakSimulationRequest>;

using _SaveDeserializedSimulationRequest = _ConcreteRequest<SaveDeserializedSimulationRequestData>;
using SaveDeserializedSimulationRequest = std::shared_ptr<_SaveDeserializedSimulationRequest>;
