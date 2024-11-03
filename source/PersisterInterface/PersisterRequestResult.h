#pragma once

#include "PersisterInterface/ReadSimulationResultData.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/SaveSimulationResultData.h"
#include "PersisterInterface/LoginResultData.h"
#include "PersisterInterface/GetNetworkResourcesResultData.h"
#include "PersisterInterface/DownloadNetworkResourceResultData.h"
#include "PersisterInterface/UploadNetworkResourceResultData.h"
#include "PersisterInterface/ReplaceNetworkResourceResultData.h"
#include "PersisterInterface/GetUserNamesForReactionResultData.h"
#include "PersisterInterface/DeleteNetworkResourceResultData.h"
#include "PersisterInterface/EditNetworkResourceResultData.h"
#include "PersisterInterface/MoveNetworkResourceResultData.h"
#include "PersisterInterface/ToggleReactionNetworkResourceResultData.h"
#include "PersisterInterface/GetPeakSimulationResultData.h"
#include "PersisterInterface/SaveDeserializedSimulationResultData.h"

class _PersisterRequestResult
{
public:
    PersisterRequestId const& getRequestId() const { return _requestId; }

protected:
    _PersisterRequestResult(PersisterRequestId const& requestId) : _requestId(requestId) {}
    virtual ~_PersisterRequestResult() = default;

    PersisterRequestId _requestId;
};
using PersisterRequestResult = std::shared_ptr<_PersisterRequestResult>;

template <typename Data_t>
class _ConcreteRequestResult : public _PersisterRequestResult
{
public:
    Data_t const& getData() const { return _data; }

    _ConcreteRequestResult(PersisterRequestId const& requestId, Data_t const& data)
        : _PersisterRequestResult(requestId)
        , _data(data)
    {}

    virtual ~_ConcreteRequestResult() = default;

private:
    Data_t _data;
};
using PersisterRequestResult = std::shared_ptr<_PersisterRequestResult>;

template <typename Data_t>
using ConcreteRequestResult = std::shared_ptr<_ConcreteRequestResult<Data_t>>;

using _SaveSimulationRequestResult = _ConcreteRequestResult<SaveSimulationResultData>;
using _ReadSimulationRequestResult = _ConcreteRequestResult<ReadSimulationResultData>;
using _LoginRequestResult = _ConcreteRequestResult<LoginResultData>;
using _GetNetworkResourcesRequestResult = _ConcreteRequestResult<GetNetworkResourcesResultData>;
using _DownloadNetworkResourceRequestResult = _ConcreteRequestResult<DownloadNetworkResourceResultData>;
using _UploadNetworkResourceRequestResult = _ConcreteRequestResult<UploadNetworkResourceResultData>;
using _ReplaceNetworkResourceRequestResult = _ConcreteRequestResult<ReplaceNetworkResourceResultData>;
using _GetUserNamesForEmojiRequestResult = _ConcreteRequestResult<GetUserNamesForReactionResultData>;
using _DeleteNetworkResourceRequestResult = _ConcreteRequestResult<DeleteNetworkResourceResultData>;
using _EditNetworkResourceRequestResult = _ConcreteRequestResult<EditNetworkResourceResultData>;
using _MoveNetworkResourceRequestResult = _ConcreteRequestResult<MoveNetworkResourceResultData>;
using _ToggleReactionNetworkResourceRequestResult = _ConcreteRequestResult<ToggleReactionNetworkResourceResultData>;
using _GetPeakSimulationRequestResult = _ConcreteRequestResult<GetPeakSimulationResultData>;
using _SaveDeserializedSimulationRequestResult = _ConcreteRequestResult<SaveDeserializedSimulationResultData>;
