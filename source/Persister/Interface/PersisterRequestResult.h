#pragma once

#include <Persister/Interface/DeleteNetworkResourceResultData.h>
#include <Persister/Interface/DownloadNetworkResourceResultData.h>
#include <Persister/Interface/EditNetworkResourceResultData.h>
#include <Persister/Interface/GetNetworkResourcesResultData.h>
#include <Persister/Interface/GetPeakSimulationResultData.h>
#include <Persister/Interface/GetSimulationPicturesResultData.h>
#include <Persister/Interface/GetUserNamesForReactionResultData.h>
#include <Persister/Interface/LoginResultData.h>
#include <Persister/Interface/MoveNetworkResourceResultData.h>
#include <Persister/Interface/PersisterRequestId.h>
#include <Persister/Interface/ReadSimulationResultData.h>
#include <Persister/Interface/ReplaceNetworkResourceResultData.h>
#include <Persister/Interface/SaveDeserializedSimulationResultData.h>
#include <Persister/Interface/SaveSimulationResultData.h>
#include <Persister/Interface/ToggleReactionNetworkResourceResultData.h>
#include <Persister/Interface/UploadNetworkResourceResultData.h>

class _PersisterRequestResult
{
public:
    PersisterRequestId const& getRequestId() const { return _requestId; }

protected:
    _PersisterRequestResult(PersisterRequestId const& requestId)
        : _requestId(requestId)
    {}
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
using _GetSimulationPicturesRequestResult = _ConcreteRequestResult<GetSimulationPicturesResultData>;
using _GetUserNamesForEmojiRequestResult = _ConcreteRequestResult<GetUserNamesForReactionResultData>;
using _DeleteNetworkResourceRequestResult = _ConcreteRequestResult<DeleteNetworkResourceResultData>;
using _EditNetworkResourceRequestResult = _ConcreteRequestResult<EditNetworkResourceResultData>;
using _MoveNetworkResourceRequestResult = _ConcreteRequestResult<MoveNetworkResourceResultData>;
using _ToggleReactionNetworkResourceRequestResult = _ConcreteRequestResult<ToggleReactionNetworkResourceResultData>;
using _GetPeakSimulationRequestResult = _ConcreteRequestResult<GetPeakSimulationResultData>;
using _SaveDeserializedSimulationRequestResult = _ConcreteRequestResult<SaveDeserializedSimulationResultData>;
