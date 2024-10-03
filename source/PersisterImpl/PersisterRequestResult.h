#pragma once

#include "PersisterInterface/ReadSimulationResultData.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/SavedSimulationResultData.h"

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

using _SaveToFileRequestResult = _ConcreteRequestResult<SavedSimulationResultData>;
using SaveToFileRequestResult = std::shared_ptr<_SaveToFileRequestResult>;

using _ReadFromFileRequestResult = _ConcreteRequestResult<ReadSimulationResultData>;
using ReadFromFileRequestResult = std::shared_ptr<_ReadFromFileRequestResult>;
