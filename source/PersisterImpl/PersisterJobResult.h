#pragma once

#include "Definitions.h"
#include "EngineInterface/DeserializedSimulation.h"
#include "PersisterInterface/Definitions.h"

class _PersisterJobResult
{
public:
    PersisterJobId getId() const;

protected:
    _PersisterJobResult(PersisterJobId const& id);
    virtual ~_PersisterJobResult() = default;

    PersisterJobId _id;
};
using PersisterJobResult = std::shared_ptr<_PersisterJobResult>;

class _SaveToDiscJobResult : public _PersisterJobResult
{
public:
    _SaveToDiscJobResult(PersisterJobId const& id, uint64_t const& timestep, std::chrono::milliseconds const& realtime);

    uint64_t getTimestep() const;
    std::chrono::milliseconds getRealtime();

private:
    uint64_t _timestep = 0;
    std::chrono::milliseconds _realtime;
};
using SaveToDiscJobResult = std::shared_ptr<_SaveToDiscJobResult>;
