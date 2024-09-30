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

    PersisterJobId _id = 0;
};
using PersisterJobResult = std::shared_ptr<_PersisterJobResult>;

class _SaveToDiscJobResult : public _PersisterJobResult
{
public:
    _SaveToDiscJobResult(PersisterJobId const& id);

private:
};
using SaveToDiscJobResult = std::shared_ptr<_SaveToDiscJobResult>;
