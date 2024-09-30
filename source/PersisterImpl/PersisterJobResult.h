#pragma once

#include "Definitions.h"
#include "EngineInterface/DeserializedSimulation.h"
#include "PersisterInterface/Definitions.h"

class _PersisterJobResult
{
public:
    PersisterJobId const& getId() const;

protected:
    _PersisterJobResult(PersisterJobId const& id);
    virtual ~_PersisterJobResult() = default;

    PersisterJobId _id;
};
using PersisterJobResult = std::shared_ptr<_PersisterJobResult>;

class _SaveToFileJobResult : public _PersisterJobResult
{
public:
    _SaveToFileJobResult(
        PersisterJobId const& id,
        std::string const& simulationName,
        uint64_t const& timestep,
        std::chrono::system_clock::time_point const& timestamp);

    std::string const& getSimulationName() const;
    uint64_t const& getTimestep() const;
    std::chrono::system_clock::time_point const& getTimestamp();

private:
    std::string _simulationName;
    uint64_t _timestep = 0;
    std::chrono::system_clock::time_point _timestamp;
};
using SaveToFileJobResult = std::shared_ptr<_SaveToFileJobResult>;
