#pragma once

#include "EngineInterface/DeserializedSimulation.h"

#include "Definitions.h"

class _PersisterJob
{
public:
    int getId();

protected:
    _PersisterJob(int id);
    virtual ~_PersisterJob() = default;

    int _id = 0;
};
using PersisterJob = std::shared_ptr<_PersisterJob>;

class _SaveToDiscJob : public _PersisterJob
{
public:
    _SaveToDiscJob(int id, std::string const& filename);

    std::string const& getFilename() const;

    SimulationController _simController;

private:
    std::string _filename;
};
using SaveToDiscJob = std::shared_ptr<_SaveToDiscJob>;


class _PersisterJobResult
{
protected:
    _PersisterJobResult(int id);
    virtual ~_PersisterJobResult() = default;

    int _id = 0;
};
using PersisterJobResult = std::shared_ptr<_PersisterJobResult>;

class _SaveToDiscJobResult : public _PersisterJobResult
{
public:
    _SaveToDiscJobResult(int id);

private:
};
using SaveToDiscJobResult = std::shared_ptr<_SaveToDiscJobResult>;
