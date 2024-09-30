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
    _SaveToDiscJob(int id, std::string const& filename, float const& zoom, RealVector2D const& center);

    std::string const& getFilename() const;
    float const& getZoom() const;
    RealVector2D const& getCenter() const;

private:
    std::string _filename;
    float _zoom;
    RealVector2D _center;
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
