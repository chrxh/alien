#pragma once

#include "EngineInterface/DeserializedSimulation.h"

#include "PersisterInterface/Definitions.h"
#include "Definitions.h"

class _PersisterJob
{
public:
    PersisterJobId const& getId() const;
    bool isCritical() const;

protected:
    _PersisterJob(PersisterJobId const& id, bool critical);
    virtual ~_PersisterJob() = default;

    PersisterJobId _id;
    bool _critical = true;
};
using PersisterJob = std::shared_ptr<_PersisterJob>;

class _SaveToDiscJob : public _PersisterJob
{
public:
    _SaveToDiscJob(PersisterJobId const& id, bool critical, std::string const& filename, float const& zoom, RealVector2D const& center);

    std::string const& getFilename() const;
    float const& getZoom() const;
    RealVector2D const& getCenter() const;

private:
    std::string _filename;
    float _zoom = 0;
    RealVector2D _center;
};
using SaveToDiscJob = std::shared_ptr<_SaveToDiscJob>;
