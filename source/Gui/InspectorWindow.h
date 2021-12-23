#pragma once

#include <variant>

#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

using CellOrParticleDescription = std::variant<CellDescription, ParticleDescription>;

class _InspectorWindow
{
public:
    _InspectorWindow(CellOrParticleDescription const& entity, Viewport const& viewport);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;

private:
    std::string generateTitle() const;
    RealVector2D getEntityPos() const;

private:
    Viewport _viewport;

    bool _on = true;
    CellOrParticleDescription _entity;
};
