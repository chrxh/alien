#pragma once

#include <variant>

#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

using CellOrParticleDescription = std::variant<CellDescription, ParticleDescription>;

class _InspectorWindow
{
public:
    _InspectorWindow(CellOrParticleDescription const& entity, RealVector2D const& initialPos);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;
    CellOrParticleDescription getDescription() const;

private:
    std::string generateTitle() const;
    
private:
    RealVector2D _initialPos;

    bool _on = true;
    CellOrParticleDescription _entity;
};
