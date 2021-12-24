#pragma once

#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

class _InspectorWindow
{
public:
    _InspectorWindow(Viewport const& viewport, CellOrParticleDescription const& entity, RealVector2D const& initialPos);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;
    CellOrParticleDescription getDescription() const;

private:
    std::string generateTitle() const;
    
private:
    Viewport _viewport; 
    RealVector2D _initialPos;

    bool _on = true;
    CellOrParticleDescription _entity;
};
