#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"

#include "Definitions.h"

//TODO ViewDataRepository
class _SimulationScrollbar
{
public:
    enum class Orientation
    {
        Horizontal, Vertical
    };
    _SimulationScrollbar(std::string const& id, Orientation orientation, SimulationController const& simController);

    void setVisibleWorldSection(float startWorldPos, float endWorldPos);
    void draw(RealVector2D const& topLeft, RealVector2D const& size);
private:
    std::string _id;
    Orientation _orientation = Orientation::Horizontal;
    SimulationController _simController;

    float _startWorldPos = 0.0f;
    float _endWorldPos = 0.0f;
};