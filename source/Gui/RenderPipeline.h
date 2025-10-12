#pragma once

#include "EngineInterface/RenderData.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"

class _GeometrySource
{
public:
    static GeometrySource create();

    unsigned int getVao() const { return vao; }
    unsigned int getVbo() const { return vbo; }
    unsigned int getEbo() const { return ebo; }

private:

    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;

    _GeometrySource() = default;
};

class _RenderPipeline
{
public:
    _RenderPipeline(SimulationFacade const& simulationFacade);

    void addStep(RenderStep const& step);

    void resize(IntVector2D const& size);
    void execute();

private:
    RenderStep findNextStep(std::set<RenderStep> const& finishedSteps) const;

    SimulationFacade _simulationFacade;
    std::vector<RenderStep> _steps;

    NumRenderObjects _numObjects;

    GeometrySource _geometrySource;
};
