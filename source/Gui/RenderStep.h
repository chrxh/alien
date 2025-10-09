#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include <filesystem>

class _RenderStep
{
public:
    _RenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, std::vector<RenderStep> const& dependentSteps);

    virtual void resize() = 0;
    virtual void execute(NumRenderObjects const& numObjects) = 0;

protected:
    std::vector<RenderStep> _dependentSteps;

    Shader _shader;
    unsigned int _outputTexture = 0;
    unsigned int _fbo = 0;
};

class _PointRenderStep : public _RenderStep
{
public:
    _PointRenderStep(
        std::filesystem::path const& vertexShader,
        std::filesystem::path const& fragmentShader,
        std::vector<RenderStep> const& dependentSteps,
        SimulationFacade const& simulationFacade);

    void resize() override;
    void execute(NumRenderObjects const& numObjects) override;

private:
    SimulationFacade _simulationFacade;
};

class _PostProcessingRenderStep : public _RenderStep
{
public:
    _PostProcessingRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader);

    void resize() override;
    void execute(NumRenderObjects const& numObjects) override;

private:
};
