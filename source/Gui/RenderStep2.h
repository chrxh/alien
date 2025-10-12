#pragma once

#include <variant>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

struct _TextureTarget
{
    bool initialized = false;
    unsigned int fbo = 0;
    unsigned int texture = 0;
};
using TextureTarget = std::shared_ptr<_TextureTarget>;
struct ScreenTarget
{
    // FBO is automatically determined
};
using RenderTarget = std::variant<ScreenTarget, TextureTarget>;

struct GeneralRenderInfo
{
    int screenFbo = 0;
};

class _RenderStep
{
public:
    virtual ~_RenderStep() = default;

    std::vector<RenderStep> const& getDependentSteps() const;
    RenderTarget const& getTarget() const;

protected:
    _RenderStep(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps);

    void activateShader(SimulationFacade const& simulationFacade);

    Shader _shader;
    RenderTarget _target;
    std::vector<RenderStep> _dependentSteps;
};

class _PointRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static PointRenderStep create(Shader const& shader, RenderTarget const& target);

protected:
    void execute(uint64_t const& numVertices, GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _PointRenderStep(Shader const& shader, RenderTarget const& target);
};

class _LineRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static LineRenderStep create(Shader const& shader, RenderTarget const& target, RenderStep const& dependentStep);

protected:
    void execute(uint64_t const& numLines, GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _LineRenderStep(Shader const& shader, RenderTarget const& target, RenderStep const& dependentStep);
};
