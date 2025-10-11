#pragma once

#include <variant>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

struct GeometrySource
{
    static GeometrySource create();

    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;

private:
    GeometrySource() = default;
};
struct TextureSource
{
    // Automatically determined by dependent steps
};
using RenderSource = std::variant<GeometrySource, TextureSource>;

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
    RenderSource const& getSource() const;
    RenderTarget const& getTarget() const;

protected:
    _RenderStep(Shader const& shader, RenderSource const& source, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps);

    void activateShader(SimulationFacade const& simulationFacade);

    Shader _shader;
    RenderSource _source;
    RenderTarget _target;
    std::vector<RenderStep> _dependentSteps;
};

class _PointRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static PointRenderStep create(Shader const& shader, GeometrySource const& source, RenderTarget const& target);

protected:
    void execute(uint64_t const& numVertices, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _PointRenderStep(Shader const& shader, GeometrySource const& source, RenderTarget const& target);
};

class _LineRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static LineRenderStep create(Shader const& shader, GeometrySource const& source, RenderTarget const& target, RenderStep const& dependentStep);

protected:
    void execute(uint64_t const& numLines, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _LineRenderStep(Shader const& shader, GeometrySource const& source, RenderTarget const& target, RenderStep const& dependentStep);
};
