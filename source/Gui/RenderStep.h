#pragma once

#include <variant>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

struct _TextureTarget
{
    static TextureTarget create();

    bool initialized = false;
    unsigned int fbo = 0;
    unsigned int texture = 0;

private: 
    _TextureTarget() = default;
};
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

    template<typename T>
    void setUniform(std::string const& name, T value)
    {
        _uniformValues.insert_or_assign(name, value);
    }

protected:
    _RenderStep(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps);

    void activateShader(SimulationFacade const& simulationFacade);

    Shader _shader;
    RenderTarget _target;
    std::vector<RenderStep> _dependentSteps;

    std::map<std::string, std::variant<int>> _uniformValues;
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

class _PostProcessingRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static PostProcessingRenderStep create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps);

protected:
    void execute(GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _PostProcessingRenderStep(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps);

    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _ebo = 0;
};
