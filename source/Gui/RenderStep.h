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
    friend _RenderPipeline;
    friend _PointRenderStep;
    friend _PostProcessingRenderStep;
    friend _LineRenderStep;
    friend _TriangleRenderStep;

public:
    virtual ~_RenderStep() = default;

    std::vector<RenderStep> const& getDependentSteps() const;

    template<typename T>
    void setUniform(std::string const& name, T value)
    {
        _uniformValues.insert_or_assign(name, value);
    }

protected:
    _RenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps);

    std::optional<RenderTarget> const& getTarget() const;
    void setTarget(RenderTarget const& target);
    void activateShader(SimulationFacade const& simulationFacade);
    void setFramebuffer(GeneralRenderInfo const& renderInfo);

    Shader _shader;
    std::optional<RenderTarget> _target;
    std::vector<RenderStep> _dependentSteps;

    std::map<std::string, std::variant<int>> _uniformValues;
};

class _PointRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static PointRenderStep create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps = {});
    static PointRenderStep create(Shader const& shader, std::vector<RenderStep> const& dependentSteps = {});

protected:
    void execute(uint64_t const& numVertices, GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _PointRenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps);
};

class _LineRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static LineRenderStep create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps = {});
    static LineRenderStep create(Shader const& shader, std::vector<RenderStep> const& dependentSteps = {});

protected:
    void execute(uint64_t const& numLines, GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _LineRenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps);
};

class _TriangleRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static TriangleRenderStep create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps = {});
    static TriangleRenderStep create(Shader const& shader, std::vector<RenderStep> const& dependentSteps = {});

protected:
    void execute(uint64_t const& numTriangles, GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _TriangleRenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps);
};

class _PostProcessingRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static PostProcessingRenderStep create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps);
    static PostProcessingRenderStep create(Shader const& shader, std::vector<RenderStep> const& dependentSteps);

protected:
    void execute(GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _PostProcessingRenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps);

    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _ebo = 0;
};
