#pragma once

#include <variant>
#include <filesystem>

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
    auto operator<=>(ScreenTarget const&) const = default;

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

    template<typename T>
    void setUniform(std::string const& name, T value)
    {
        _uniformValues.insert_or_assign(name, value);
    }

protected:
    _RenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);

    bool isUsePreviousOutput() const;
    std::optional<RenderTarget> const& getTarget() const;
    void setTarget(RenderTarget const& target);

    void setClearBackground(bool value);

    void prepareExecution(GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

    Shader _shader;
    bool _usePreviousOutput;
    std::optional<RenderTarget> _target;
    bool _clearBackground = false;

    std::map<std::string, std::variant<int, float>> _uniformValues;
};

class _PointRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static PointRenderStep create(std::filesystem::path const& shaderFilename, bool usePreviousOutput = false);

protected:
    void execute(GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _PointRenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);
};

class _LineRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static LineRenderStep create(std::filesystem::path const& shaderFilename, bool usePreviousOutput = false);

protected:
    void execute(GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _LineRenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);
};

class _TriangleRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static TriangleRenderStep create(std::filesystem::path const& shaderFilename, bool usePreviousOutput = false);

protected:
    void execute(GeometryBuffers const& geometryBuffers, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

private:
    _TriangleRenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);
};

class _PostProcessingRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static PostProcessingRenderStep create(std::filesystem::path const& shaderFilename);

protected:
    void execute(GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade, std::vector<RenderStep> const& dependentSteps);

private:
    _PostProcessingRenderStep(std::filesystem::path const& shaderFilename);

    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _ebo = 0;
};
