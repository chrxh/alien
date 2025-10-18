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
    // FBO is automatically determined
};
using RenderTarget = std::variant<ScreenTarget, TextureTarget>;

struct GeneralRenderInfo
{
    int screenFbo = 0;
};

using UniformValueType = std::variant<int, float>;
using UniformValueMap = std::map<std::string, UniformValueType>;

struct StepParameters
{
    MEMBER(StepParameters, std::filesystem::path, shader, std::filesystem::path());
    MEMBER(StepParameters, std::optional<int>, previousTargetSelection, std::nullopt);
    MEMBER(StepParameters, UniformValueMap, uniformValues, {});
};

class _RenderStep
{
public:
    virtual ~_RenderStep() = default;

    virtual void execute(
        // Input
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,

        // Output
        RenderTarget const& target,

        // Misc
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) = 0;

    std::optional<int> const& getPreviousTargetSelection() const;

    std::optional<TextureTarget> const& getTextureTarget() const;
    void setTextureTarget(TextureTarget const& target);

protected:
    _RenderStep(StepParameters const& parameters);

    void prepareExecution(bool clearBackground, RenderTarget const& target, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

    Shader _shader;
    std::optional<int> _previousTargetSelection;
    std::optional<TextureTarget> _target;

    UniformValueMap _uniformValues;
};

class _PointRenderStep : public _RenderStep
{
public:
    static PointRenderStep create(StepParameters const& parameters);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _PointRenderStep(StepParameters const& parameters);
};

class _LineRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static LineRenderStep create(StepParameters const& parameters);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _LineRenderStep(StepParameters const& parameters);
};

class _TriangleRenderStep : public _RenderStep
{
public:
    static TriangleRenderStep create(StepParameters const& parameters);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _TriangleRenderStep(StepParameters const& parameters);
};

class _PostProcessingRenderStep : public _RenderStep
{
public:
    static PostProcessingRenderStep create(StepParameters const& parameters);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _PostProcessingRenderStep(StepParameters const& parameters);

    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _ebo = 0;
};

class _ForwardRenderStep : public _RenderStep
{
public:
    static ForwardRenderStep create(StepParameters const& parameters);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _ForwardRenderStep(StepParameters const& parameters);
};
