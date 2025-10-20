#pragma once

#include <variant>
#include <filesystem>

#include "Base/MathTypes.h"
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

using UniformValueType = std::variant<int, float, FloatColorRGB>;
using UniformValueMap = std::map<std::string, UniformValueType>;

struct StepParameters
{
    MEMBER(StepParameters, std::filesystem::path, shader, std::filesystem::path());
    MEMBER(StepParameters, std::optional<int>, previousTargetSelection, std::nullopt);
    MEMBER(StepParameters, float, textureScale, 1.0f);
    MEMBER(StepParameters, bool, preventMoirePatterns, true);
    MEMBER(StepParameters, UniformValueMap, uniforms, {});
    MEMBER(StepParameters, std::function<UniformValueMap()>, uniformFunc, {});
};

struct ExecutionParameters
{
    // Input
    MEMBER(ExecutionParameters, GeometryBuffers, geometryBuffers, GeometryBuffers());
    MEMBER(ExecutionParameters, std::vector<unsigned int>, textures, {});
    MEMBER(ExecutionParameters, bool, clearBackground, false);

    // Output
    MEMBER(ExecutionParameters, RenderTarget, target, ScreenTarget());
    MEMBER(ExecutionParameters, float, scale, 1.0f);

    // Misc
    MEMBER(ExecutionParameters, GeneralRenderInfo, renderInfo, GeneralRenderInfo());
    MEMBER(ExecutionParameters, SimulationFacade, simulationFacade, SimulationFacade());
};

class _RenderStep
{
public:
    virtual ~_RenderStep() = default;

    virtual void execute(ExecutionParameters parameters) = 0;

    std::optional<int> const& getPreviousTargetSelection() const;

    TextureTarget const& getTextureTarget() const;
    void setTextureTarget(TextureTarget const& target);

    void resize(IntVector2D const& size);

    float getTextureScale() const;
    void setTextureScale(float scale);

protected:
    _RenderStep(StepParameters const& parameters);

    void prepareExecution(ExecutionParameters const& parameters);

    Shader _shader;
    std::optional<int> _previousTargetSelection;
    TextureTarget _target;
    float _textureScale = 1.0f;
    bool _preventMoirePatterns = true;
    UniformValueMap _uniforms;
    std::function<UniformValueMap()> _uniformFunc;
};

class _CellRenderStep : public _RenderStep
{
public:
    static CellRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _CellRenderStep(StepParameters const& parameters);
};

class _LineRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static LineRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _LineRenderStep(StepParameters const& parameters);
};

class _TriangleRenderStep : public _RenderStep
{
public:
    static TriangleRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _TriangleRenderStep(StepParameters const& parameters);
};

class _PostProcessingRenderStep : public _RenderStep
{
public:
    static PostProcessingRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

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
    void execute(ExecutionParameters parameters) override;

private:
    _ForwardRenderStep(StepParameters const& parameters);
};

class _EnergyParticleRenderStep : public _RenderStep
{
public:
    static EnergyParticleRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _EnergyParticleRenderStep(StepParameters const& parameters);
};

class _ZoneRenderStep : public _RenderStep
{
public:
    static ZoneRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _ZoneRenderStep(StepParameters const& parameters);
};
