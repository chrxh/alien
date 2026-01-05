#pragma once

#include <filesystem>
#include <variant>

#include <Base/MathTypes.h>

#include <Shaders/ShaderSources.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/SimulationParameters.h>

#include "Definitions.h"

struct _TextureTarget
{
    static TextureTarget create();

    bool initialized = false;
    unsigned int fbo = 0;
    unsigned int texture = 0;
    unsigned int depthBuffer = 0;

private:
    _TextureTarget() = default;
};
struct ScreenTarget
{
    auto operator<=>(ScreenTarget const&) const = default;
    bool operator==(ScreenTarget const&) const = default;

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
    MEMBER(StepParameters, ShaderSources::ShaderSource, shader, ShaderSources::ShaderSource());
    MEMBER(StepParameters, std::optional<int>, previousTargetSelection, std::nullopt);
    MEMBER(StepParameters, float, textureScale, 1.0f);
    MEMBER(StepParameters, UniformValueMap, uniforms, {});
    MEMBER(StepParameters, std::function<UniformValueMap(SimulationParameters const&)>, uniformFunc, {});

    StepParameters& addUniform(std::string const& key, UniformValueType const& value);
};

struct ExecutionParameters
{
    // Input
    MEMBER(ExecutionParameters, GeometryBuffers, geometryBuffers, GeometryBuffers());
    MEMBER(ExecutionParameters, std::vector<unsigned int>, textures, {});
    MEMBER(ExecutionParameters, bool, clearBackground, false);

    // Output
    MEMBER(ExecutionParameters, RenderTarget, target, ScreenTarget());

    // Misc
    MEMBER(ExecutionParameters, float, minBallRadius, 6.0f);
    MEMBER(ExecutionParameters, GeneralRenderInfo, renderInfo, GeneralRenderInfo());
    MEMBER(ExecutionParameters, SimulationFacade, simulationFacade, SimulationFacade());
    MEMBER(ExecutionParameters, std::shared_ptr<SimulationParameters>, simulationParameters, nullptr);
};

class _RenderStep
{
public:
    virtual ~_RenderStep() = default;

    virtual void execute(ExecutionParameters parameters) = 0;

    std::optional<int> const& getPreviousTargetSelection() const;

    float getTextureScaling() const;
    void setTextureScaling(float scale);

protected:
    _RenderStep(StepParameters const& parameters);

    void prepareExecution(ExecutionParameters const& parameters);

    Shader _shader;
    std::optional<int> _previousTargetSelection;
    float _textureScale = 1.0f;
    UniformValueMap _uniforms;
    std::function<UniformValueMap(SimulationParameters const&)> _uniformFunc;
    std::vector<unsigned int> _inputTextures;

public:
    std::vector<unsigned int> const& getInputTextures() const { return _inputTextures; }
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

class _LocationRenderStep : public _RenderStep
{
public:
    static LocationRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _LocationRenderStep(StepParameters const& parameters);
};

class _SelectedObjectRenderStep : public _RenderStep
{
public:
    static SelectedObjectRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _SelectedObjectRenderStep(StepParameters const& parameters);
};

class _CellTypeOverlayRenderStep : public _RenderStep
{
public:
    static CellTypeOverlayRenderStep create(StepParameters const& parameters);
    ~_CellTypeOverlayRenderStep();

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _CellTypeOverlayRenderStep(StepParameters const& parameters);

    void createCellTypeTextureAtlas();

    unsigned int _cellTypeTextureAtlas = 0;
};

class _SelectedConnectionRenderStep : public _RenderStep
{
public:
    static SelectedConnectionRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _SelectedConnectionRenderStep(StepParameters const& parameters);
};

class _AttackEventRenderStep : public _RenderStep
{
public:
    static AttackEventRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _AttackEventRenderStep(StepParameters const& parameters);
};

class _DetonationEventRenderStep : public _RenderStep
{
public:
    static DetonationEventRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _DetonationEventRenderStep(StepParameters const& parameters);
};
