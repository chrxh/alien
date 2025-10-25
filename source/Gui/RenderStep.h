#pragma once

#include <filesystem>
#include <variant>

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

    float getTextureScaling() const;
    void setTextureScaling(float scale);

protected:
    _RenderStep(StepParameters const& parameters);

    void prepareExecution(ExecutionParameters const& parameters);

    Shader _shader;
    std::optional<int> _previousTargetSelection;
    float _textureScale = 1.0f;
    bool _preventMoirePatterns = true;
    UniformValueMap _uniforms;
    std::function<UniformValueMap()> _uniformFunc;
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

class _SelectedCellRenderStep : public _RenderStep
{
public:
    static SelectedCellRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _SelectedCellRenderStep(StepParameters const& parameters);
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