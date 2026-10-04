#pragma once

#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <variant>

#include <Base/MathTypes.h>

#include <Shaders/ShaderSources.h>

#include <Data/SimulationParameters.h>

#include <EngineInterface/Definitions.h>

#include "Definitions.h"
#include "Shader.h"
#include "VulkanContext.h"
#include "VulkanGeometryBuffers.h"

// Color and depth image with the same memory layout as an OpenGL framebuffer, i.e. its rows are ordered from bottom to top
struct _TextureTarget
{
    static TextureTarget create();
    ~_TextureTarget();

    void resize(IntVector2D const& size, VkFormat colorFormat);

    // Created on first use since only a few render steps test depth, afterwards it follows the size of the color image
    VulkanImage& getDepth();

    VulkanImage color;

private:
    _TextureTarget() = default;

    void destroyImages();

    VulkanImage _depth;
};
struct ScreenTarget
{
    auto operator<=>(ScreenTarget const&) const = default;
    bool operator==(ScreenTarget const&) const = default;

    // Image is provided by the render pipeline
};
using RenderTarget = std::variant<ScreenTarget, TextureTarget>;

struct GeneralRenderInfo
{
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
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
    MEMBER(ExecutionParameters, VulkanGeometryBuffers, geometryBuffers, VulkanGeometryBuffers());
    MEMBER(ExecutionParameters, std::vector<TextureTarget>, textures, {});
    MEMBER(ExecutionParameters, bool, clearBackground, false);

    // Output
    MEMBER(ExecutionParameters, TextureTarget, target, TextureTarget());

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
    _RenderStep(StepParameters const& parameters, DepthTest depthTest = DepthTest::None);

    // Starts rendering into the target, the sampled textures must not be the target
    void prepareExecution(ExecutionParameters const& parameters, std::vector<TextureTarget> const& sampledTextures = {});
    void finishExecution(ExecutionParameters const& parameters);

    // Draws the vertices or, if an index buffer is given, the indexed vertices into the target
    void draw(ExecutionParameters const& parameters, PipelineState state, VkBuffer vertexBuffer, uint64_t numElements, VkBuffer indexBuffer = VK_NULL_HANDLE);

    Shader _shader;
    std::optional<int> _previousTargetSelection;
    float _textureScale = 1.0f;
    UniformValueMap _uniforms;
    std::function<UniformValueMap(SimulationParameters const&)> _uniformFunc;
    DepthTest _depthTest = DepthTest::None;
    std::vector<TextureTarget> _inputTextures;

public:
    std::vector<TextureTarget> const& getInputTextures() const { return _inputTextures; }
};

class _NonFluidObjectRenderStep : public _RenderStep
{
public:
    static CellRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _NonFluidObjectRenderStep(StepParameters const& parameters);
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

struct FullscreenQuad;

class _PostProcessingRenderStep : public _RenderStep
{
public:
    static PostProcessingRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _PostProcessingRenderStep(StepParameters const& parameters);

    std::shared_ptr<FullscreenQuad> _fullscreenQuad;
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

class _FluidParticleRenderStep : public _RenderStep
{
public:
    static FluidParticleRenderStep create(StepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _FluidParticleRenderStep(StepParameters const& parameters);
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

    VulkanImage _cellTypeTextureAtlas;
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
    ~_DetonationEventRenderStep();

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _DetonationEventRenderStep(StepParameters const& parameters);

    void updateDetonations(GeometryBuffers const& geometryBuffers, std::chrono::steady_clock::time_point now);

    struct Detonation
    {
        RealVector2D pos;
        float radius = 0;
        std::chrono::steady_clock::time_point startTime;
        bool reported = false;
    };
    std::unordered_map<uint64_t, Detonation> _detonations;

    VulkanBuffer _instanceBuffer;
};
