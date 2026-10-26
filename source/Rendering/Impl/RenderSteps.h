#pragma once

#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <variant>

#include <imgui.h>

#include <Base/MathTypes.h>

#include <Rendering/Impl/Shaders/ShaderSources.h>

#include <Data/SimulationParameters.h>

#include <Engine/Interface/Definitions.h>

#include <Rendering/Interface/RenderView.h>

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

    // Required if the recorded commands are discarded instead of submitted. The contents become undefined.
    void resetImageStates();

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

    // Image is provided by the render graph
};
using RenderTarget = std::variant<ScreenTarget, TextureTarget>;

struct GeneralRenderInfo
{
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
};

using UniformValueType = std::variant<int, float, FloatColorRGB>;
using UniformValueMap = std::map<std::string, UniformValueType>;
using UniformFunc = std::function<UniformValueMap(SimulationParameters const&, RenderView const&)>;

// Determines from the uniforms of a step whether its output equals its first input in a frame
using PassThroughFunc = std::function<bool(UniformValueMap const&, RenderView const&)>;

// Region within the target outside of which the output of a step is black, depending on the repetition of its sequence
using RenderAreaFunc = std::function<std::optional<VkRect2D>(UniformValueMap const&, RenderView const&, int repetition)>;

struct StepParameters
{
    MEMBER(StepParameters, ShaderSources::ShaderSource, shader, ShaderSources::ShaderSource());
    MEMBER(StepParameters, std::optional<int>, previousTargetSelection, std::nullopt);
    MEMBER(StepParameters, float, textureScale, 1.0f);
    MEMBER(StepParameters, UniformValueMap, uniforms, {});
    MEMBER(StepParameters, UniformFunc, uniformFunc, {});

    // A step that passes its input through is skipped and hands it on to the next step
    MEMBER(StepParameters, PassThroughFunc, passThroughFunc, {});

    // A step with a render area only renders inside of it and clears the rest of the target to black
    MEMBER(StepParameters, RenderAreaFunc, renderAreaFunc, {});

    StepParameters& addUniform(std::string const& key, UniformValueType const& value);
};

struct ExecutionParameters
{
    // Input
    MEMBER(ExecutionParameters, RenderView, view, RenderView());
    MEMBER(ExecutionParameters, VulkanGeometryBuffers, geometryBuffers, VulkanGeometryBuffers());
    MEMBER(ExecutionParameters, std::vector<TextureTarget>, textures, {});
    MEMBER(ExecutionParameters, bool, clearBackground, false);
    MEMBER(ExecutionParameters, int, repetition, 0);

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

    // Steps without target compute into resources outside the render graph, which hands their input textures on to the next step
    virtual bool hasTarget() const { return true; }

    std::optional<int> const& getPreviousTargetSelection() const;

    bool isPassThrough(SimulationParameters const& parameters, RenderView const& view) const;

protected:
    _RenderStep(StepParameters const& parameters, DepthTest depthTest = DepthTest::None);

    UniformValueMap getUniforms(SimulationParameters const& parameters, RenderView const& view) const;

    // Starts rendering into the target, the sampled textures must not be the target
    void prepareExecution(ExecutionParameters const& parameters, std::vector<TextureTarget> const& sampledTextures = {});
    void finishExecution(ExecutionParameters const& parameters);

    // Draws the vertices or, if an index buffer is given, the indexed vertices into the target
    void draw(ExecutionParameters const& parameters, PipelineState state, VkBuffer vertexBuffer, uint64_t numElements, VkBuffer indexBuffer = VK_NULL_HANDLE);

    Shader _shader;
    std::optional<int> _previousTargetSelection;
    float _textureScale = 1.0f;
    UniformValueMap _uniforms;
    UniformFunc _uniformFunc;
    PassThroughFunc _passThroughFunc;
    RenderAreaFunc _renderAreaFunc;
    DepthTest _depthTest = DepthTest::None;
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
    static CellTypeOverlayRenderStep create(StepParameters const& parameters, ImFont* labelFont);
    ~_CellTypeOverlayRenderStep();

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _CellTypeOverlayRenderStep(StepParameters const& parameters, ImFont* labelFont);

    void createCellTypeTextureAtlas(ImFont* labelFont);

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
