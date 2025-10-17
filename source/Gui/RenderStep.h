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
public:
    virtual ~_RenderStep() = default;

    template<typename T>
    void setUniform(std::string const& name, T value)
    {
        _uniformValues.insert_or_assign(name, value);
    }

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

    bool isUsePreviousTarget() const;

    std::optional<TextureTarget> const& getTextureTarget() const;
    void setTextureTarget(TextureTarget const& target);

protected:
    _RenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);
    _RenderStep(bool usePreviousOutput);

    void prepareExecution(bool clearBackground, RenderTarget const& target, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade);

    Shader _shader;
    bool _usePreviousOutput;
    std::optional<TextureTarget> _target;

    std::map<std::string, std::variant<int, float>> _uniformValues;
};

class _PointRenderStep : public _RenderStep
{
public:
    static PointRenderStep create(std::filesystem::path const& shaderFilename, bool usePreviousOutput = false);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _PointRenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);
};

class _LineRenderStep : public _RenderStep
{
    friend _RenderPipeline;

public:
    static LineRenderStep create(std::filesystem::path const& shaderFilename, bool usePreviousOutput = false);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _LineRenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);
};

class _TriangleRenderStep : public _RenderStep
{
public:
    static TriangleRenderStep create(std::filesystem::path const& shaderFilename, bool usePreviousOutput = false);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _TriangleRenderStep(std::filesystem::path const& shaderFilename, bool usePreviousOutput);
};

class _PostProcessingRenderStep : public _RenderStep
{
public:
    static PostProcessingRenderStep create(std::filesystem::path const& shaderFilename);

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _PostProcessingRenderStep(std::filesystem::path const& shaderFilename);

    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _ebo = 0;
};

class _ForwardRenderStep : public _RenderStep
{
public:
    static ForwardRenderStep create();

protected:
    void execute(
        GeometryBuffers const& geometryBuffers,
        std::vector<unsigned int> const& textures,
        bool clearBackground,
        RenderTarget const& target,
        GeneralRenderInfo const& renderInfo,
        SimulationFacade const& simulationFacade) override;

private:
    _ForwardRenderStep();
};
