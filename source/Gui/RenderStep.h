//#pragma once
//
//#include <filesystem>
//#include <variant>
//
//#include "EngineInterface/Definitions.h"
//
//#include "Definitions.h"
//
//struct TextureTarget
//{};
//struct ScreenTarget
//{
//    int fbo = 0;
//};
//using RenderTarget = std::variant<ScreenTarget, TextureTarget>;
//
//struct _TargetData
//{
//    unsigned int outputTexture = 0;
//    unsigned int fbo = 0;
//};
//using TargetData = std::shared_ptr<_TargetData>;
//
//class _RenderStep
//{
//    friend _RenderPipeline;
//
//public:
//    virtual ~_RenderStep() = default;
//
//    std::vector<RenderStep> const& getDependentSteps() const;
//    Shader getShader() const;
//    TargetData getTargetData() const;
//    unsigned int getTexture() const;
//    unsigned int getFbo() const;
//
//    void setBool(std::string const& name, bool value);
//
//protected:
//    _RenderStep(Shader const& shader, TargetData const& targetData, std::vector<RenderStep> const& dependentSteps);
//
//    virtual void resize(IntVector2D const& size);
//    virtual void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) = 0;
//
//    void activateShader(SimulationFacade const& simulationFacade);
//
//    std::vector<RenderStep> _dependentSteps;
//
//    Shader _shader;
//    bool _sharedTarget = false;
//    TargetData _targetData;
//
//    std::map<std::string, bool> _boolValues;
//};
//
//class _PointRenderStep : public _RenderStep
//{
//public:
//    static PointRenderStep create(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader);
//    static PointRenderStep
//    createWithSharedVbo(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, RenderStep const& sharedStep);
//
//protected:
//    void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) override;
//
//private:
//    _PointRenderStep(Shader const& shader);
//};
//
//class _LineRenderStep : public _RenderStep
//{
//public:
//    static LineRenderStep
//    createWithSharedVboAndTarget(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, RenderStep const& sharedStep);
//
//protected:
//    void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) override;
//
//private:
//    _LineRenderStep(Shader const& shader, TargetData const& targetData, RenderStep const& dependentStep);
//};
//
//class _PostProcessingRenderStep : public _RenderStep
//{
//public:
//    static PostProcessingRenderStep
//    create(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, std::vector<RenderStep> const& dependentSteps);
//
//protected:
//    void execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade) override;
//
//private:
//    _PostProcessingRenderStep(Shader const& shader, std::vector<RenderStep> const& dependentSteps);
//};
