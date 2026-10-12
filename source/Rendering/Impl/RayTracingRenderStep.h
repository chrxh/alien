#pragma once

#include <array>

#include <Engine/Interface/GeometryBuffers.h>

#include "ComputePipeline.h"
#include "RenderStep.h"

// Renders the creatures with hardware ray tracing as living tissue under a microscope, see Shaders/RayTracingTissueCS.h, and the fluids as
// gas around it, see Shaders/RayTracingGasShadingCS.h. The inputs are the background and the energy particles, the output is the light in
// front of the background in high dynamic range, to which the background is added after the tone mapping as in the rasterized view. Each
// frame is computed completely, only still pictures average several frames for the anti-aliasing.
class _RayTracingRenderStep : public _RenderStep
{
public:
    static RayTracingRenderStep create(StepParameters const& parameters);
    ~_RayTracingRenderStep() override;

    // Zoomed out, the rasterized objects replace the tissue, so that the view keeps the details and the brightness of the rasterized view
    static float getTissueVisibility(RenderView const& view);

    // Allocates the resources for the view and the visible objects. Calling it before the commands are recorded lets a lack of memory
    // surface without leaving a partially recorded frame behind.
    void prepareResources(RenderView const& view, NumRenderObjects const& numObjects);

    // Successive frames of a still picture are averaged with rays jittered within the pixels
    void setStillPicture(bool value);

    void resetImageStates() override;

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingRenderStep(StepParameters const& parameters);

    // Buffers and acceleration structures for a maximum number of objects, connections and triangles
    struct SceneResources
    {
        uint64_t objectCapacity = 0;
        uint64_t capsuleCapacity = 0;
        uint64_t triangleCapacity = 0;
        VulkanBuffer spheres;
        VulkanBuffer sphereAabbs;
        VulkanBuffer capsuleAabbs;
        VulkanBuffer triangleAabbs;
        VulkanBuffer lightEntries;
        VulkanBuffer bottomLevelBuffer;
        VulkanBuffer topLevelBuffer;
        VulkanBuffer scratchBuffer;
        VulkanBuffer instanceBuffer;
        VkAccelerationStructureKHR bottomLevel = VK_NULL_HANDLE;
        VkAccelerationStructureKHR topLevel = VK_NULL_HANDLE;
    };
    static SceneResources createSceneResources(uint64_t objectCapacity, uint64_t capsuleCapacity, uint64_t triangleCapacity);
    static void destroySceneResources(SceneResources& resources, bool immediately);

    // Images of the view size, the anti-aliased images of the current and the previous frame alternate
    struct ViewResources
    {
        IntVector2D size;
        VulkanImage sceneColor;
        std::array<VulkanImage, 2> antiAliased;

        std::vector<VulkanImage*> getImages();
    };
    static ViewResources createViewResources(IntVector2D const& size);
    static void destroyViewResources(ViewResources& resources, bool immediately);

    // Uniform grid in world coordinates whose cells list the glowing objects inside
    struct LightGrid
    {
        RealVector2D origin;
        float cellSize = 1.0f;
        IntVector2D size;

        uint64_t getNumCells() const;
    };

    // Grid in world coordinates whose nodes hold the density of the gas. The nodes are aligned to the world and their distance is a power of
    // two multiple of a base distance, so that moving or zooming the view keeps the gas stable.
    struct GasGrid
    {
        RealVector2D origin;
        float nodeDistance = 1.0f;
        IntVector2D size;
        int blurRadius = 0;
        float cloudSpreadInNodes = 1.0f;

        // The light of the cells is collected in clusters and reaches the nodes of the irradiance grid
        float lightFalloffDistance = 1.0f;
        RealVector2D lightClusterOrigin;
        float lightClusterSize = 1.0f;
        IntVector2D numLightClusters;
        float irradianceNodeDistance = 1.0f;
        IntVector2D irradianceSize;
    };

    void createPipelines();
    void createGasResources();
    void updateLightGrid(RealRect const& visibleWorldRect);
    void updateGasGrid(RenderView const& view);

    void updateFrameConstants(RenderView const& view, NumRenderObjects const& numObjects, SimulationParameters const& parameters);
    VkDescriptorSet writeDescriptorSet(VulkanGeometryBuffers const& geometryBuffers, std::vector<TextureTarget> const& textures);
    void buildScene(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects);
    void buildGas(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects);
    void shade(VkCommandBuffer commandBuffer);
    void present(ExecutionParameters const& parameters);

    int getPrevious() const;
    bool isSameViewAsPrevious(RenderView const& view) const;

    VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
    ComputePipeline _preparePipeline;
    ComputePipeline _allocatePipeline;
    ComputePipeline _scatterPipeline;
    ComputePipeline _gasSplatPipeline;
    ComputePipeline _gasLightsPipeline;
    ComputePipeline _gasIrradiancePipeline;
    ComputePipeline _gasBlurHorizontalPipeline;
    ComputePipeline _gasBlurVerticalPipeline;
    ComputePipeline _tissuePipeline;
    ComputePipeline _antiAliasingPipeline;
    VulkanBuffer _frameConstants;

    SceneResources _scene;
    ViewResources _view;
    int _current = 0;

    LightGrid _lightGrid;
    uint64_t _cellCapacity = 0;
    VulkanBuffer _cellCounts;
    VulkanBuffer _cellOffsets;
    VulkanBuffer _cellCursors;
    VulkanBuffer _allocation;

    GasGrid _gasGrid;
    VulkanBuffer _gasAccumulation;
    VulkanBuffer _gasIntermediate;
    VulkanBuffer _gasLightClusters;
    VulkanImage _gasImage;
    VulkanImage _gasIrradiance;

    bool _stillPicture = false;
    bool _previousFrameValid = false;
    RenderView _previousView;
    uint32_t _numAveragedFrames = 0;
};
