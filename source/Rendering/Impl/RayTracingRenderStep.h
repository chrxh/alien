#pragma once

#include <array>

#include <Engine/Interface/GeometryBuffers.h>

#include "RayTracingComputePipeline.h"
#include "RenderStep.h"

// Renders the creatures with hardware ray tracing as living tissue under a microscope, see Shaders/RayTracingTissueCS.h, and the fluids as
// gas around it, see Shaders/RayTracingGasShadingCS.h. Regions with a high energy density radiate light, which forms columns of light and
// shadow behind the objects on its way, see Shaders/RayTracingGasIrradianceCS.h. The inputs are the background and the energy particles,
// the output is the light in front of the background in high dynamic range, to which the background is added after the tone mapping as in
// the rasterized view, and in the alpha channel the strength of the energy light for the rasterized objects. Each frame is computed
// completely, only still pictures average several frames for the anti-aliasing.
class _RayTracingRenderStep : public _RenderStep
{
public:
    static RayTracingRenderStep create(StepParameters const& parameters);
    ~_RayTracingRenderStep() override;

    // Zoomed out, the rasterized objects replace the tissue, so that the view keeps the details and the brightness of the rasterized view
    static float getTissueVisibility(RenderView const& view);

    // The background is scaled by the inverse of the factor with which the bloom weights the scene
    static float getBackgroundScale(SimulationParameters const& parameters);

    // The objects form a scene for the tissue and for the shadows of the energy light
    static bool isSceneNeeded(RenderView const& view, SimulationParameters const& parameters);

    // Allocates the resources for the view and the visible objects. Calling it before the commands are recorded lets a lack of memory
    // surface without leaving a partially recorded frame behind.
    void prepareResources(RenderView const& view, NumRenderObjects const& numObjects, SimulationParameters const& parameters);

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

    // Grid in world coordinates whose cells collect the energy of the objects. Its cells have the same size at all zoom levels unless the
    // grid would get too large, so that the regions with a high energy density stay the same.
    struct EnergyGrid
    {
        RealVector2D origin;
        float cellSize = 1.0f;
        IntVector2D size;
    };

    void createPipelines();
    void createGasResources();
    void updateLightGrid(RealRect const& visibleWorldRect);
    void updateGasGrid(RenderView const& view);
    void updateEnergyGrid();

    void updateFrameConstants(RenderView const& view, NumRenderObjects const& numObjects, SimulationParameters const& parameters, IntVector2D const& worldSize);
    VkDescriptorSet writeDescriptorSet(VulkanGeometryBuffers const& geometryBuffers, std::vector<TextureTarget> const& textures);
    void buildScene(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects);
    void buildGas(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects, bool energyLightActive);
    void shade(VkCommandBuffer commandBuffer);
    void present(ExecutionParameters const& parameters);

    int getPrevious() const;
    bool isSameViewAsPrevious(RenderView const& view) const;

    VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
    RayTracingComputePipeline _preparePipeline;
    RayTracingComputePipeline _allocatePipeline;
    RayTracingComputePipeline _scatterPipeline;
    RayTracingComputePipeline _gasSplatPipeline;
    RayTracingComputePipeline _energyGridPipeline;
    RayTracingComputePipeline _gasLightsPipeline;
    RayTracingComputePipeline _gasIrradiancePipeline;
    RayTracingComputePipeline _gasBlurHorizontalPipeline;
    RayTracingComputePipeline _gasBlurVerticalPipeline;
    RayTracingComputePipeline _tissuePipeline;
    RayTracingComputePipeline _antiAliasingPipeline;
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

    EnergyGrid _energyGrid;
    VulkanBuffer _energyCells;

    bool _stillPicture = false;
    bool _previousFrameValid = false;
    RenderView _previousView;
    uint32_t _numAveragedFrames = 0;
};
