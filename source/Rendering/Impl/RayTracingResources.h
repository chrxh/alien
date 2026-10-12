#pragma once

#include <array>
#include <utility>
#include <vector>

#include <Engine/Interface/GeometryBuffers.h>

#include "RayTracingComputePipeline.h"
#include "RenderStep.h"

// Resources of the ray tracing steps, see RayTracingSteps.h, which the steps access by these names
enum class RayTracingResource
{
    Spheres,
    SphereAabbs,
    CapsuleAabbs,
    TriangleAabbs,
    CellCounts,
    CellOffsets,
    CellCursors,
    CellEntries,
    Allocation,
    GasAccumulation,
    GasIntermediate,
    GasLightClusters,
    EnergyCells,

    // Top level acceleration structure
    Scene,

    GasImage,
    GasIrradiance,
    SceneColor,
    AntiAliased,
    PreviousAntiAliased,
};

enum class RayTracingAccess
{
    Read,
    ReadWrite,

    // Clears the used part of a buffer to zero before the step reads and writes it
    Clear,
};

using RayTracingAccesses = std::vector<std::pair<RayTracingResource, RayTracingAccess>>;

// Holds the resources that the ray tracing steps share within a frame and across frames. All compute steps use the same descriptor set, which
// refers to all resources and is written once per frame.
class _RayTracingResources
{
public:
    static RayTracingResources create();
    ~_RayTracingResources();

    // Zoomed out, the rasterized objects replace the tissue, so that the view keeps the details and the brightness of the rasterized view
    static float getTissueVisibility(RenderView const& view);

    // The background is scaled by the inverse of the factor with which the bloom weights the scene
    static float getBackgroundScale(SimulationParameters const& parameters);

    static bool isEnergyLightActive(SimulationParameters const& parameters);

    // The objects form a scene for the tissue and for the shadows of the energy light
    static bool isSceneNeeded(RenderView const& view, SimulationParameters const& parameters);

    // Allocates the resources for the view and the visible objects. Calling it before the commands are recorded lets a lack of memory
    // surface without leaving a partially recorded frame behind.
    void prepare(RenderView const& view, NumRenderObjects const& numObjects, SimulationParameters const& parameters);

    // Successive frames of a still picture are averaged with rays jittered within the pixels
    void setStillPicture(bool value);

    void resetImageStates();

    // The pipelines share the layout of the descriptor set
    RayTracingComputePipeline createPipeline(std::string const& source, IntVector2D const& workGroupSize) const;

    // Called by the first and the last step of a frame. The textures of the first step are the ground and the energy particles.
    void beginFrame(ExecutionParameters const& parameters);
    void finishFrame(RenderView const& view);

    void bindDescriptorSet(VkCommandBuffer commandBuffer) const;

    // Records the clears and the barriers that the accesses of a step require
    void access(VkCommandBuffer commandBuffer, RayTracingAccesses const& accesses);

    // Builds the acceleration structures from the boxes of the objects
    void buildScene(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects);

    // Anti-aliased image of the current frame
    VulkanImage& getOutput();

    // Uniform grid in world coordinates whose cells list the glowing objects inside
    struct LightGrid
    {
        RealVector2D origin;
        float cellSize = 1.0f;
        IntVector2D size;

        uint64_t getNumCells() const;
    };
    LightGrid const& getLightGrid() const;

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
    GasGrid const& getGasGrid() const;

    IntVector2D const& getViewSize() const;

private:
    _RayTracingResources();

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

    // Grid in world coordinates whose cells collect the energy of the objects. Its cells have the same size at all zoom levels unless the
    // grid would get too large, so that the regions with a high energy density stay the same.
    struct EnergyGrid
    {
        RealVector2D origin;
        float cellSize = 1.0f;
        IntVector2D size;
    };

    void createFrameResources();
    void updateLightGrid(RealRect const& visibleWorldRect);
    void updateGasGrid(RenderView const& view);
    void updateEnergyGrid();

    void updateFrameConstants(RenderView const& view, NumRenderObjects const& numObjects, SimulationParameters const& parameters, IntVector2D const& worldSize);
    void writeDescriptorSet(VulkanGeometryBuffers const& geometryBuffers, std::vector<TextureTarget> const& textures);

    VulkanBuffer& getBuffer(RayTracingResource resource);
    VulkanImage* getImage(RayTracingResource resource);
    VkDeviceSize getUsedSize(RayTracingResource resource) const;

    int getPrevious() const;
    bool isSameViewAsPrevious(RenderView const& view) const;

    VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet _descriptorSet = VK_NULL_HANDLE;
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
