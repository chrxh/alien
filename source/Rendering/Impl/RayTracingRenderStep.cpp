#include "RayTracingRenderStep.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ranges>

#include <Rendering/Impl/Shaders/ShaderSources.h>

#include "VulkanContext.h"

namespace
{
    auto constexpr MinCapacity = uint64_t(4096);
    auto constexpr LightRange = 3.0f;
    auto constexpr MaxLightGridCellsPerDimension = 512.0f;
    auto constexpr NumJitterPositions = 16u;
    auto constexpr Bioluminescence = 1.0f;

    // Between these screen zoom factors, the tissue fades in over the rasterized objects
    auto constexpr TissueZoomStart = 4.0f;
    auto constexpr TissueZoomEnd = 10.0f;

    // Between these screen zoom factors, the thin lines of the zoomed out view grow into the tissue
    auto constexpr CloseUpZoomStart = 3.0f;
    auto constexpr CloseUpZoomEnd = 12.0f;

    // Cell details such as walls and nuclei appear between these screen zoom factors
    auto constexpr DetailZoomStart = 8.0f;
    auto constexpr DetailZoomEnd = 20.0f;

    // The gas grid covers the view and a margin for the spread of the particles and the range of the lights. Its number of nodes and the sums
    // per node must match GasImageSize and GasAccumulationStride in Shaders/RayTracingCommonCS.h.
    auto constexpr MinGasNodeDistance = 0.25f;
    auto constexpr MaxGasNodes = 512;
    auto constexpr GasAccumulationStride = 5;
    auto constexpr GasLayers = 2;
    auto constexpr MaxGasBlurRadius = 32;

    // Standard deviation of the spread of the particles, like the soft disks of radius 7.5 in the rasterized view. Zoomed out, the spread
    // covers a minimum number of pixels.
    auto constexpr CloudSpread = 2.6f;
    auto constexpr MinCloudSpreadInPixels = 2.0f;

    // The cells light the gas within this range. Their light is collected in clusters of at least the given size and two node distances, the
    // irradiance grid has twice the node distance of the gas grid. The sizes must match GasLightClusterStride and GasIrradianceImageSize in
    // Shaders/RayTracingCommonCS.h.
    auto constexpr GasLightRange = 20.0f;
    auto constexpr MinLightClusterSize = 4.0f;
    auto constexpr MaxLightClusters = 256;
    auto constexpr GasLightClusterStride = 6;
    auto constexpr GasIrradianceImageSize = 256;

    auto constexpr SphereSize = VkDeviceSize(64);
    auto constexpr AabbBufferUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto constexpr ViewImageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    // Bindings declared in Shaders/RayTracingCommonCS.h
    enum Binding_ : uint32_t
    {
        Binding_FrameConstants,
        Binding_Scene,
        Binding_Objects,
        Binding_LineIndices,
        Binding_TriangleIndices,
        Binding_Spheres,
        Binding_SphereAabbs,
        Binding_CapsuleAabbs,
        Binding_TriangleAabbs,
        Binding_CellCounts,
        Binding_CellOffsets,
        Binding_CellCursors,
        Binding_CellEntries,
        Binding_Allocation,
        Binding_SceneColor,
        Binding_AntiAliased,
        Binding_PreviousAntiAliased,
        Binding_GroundTexture,
        Binding_GasSampler,
        Binding_SceneColorSampler,
        Binding_FluidParticles,
        Binding_GasAccumulation,
        Binding_GasIntermediate,
        Binding_GasImage,
        Binding_EnergyParticleTexture,
        Binding_GasLightClusters,
        Binding_GasIrradiance,
        Binding_GasIrradianceSampler,
        Binding_Count
    };

    VkDescriptorType getDescriptorType(uint32_t binding)
    {
        switch (binding) {
        case Binding_FrameConstants:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case Binding_Scene:
            return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        case Binding_SceneColor:
        case Binding_AntiAliased:
        case Binding_GasImage:
        case Binding_GasIrradiance:
            return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case Binding_PreviousAntiAliased:
        case Binding_GroundTexture:
        case Binding_GasSampler:
        case Binding_SceneColorSampler:
        case Binding_EnergyParticleTexture:
        case Binding_GasIrradianceSampler:
            return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        default:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
    }

    // Layout of the uniform block FrameConstants in Shaders/RayTracingCommonCS.h
    struct FrameConstants
    {
        float view[4];
        float rect[4];
        uint32_t counts[4];
        float lightGrid[4];
        int32_t gridSize[4];
        float tissue[4];
        float parameters[4];
        float gasGrid[4];
        int32_t gasSize[4];
        float gasLight[4];
        int32_t gasLightGrid[4];
    };
}

RayTracingRenderStep _RayTracingRenderStep::create(StepParameters const& parameters)
{
    return RayTracingRenderStep(new _RayTracingRenderStep(parameters));
}

_RayTracingRenderStep::~_RayTracingRenderStep()
{
    auto& context = VulkanContext::get();
    if (!context.isActive()) {
        return;
    }
    destroySceneResources(_scene, false);
    destroyViewResources(_view, false);
    for (auto buffer : {&_cellCounts, &_cellOffsets, &_cellCursors, &_allocation, &_frameConstants, &_gasAccumulation, &_gasIntermediate, &_gasLightClusters}) {
        context.destroyBufferLater(*buffer);
    }
    context.destroyImageLater(_gasImage);
    context.destroyImageLater(_gasIrradiance);
    context.destroyLater([device = context.getDevice(), pipelineLayout = _pipelineLayout, descriptorSetLayout = _descriptorSetLayout] {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    });
}

void _RayTracingRenderStep::prepareResources(RenderView const& view, NumRenderObjects const& numObjects)
{
    if (!_preparePipeline) {
        createPipelines();
    }

    // The descriptor set always refers to the scene, which only grows while the tissue is visible
    auto tissueVisible = getTissueVisibility(view) > 0.0f;
    auto numSpheres = tissueVisible ? numObjects.objects : 0;
    auto numCapsules = tissueVisible ? numObjects.lineIndices / 2 : 0;
    auto numTriangles = tissueVisible ? numObjects.triangleIndices / 3 : 0;
    if (_scene.objectCapacity == 0 || numSpheres > _scene.objectCapacity || numCapsules > _scene.capsuleCapacity || numTriangles > _scene.triangleCapacity) {
        auto grow = [](uint64_t capacity, uint64_t required) { return std::max({MinCapacity, capacity, required * 3 / 2}); };
        auto scene = createSceneResources(
            grow(_scene.objectCapacity, numSpheres), grow(_scene.capsuleCapacity, numCapsules), grow(_scene.triangleCapacity, numTriangles));
        destroySceneResources(_scene, false);
        _scene = scene;
    }

    updateLightGrid(view.visibleWorldRect);
    updateGasGrid(view);

    if (_view.size != view.viewSize) {
        destroyViewResources(_view, false);
        _view = createViewResources(view.viewSize);
        _current = 0;
        _previousFrameValid = false;
    }
}

void _RayTracingRenderStep::setStillPicture(bool value)
{
    _stillPicture = value;
}

void _RayTracingRenderStep::resetImageStates()
{
    auto images = _view.getImages();
    images.emplace_back(&_gasImage);
    images.emplace_back(&_gasIrradiance);
    for (auto image : images) {
        image->layout = VK_IMAGE_LAYOUT_UNDEFINED;
        image->lastStages = VK_PIPELINE_STAGE_2_NONE;
        image->lastAccesses = VK_ACCESS_2_NONE;
    }
    _previousFrameValid = false;
}

void _RayTracingRenderStep::execute(ExecutionParameters parameters)
{
    auto const& geometryBuffers = parameters._geometryBuffers;
    auto const& view = parameters._view;
    auto numObjects = geometryBuffers->getNumObjects();
    prepareResources(view, numObjects);

    auto commandBuffer = parameters._renderInfo.commandBuffer;
    auto const& textures = parameters._textures;
    CHECK(textures.size() == 2);
    VulkanImageBarriers().add(textures.at(0)->color, ImageUsage::ShaderRead).add(textures.at(1)->color, ImageUsage::ShaderRead).record(commandBuffer);

    if (!_stillPicture || !isSameViewAsPrevious(view)) {
        _numAveragedFrames = 0;
    }
    updateFrameConstants(view, numObjects, *parameters._simulationParameters);
    auto descriptorSet = writeDescriptorSet(geometryBuffers, textures);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    if (getTissueVisibility(view) > 0.0f) {
        buildScene(commandBuffer, numObjects);
    }
    buildGas(commandBuffer, numObjects);
    shade(commandBuffer);
    present(parameters);

    _previousView = view;
    _previousFrameValid = true;
    _numAveragedFrames = _stillPicture ? _numAveragedFrames + 1 : 0;
    _current = getPrevious();
}

_RayTracingRenderStep::_RayTracingRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

namespace
{
    VkAccelerationStructureGeometryKHR createAabbGeometry(VkDeviceAddress address)
    {
        return VkAccelerationStructureGeometryKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
            .geometry =
                {.aabbs =
                     {
                         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
                         .data = {.deviceAddress = address},
                         .stride = sizeof(VkAabbPositionsKHR),
                     }},
            // The search for the nearest cells requires that every primitive is found exactly once
            .flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
        };
    }

    // The order determines the geometry index in the shaders
    std::vector<VkAccelerationStructureGeometryKHR>
    createBottomLevelGeometries(VulkanBuffer const& sphereAabbs, VulkanBuffer const& capsuleAabbs, VulkanBuffer const& triangleAabbs)
    {
        auto& context = VulkanContext::get();
        return {
            createAabbGeometry(context.getDeviceAddress(sphereAabbs)),
            createAabbGeometry(context.getDeviceAddress(capsuleAabbs)),
            createAabbGeometry(context.getDeviceAddress(triangleAabbs)),
        };
    }

    VkAccelerationStructureGeometryKHR createInstanceGeometry(VkDeviceAddress address)
    {
        return VkAccelerationStructureGeometryKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry =
                {.instances =
                     {
                         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                         .arrayOfPointers = VK_FALSE,
                         .data = {.deviceAddress = address},
                     }},
        };
    }

    // The objects change every frame and are rebuilt from scratch, a fast build pays off more than fast tracing
    VkAccelerationStructureBuildGeometryInfoKHR createBuildInfo(
        VkAccelerationStructureTypeKHR type,
        std::vector<VkAccelerationStructureGeometryKHR> const& geometries)
    {
        return VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = type,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = static_cast<uint32_t>(geometries.size()),
            .pGeometries = geometries.data(),
        };
    }

    VkAccelerationStructureKHR createAccelerationStructure(VkAccelerationStructureTypeKHR type, VulkanBuffer const& buffer)
    {
        VkAccelerationStructureCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = buffer.buffer,
            .size = buffer.size,
            .type = type,
        };
        VkAccelerationStructureKHR result;
        checkVkResult(vkCreateAccelerationStructureKHR(VulkanContext::get().getDevice(), &createInfo, nullptr, &result), "vkCreateAccelerationStructureKHR");
        return result;
    }

    VkDeviceAddress getAlignedScratchAddress(VulkanBuffer const& scratchBuffer)
    {
        auto& context = VulkanContext::get();
        auto alignment = context.getAccelerationStructureScratchAlignment();
        return (context.getDeviceAddress(scratchBuffer) + alignment - 1) / alignment * alignment;
    }
}

_RayTracingRenderStep::SceneResources _RayTracingRenderStep::createSceneResources(uint64_t objectCapacity, uint64_t capsuleCapacity, uint64_t triangleCapacity)
{
    auto& context = VulkanContext::get();
    auto device = context.getDevice();
    SceneResources result{.objectCapacity = objectCapacity, .capsuleCapacity = capsuleCapacity, .triangleCapacity = triangleCapacity};
    try {
        result.spheres = context.createBuffer(objectCapacity * SphereSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VulkanMemory::DeviceLocal);
        result.sphereAabbs = context.createBuffer(objectCapacity * sizeof(VkAabbPositionsKHR), AabbBufferUsage, VulkanMemory::DeviceLocal);
        result.capsuleAabbs = context.createBuffer(capsuleCapacity * sizeof(VkAabbPositionsKHR), AabbBufferUsage, VulkanMemory::DeviceLocal);
        result.triangleAabbs = context.createBuffer(triangleCapacity * sizeof(VkAabbPositionsKHR), AabbBufferUsage, VulkanMemory::DeviceLocal);
        result.lightEntries = context.createBuffer(objectCapacity * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VulkanMemory::DeviceLocal);

        auto bottomLevelGeometries = createBottomLevelGeometries(result.sphereAabbs, result.capsuleAabbs, result.triangleAabbs);
        auto bottomLevelBuildInfo = createBuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, bottomLevelGeometries);
        std::array maxPrimitiveCounts{static_cast<uint32_t>(objectCapacity), static_cast<uint32_t>(capsuleCapacity), static_cast<uint32_t>(triangleCapacity)};
        VkAccelerationStructureBuildSizesInfoKHR bottomLevelSizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(
            device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bottomLevelBuildInfo, maxPrimitiveCounts.data(), &bottomLevelSizes);
        result.bottomLevelBuffer = context.createBuffer(
            bottomLevelSizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VulkanMemory::DeviceLocal);
        result.bottomLevel = createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, result.bottomLevelBuffer);

        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = result.bottomLevel,
        };
        VkAccelerationStructureInstanceKHR instance{
            .transform = {.matrix = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}},
            .instanceCustomIndex = 0,
            .mask = 0xff,
            .instanceShaderBindingTableRecordOffset = 0,
            .flags = 0,
            .accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo),
        };
        result.instanceBuffer = context.createBuffer(
            sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VulkanMemory::HostVisible);
        std::memcpy(result.instanceBuffer.mapped, &instance, sizeof(instance));

        std::vector<VkAccelerationStructureGeometryKHR> topLevelGeometries{createInstanceGeometry(context.getDeviceAddress(result.instanceBuffer))};
        auto topLevelBuildInfo = createBuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, topLevelGeometries);
        uint32_t numInstances = 1;
        VkAccelerationStructureBuildSizesInfoKHR topLevelSizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &topLevelBuildInfo, &numInstances, &topLevelSizes);
        result.topLevelBuffer = context.createBuffer(
            topLevelSizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VulkanMemory::DeviceLocal);
        result.topLevel = createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, result.topLevelBuffer);

        // Both builds use the same scratch memory one after the other
        result.scratchBuffer = context.createBuffer(
            std::max(bottomLevelSizes.buildScratchSize, topLevelSizes.buildScratchSize) + context.getAccelerationStructureScratchAlignment(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VulkanMemory::DeviceLocal);
    } catch (...) {
        destroySceneResources(result, true);
        throw;
    }
    return result;
}

void _RayTracingRenderStep::destroySceneResources(SceneResources& resources, bool immediately)
{
    auto& context = VulkanContext::get();
    if (context.isActive()) {
        auto destroyFunc = [device = context.getDevice(), bottomLevel = resources.bottomLevel, topLevel = resources.topLevel] {
            vkDestroyAccelerationStructureKHR(device, topLevel, nullptr);
            vkDestroyAccelerationStructureKHR(device, bottomLevel, nullptr);
        };
        if (immediately) {
            destroyFunc();
        } else {
            context.destroyLater(destroyFunc);
        }
    }
    for (auto buffer : {
             &resources.spheres,
             &resources.sphereAabbs,
             &resources.capsuleAabbs,
             &resources.triangleAabbs,
             &resources.lightEntries,
             &resources.bottomLevelBuffer,
             &resources.topLevelBuffer,
             &resources.scratchBuffer,
             &resources.instanceBuffer,
         }) {
        if (immediately) {
            context.destroyBuffer(*buffer);
        } else {
            context.destroyBufferLater(*buffer);
        }
    }
    resources = SceneResources();
}

std::vector<VulkanImage*> _RayTracingRenderStep::ViewResources::getImages()
{
    return {&sceneColor, &antiAliased.at(0), &antiAliased.at(1)};
}

_RayTracingRenderStep::ViewResources _RayTracingRenderStep::createViewResources(IntVector2D const& size)
{
    auto& context = VulkanContext::get();
    ViewResources result{.size = size};
    try {
        for (auto image : result.getImages()) {
            *image = context.createImage(size, VK_FORMAT_R16G16B16A16_SFLOAT, ViewImageUsage);
        }
    } catch (...) {
        destroyViewResources(result, true);
        throw;
    }
    return result;
}

void _RayTracingRenderStep::destroyViewResources(ViewResources& resources, bool immediately)
{
    auto& context = VulkanContext::get();
    for (auto image : resources.getImages()) {
        if (immediately) {
            context.destroyImage(*image);
        } else {
            context.destroyImageLater(*image);
        }
    }
    resources = ViewResources();
}

uint64_t _RayTracingRenderStep::LightGrid::getNumCells() const
{
    return static_cast<uint64_t>(size.x) * size.y;
}

void _RayTracingRenderStep::createPipelines()
{
    auto& context = VulkanContext::get();
    auto device = context.getDevice();
    if (_pipelineLayout != VK_NULL_HANDLE) {
        context.destroyLater([device, pipelineLayout = _pipelineLayout, descriptorSetLayout = _descriptorSetLayout] {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        });
        _pipelineLayout = VK_NULL_HANDLE;
        _descriptorSetLayout = VK_NULL_HANDLE;
    }
    context.destroyBufferLater(_frameConstants);
    context.destroyBufferLater(_allocation);

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (auto binding : std::views::iota(0u, static_cast<uint32_t>(Binding_Count))) {
        bindings.emplace_back(VkDescriptorSetLayoutBinding{
            .binding = binding,
            .descriptorType = getDescriptorType(binding),
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        });
    }
    VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    checkVkResult(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &_descriptorSetLayout), "vkCreateDescriptorSetLayout");
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &_descriptorSetLayout,
    };
    checkVkResult(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &_pipelineLayout), "vkCreatePipelineLayout");

    auto createPipeline = [this](std::string_view source, IntVector2D const& workGroupSize) {
        return _ComputePipeline::create(std::string(Shaders::RayTracingCommonCS) + std::string(source), _pipelineLayout, workGroupSize);
    };
    _allocatePipeline = createPipeline(Shaders::RayTracingAllocateCS, {256, 1});
    _scatterPipeline = createPipeline(Shaders::RayTracingScatterCS, {64, 1});
    _gasSplatPipeline = createPipeline(Shaders::RayTracingGasSplatCS, {64, 1});
    _gasLightsPipeline = createPipeline(Shaders::RayTracingGasLightsCS, {64, 1});
    _gasIrradiancePipeline = createPipeline(Shaders::RayTracingGasIrradianceCS, {8, 8});
    _gasBlurHorizontalPipeline = createPipeline("const ivec2 BlurDirection = ivec2(1, 0);\n" + std::string(Shaders::RayTracingGasBlurCS), {8, 8});
    _gasBlurVerticalPipeline =
        createPipeline("#define VERTICAL\nconst ivec2 BlurDirection = ivec2(0, 1);\n" + std::string(Shaders::RayTracingGasBlurCS), {8, 8});
    _tissuePipeline = createPipeline(std::string(Shaders::RayTracingGasShadingCS) + std::string(Shaders::RayTracingTissueCS), {8, 8});
    _antiAliasingPipeline = createPipeline(Shaders::RayTracingAntiAliasingCS, {8, 8});
    _frameConstants = context.createBuffer(sizeof(FrameConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VulkanMemory::HostVisible);
    _allocation = context.createBuffer(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VulkanMemory::DeviceLocal);
    createGasResources();

    // Created last since it marks the pipelines as complete
    _preparePipeline = createPipeline(Shaders::RayTracingPrepareCS, {64, 1});
}

void _RayTracingRenderStep::createGasResources()
{
    auto& context = VulkanContext::get();
    context.destroyBufferLater(_gasAccumulation);
    context.destroyBufferLater(_gasIntermediate);
    context.destroyBufferLater(_gasLightClusters);
    context.destroyImageLater(_gasImage);
    context.destroyImageLater(_gasIrradiance);
    auto numNodes = static_cast<VkDeviceSize>(MaxGasNodes) * MaxGasNodes;
    auto clearableUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    _gasAccumulation = context.createBuffer(numNodes * GasAccumulationStride * sizeof(uint32_t), clearableUsage, VulkanMemory::DeviceLocal);
    _gasIntermediate = context.createBuffer(numNodes * GasLayers * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VulkanMemory::DeviceLocal);
    auto numClusters = static_cast<VkDeviceSize>(MaxLightClusters) * MaxLightClusters;
    _gasLightClusters = context.createBuffer(numClusters * GasLightClusterStride * sizeof(uint32_t), clearableUsage, VulkanMemory::DeviceLocal);
    _gasImage = context.createImage({GasLayers * MaxGasNodes, MaxGasNodes}, VK_FORMAT_R16G16B16A16_SFLOAT, ViewImageUsage);
    _gasIrradiance = context.createImage({GasIrradianceImageSize, GasIrradianceImageSize}, VK_FORMAT_R16G16B16A16_SFLOAT, ViewImageUsage);
}

// Extreme zoom levels enlarge the cells and thus the range of the lights. A margin of one cell covers the lights next to the view.
void _RayTracingRenderStep::updateLightGrid(RealRect const& visibleWorldRect)
{
    auto extent = visibleWorldRect.bottomRight - visibleWorldRect.topLeft;
    auto cellSize = std::max(LightRange, std::max(extent.x, extent.y) / MaxLightGridCellsPerDimension);
    LightGrid grid{
        .origin = visibleWorldRect.topLeft - RealVector2D{cellSize, cellSize},
        .cellSize = cellSize,
        .size = {toInt(std::ceil(extent.x / cellSize)) + 2, toInt(std::ceil(extent.y / cellSize)) + 2},
    };
    auto numCells = grid.getNumCells();
    if (numCells > _cellCapacity) {
        auto& context = VulkanContext::get();
        auto capacity = numCells * 3 / 2;
        std::array<VulkanBuffer, 3> buffers;
        try {
            for (auto& buffer : buffers) {
                buffer = context.createBuffer(
                    capacity * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VulkanMemory::DeviceLocal);
            }
        } catch (...) {
            for (auto& buffer : buffers) {
                context.destroyBuffer(buffer);
            }
            throw;
        }
        std::array cellBuffers{&_cellCounts, &_cellOffsets, &_cellCursors};
        for (auto const& [buffer, newBuffer] : std::views::zip(cellBuffers, buffers)) {
            context.destroyBufferLater(*buffer);
            *buffer = newBuffer;
        }
        _cellCapacity = capacity;
    }
    _lightGrid = grid;
}

void _RayTracingRenderStep::updateGasGrid(RenderView const& view)
{
    auto pixelSize = 1.0f / view.zoomFactor;
    auto cloudSpread = std::max(CloudSpread, MinCloudSpreadInPixels * pixelSize);
    auto margin = std::max(3.0f * cloudSpread, GasLightRange);
    auto topLeft = view.visibleWorldRect.topLeft - RealVector2D{margin, margin};
    auto bottomRight = view.visibleWorldRect.bottomRight + RealVector2D{margin, margin};
    auto extent = bottomRight - topLeft;
    auto nodeDistance = MinGasNodeDistance;
    while (std::max(extent.x, extent.y) / nodeDistance > toFloat(MaxGasNodes - 2)) {
        nodeDistance *= 2.0f;
    }
    RealVector2D origin{std::floor(topLeft.x / nodeDistance) * nodeDistance, std::floor(topLeft.y / nodeDistance) * nodeDistance};
    IntVector2D size{
        std::min(toInt(std::ceil((bottomRight.x - origin.x) / nodeDistance)) + 1, MaxGasNodes),
        std::min(toInt(std::ceil((bottomRight.y - origin.y) / nodeDistance)) + 1, MaxGasNodes)};
    auto lightClusterSize = std::max(MinLightClusterSize, 2.0f * nodeDistance);
    auto getNumLightClusters = [&](int numNodes) { return std::min(toInt(std::ceil(toFloat(numNodes) * nodeDistance / lightClusterSize)), MaxLightClusters); };
    _gasGrid = GasGrid{
        .origin = origin,
        .nodeDistance = nodeDistance,
        .size = size,
        .blurRadius = std::min(toInt(std::ceil(3.0f * cloudSpread / nodeDistance)), MaxGasBlurRadius),
        .cloudSpreadInNodes = cloudSpread / nodeDistance,
        .lightClusterSize = lightClusterSize,
        .numLightClusters = {getNumLightClusters(size.x), getNumLightClusters(size.y)},
        .irradianceSize = {std::min((size.x + 1) / 2, GasIrradianceImageSize), std::min((size.y + 1) / 2, GasIrradianceImageSize)},
    };
}

namespace
{
    float getHaltonValue(uint32_t index, uint32_t base)
    {
        auto result = 0.0f;
        auto fraction = 1.0f;
        while (index > 0) {
            fraction /= toFloat(base);
            result += fraction * toFloat(index % base);
            index /= base;
        }
        return result;
    }

    float getSmoothTransition(float value, float start, float end)
    {
        auto fraction = std::clamp((value - start) / (end - start), 0.0f, 1.0f);
        return fraction * fraction * (3.0f - 2.0f * fraction);
    }
}

float _RayTracingRenderStep::getTissueVisibility(RenderView const& view)
{
    return getSmoothTransition(view.getScreenZoomFactor(), TissueZoomStart, TissueZoomEnd);
}

void _RayTracingRenderStep::updateFrameConstants(RenderView const& view, NumRenderObjects const& numObjects, SimulationParameters const& parameters)
{
    auto const& topLeft = view.visibleWorldRect.topLeft;
    RealVector2D jitter;
    if (_stillPicture) {
        auto jitterIndex = _numAveragedFrames % NumJitterPositions + 1;
        jitter = {getHaltonValue(jitterIndex, 2) - 0.5f, getHaltonValue(jitterIndex, 3) - 0.5f};
    }
    auto screenZoom = view.getScreenZoomFactor();
    auto closeUpWeight = getSmoothTransition(screenZoom, CloseUpZoomStart, CloseUpZoomEnd);
    auto detailVisibility = getSmoothTransition(screenZoom, DetailZoomStart, DetailZoomEnd);
    auto backgroundScale = 1.0f / std::max(1.5f - parameters.glow.value, 0.1f);

    FrameConstants constants{
        .view = {toFloat(view.viewSize.x), toFloat(view.viewSize.y), view.zoomFactor, screenZoom},
        .rect = {topLeft.x, topLeft.y, jitter.x, jitter.y},
        .counts =
            {static_cast<uint32_t>(numObjects.objects),
             static_cast<uint32_t>(numObjects.lineIndices / 2),
             static_cast<uint32_t>(numObjects.triangleIndices / 3),
             static_cast<uint32_t>(_lightGrid.getNumCells())},
        .lightGrid = {_lightGrid.origin.x, _lightGrid.origin.y, _lightGrid.cellSize, _lightGrid.cellSize},
        .gridSize = {_lightGrid.size.x, _lightGrid.size.y, 0, 0},
        .tissue = {closeUpWeight, detailVisibility, 1.0f / view.zoomFactor, getTissueVisibility(view)},
        .parameters = {backgroundScale, Bioluminescence, _stillPicture ? 0.0f : 1.0f, 1.0f / toFloat(_numAveragedFrames + 1)},
        .gasGrid = {_gasGrid.origin.x, _gasGrid.origin.y, _gasGrid.nodeDistance, _gasGrid.cloudSpreadInNodes},
        .gasSize = {_gasGrid.size.x, _gasGrid.size.y, _gasGrid.blurRadius, static_cast<int32_t>(numObjects.fluidParticles)},
        .gasLight = {_gasGrid.lightClusterSize, GasLightRange, 2.0f * _gasGrid.nodeDistance, 0.0f},
        .gasLightGrid = {_gasGrid.numLightClusters.x, _gasGrid.numLightClusters.y, _gasGrid.irradianceSize.x, _gasGrid.irradianceSize.y},
    };
    std::memcpy(_frameConstants.mapped, &constants, sizeof(constants));
}

VkDescriptorSet _RayTracingRenderStep::writeDescriptorSet(VulkanGeometryBuffers const& geometryBuffers, std::vector<TextureTarget> const& textures)
{
    auto descriptorSet = VulkanContext::get().allocateFrameDescriptorSet(_descriptorSetLayout);
    auto constexpr General = VK_IMAGE_LAYOUT_GENERAL;
    auto constexpr ShaderReadOnly = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    DescriptorSetWriter()
        .uniformBuffer(Binding_FrameConstants, _frameConstants.buffer)
        .accelerationStructure(Binding_Scene, _scene.topLevel)
        .storageBuffer(Binding_Objects, geometryBuffers->getBuffer(GeometryBufferType_Objects))
        .storageBuffer(Binding_LineIndices, geometryBuffers->getBuffer(GeometryBufferType_LineIndices))
        .storageBuffer(Binding_TriangleIndices, geometryBuffers->getBuffer(GeometryBufferType_TriangleIndices))
        .storageBuffer(Binding_Spheres, _scene.spheres.buffer)
        .storageBuffer(Binding_SphereAabbs, _scene.sphereAabbs.buffer)
        .storageBuffer(Binding_CapsuleAabbs, _scene.capsuleAabbs.buffer)
        .storageBuffer(Binding_TriangleAabbs, _scene.triangleAabbs.buffer)
        .storageBuffer(Binding_CellCounts, _cellCounts.buffer)
        .storageBuffer(Binding_CellOffsets, _cellOffsets.buffer)
        .storageBuffer(Binding_CellCursors, _cellCursors.buffer)
        .storageBuffer(Binding_CellEntries, _scene.lightEntries.buffer)
        .storageBuffer(Binding_Allocation, _allocation.buffer)
        .storageImage(Binding_SceneColor, _view.sceneColor.view)
        .storageImage(Binding_AntiAliased, _view.antiAliased.at(_current).view)
        .sampledImage(Binding_PreviousAntiAliased, _view.antiAliased.at(getPrevious()).view, General)
        .sampledImage(Binding_GroundTexture, textures.at(0)->color.view, ShaderReadOnly)
        .sampledImage(Binding_GasSampler, _gasImage.view, General)
        .sampledImage(Binding_SceneColorSampler, _view.sceneColor.view, General)
        .storageBuffer(Binding_FluidParticles, geometryBuffers->getBuffer(GeometryBufferType_FluidParticles))
        .storageBuffer(Binding_GasAccumulation, _gasAccumulation.buffer)
        .storageBuffer(Binding_GasIntermediate, _gasIntermediate.buffer)
        .storageImage(Binding_GasImage, _gasImage.view)
        .sampledImage(Binding_EnergyParticleTexture, textures.at(1)->color.view, ShaderReadOnly)
        .storageBuffer(Binding_GasLightClusters, _gasLightClusters.buffer)
        .storageImage(Binding_GasIrradiance, _gasIrradiance.view)
        .sampledImage(Binding_GasIrradianceSampler, _gasIrradiance.view, General)
        .write(descriptorSet);
    return descriptorSet;
}

namespace
{
    void recordMemoryBarrier(
        VkCommandBuffer commandBuffer,
        VkPipelineStageFlags2 sourceStages,
        VkAccessFlags2 sourceAccesses,
        VkPipelineStageFlags2 destinationStages,
        VkAccessFlags2 destinationAccesses)
    {
        VkMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = sourceStages,
            .srcAccessMask = sourceAccesses,
            .dstStageMask = destinationStages,
            .dstAccessMask = destinationAccesses,
        };
        VkDependencyInfo dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    void recordComputeBarrier(VkCommandBuffer commandBuffer)
    {
        recordMemoryBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
}

void _RayTracingRenderStep::buildScene(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects)
{
    auto numSpheres = static_cast<uint32_t>(numObjects.objects);
    auto numCapsules = static_cast<uint32_t>(numObjects.lineIndices / 2);
    auto numTriangles = static_cast<uint32_t>(numObjects.triangleIndices / 3);
    auto numCells = _lightGrid.getNumCells();

    vkCmdFillBuffer(commandBuffer, _cellCounts.buffer, 0, numCells * sizeof(uint32_t), 0);
    vkCmdFillBuffer(commandBuffer, _allocation.buffer, 0, sizeof(uint32_t), 0);
    recordMemoryBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_2_CLEAR_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    _preparePipeline->dispatch(commandBuffer, {toInt(std::max({numSpheres, numCapsules, numTriangles})), 1});
    recordMemoryBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // The lights are sorted into the grid while the acceleration structures are built
    _allocatePipeline->dispatch(commandBuffer, {toInt(numCells), 1});
    recordComputeBarrier(commandBuffer);
    _scatterPipeline->dispatch(commandBuffer, {toInt(numSpheres), 1});

    auto bottomLevelGeometries = createBottomLevelGeometries(_scene.sphereAabbs, _scene.capsuleAabbs, _scene.triangleAabbs);
    auto bottomLevelBuildInfo = createBuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, bottomLevelGeometries);
    bottomLevelBuildInfo.dstAccelerationStructure = _scene.bottomLevel;
    bottomLevelBuildInfo.scratchData.deviceAddress = getAlignedScratchAddress(_scene.scratchBuffer);
    std::array bottomLevelRanges{
        VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = numSpheres},
        VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = numCapsules},
        VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = numTriangles},
    };
    auto bottomLevelRangesPointer = bottomLevelRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &bottomLevelBuildInfo, &bottomLevelRangesPointer);
    recordMemoryBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    std::vector<VkAccelerationStructureGeometryKHR> topLevelGeometries{createInstanceGeometry(VulkanContext::get().getDeviceAddress(_scene.instanceBuffer))};
    auto topLevelBuildInfo = createBuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, topLevelGeometries);
    topLevelBuildInfo.dstAccelerationStructure = _scene.topLevel;
    topLevelBuildInfo.scratchData.deviceAddress = getAlignedScratchAddress(_scene.scratchBuffer);
    VkAccelerationStructureBuildRangeInfoKHR topLevelRange{.primitiveCount = 1};
    auto topLevelRangePointer = &topLevelRange;
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &topLevelBuildInfo, &topLevelRangePointer);
    recordMemoryBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

void _RayTracingRenderStep::buildGas(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects)
{
    auto numNodes = static_cast<VkDeviceSize>(_gasGrid.size.x) * _gasGrid.size.y;
    auto numClusters = static_cast<VkDeviceSize>(_gasGrid.numLightClusters.x) * _gasGrid.numLightClusters.y;
    vkCmdFillBuffer(commandBuffer, _gasAccumulation.buffer, 0, numNodes * GasAccumulationStride * sizeof(uint32_t), 0);
    vkCmdFillBuffer(commandBuffer, _gasLightClusters.buffer, 0, numClusters * GasLightClusterStride * sizeof(uint32_t), 0);
    recordMemoryBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_2_CLEAR_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    _gasSplatPipeline->dispatch(commandBuffer, {toInt(numObjects.fluidParticles), 1});
    recordComputeBarrier(commandBuffer);
    _gasLightsPipeline->dispatch(commandBuffer, {toInt(numObjects.objects), 1});
    recordComputeBarrier(commandBuffer);
    _gasBlurHorizontalPipeline->dispatch(commandBuffer, _gasGrid.size);
    recordComputeBarrier(commandBuffer);
    VulkanImageBarriers().add(_gasImage, ImageUsage::ComputeReadWrite).record(commandBuffer);
    _gasBlurVerticalPipeline->dispatch(commandBuffer, _gasGrid.size);

    // The light reaches the gas through the gas and the cells
    recordComputeBarrier(commandBuffer);
    VulkanImageBarriers().add(_gasImage, ImageUsage::ComputeRead).add(_gasIrradiance, ImageUsage::ComputeReadWrite).record(commandBuffer);
    _gasIrradiancePipeline->dispatch(commandBuffer, _gasGrid.irradianceSize);
}

void _RayTracingRenderStep::shade(VkCommandBuffer commandBuffer)
{
    auto& view = _view;
    recordComputeBarrier(commandBuffer);
    VulkanImageBarriers()
        .add(view.sceneColor, ImageUsage::ComputeReadWrite)
        .add(_gasImage, ImageUsage::ComputeRead)
        .add(_gasIrradiance, ImageUsage::ComputeRead)
        .record(commandBuffer);
    _tissuePipeline->dispatch(commandBuffer, view.size);

    recordComputeBarrier(commandBuffer);
    VulkanImageBarriers()
        .add(view.sceneColor, ImageUsage::ComputeRead)
        .add(view.antiAliased.at(getPrevious()), ImageUsage::ComputeRead)
        .add(view.antiAliased.at(_current), ImageUsage::ComputeReadWrite)
        .record(commandBuffer);
    _antiAliasingPipeline->dispatch(commandBuffer, view.size);
}

void _RayTracingRenderStep::present(ExecutionParameters const& parameters)
{
    auto commandBuffer = parameters._renderInfo.commandBuffer;
    auto& image = _view.antiAliased.at(_current);
    VulkanImageBarriers().add(image, ImageUsage::ShaderRead).record(commandBuffer);

    prepareExecution(parameters);
    _shader->setTexture("inputTexture1", image);
    _shader->bind(commandBuffer, PipelineState{.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .colorFormat = parameters._target->color.format});
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    finishExecution(parameters);
}

int _RayTracingRenderStep::getPrevious() const
{
    return 1 - _current;
}

bool _RayTracingRenderStep::isSameViewAsPrevious(RenderView const& view) const
{
    return _previousFrameValid && view.viewSize == _previousView.viewSize && view.zoomFactor == _previousView.zoomFactor
        && view.visibleWorldRect.topLeft == _previousView.visibleWorldRect.topLeft
        && view.visibleWorldRect.bottomRight == _previousView.visibleWorldRect.bottomRight;
}
