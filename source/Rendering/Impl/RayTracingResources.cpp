#include "RayTracingResources.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ranges>
#include <format>

#include <Engine/Interface/SimulationFacade.h>

#include <Rendering/Impl/Shaders/ShaderSources.h>

#include "SimulationRenderer.h"
#include "VulkanContext.h"

namespace
{
    auto constexpr MinCapacity = uint64_t(4096);
    auto constexpr NumJitterPositions = 16u;

    // Between these screen zoom factors, the tissue fades in over the rasterized objects. It starts from this visibility of the transition,
    // below which it would hardly show and is not traced.
    auto constexpr TissueZoomStart = 4.0f;
    auto constexpr TissueZoomEnd = 10.0f;
    auto constexpr TissueVisibilityStart = 0.05f;

    // The light in the gas is only reused while it lags behind the objects by at most this number of time steps
    auto constexpr MaxLightDelay = uint64_t(4);

    // Between these screen zoom factors, the thin lines of the zoomed out view grow into the tissue
    auto constexpr CloseUpZoomStart = 3.0f;
    auto constexpr CloseUpZoomEnd = 12.0f;

    // The inner details of the cells appear between these screen zoom factors, the walls already with the tissue
    auto constexpr DetailZoomStart = 8.0f;
    auto constexpr DetailZoomEnd = 20.0f;

    // Layouts of the sums at the nodes of the gas grid and in the light clusters, the sizes here and of the grids are declared for the shaders as
    // well, see createPipeline. The occlusion of the extracted objects has sums of its own at the nodes of the gas grid, see
    // Shaders/RayTracingCommonCS.h.
    auto constexpr GasAccumulationStride = 5;
    auto constexpr GasLayers = 2;
    auto constexpr GasLightClusterStride = 6;
    auto constexpr OcclusionStride = 3;

    // Factor of the cell light in the gas at the highest fluid illumination
    auto constexpr MaxFluidIllumination = 4.0f;

    auto constexpr MaxShadowGridBlurRadius = 16;

    auto constexpr SphereSize = VkDeviceSize(64);
    auto constexpr AabbBufferUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto constexpr ViewImageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    // Entries per capsule and per triangle in the buffer of the shadow casters, see Shaders/RayTracingCommonCS.h
    auto constexpr ShadowCasterEntrySize = VkDeviceSize(16);
    auto constexpr CapsuleShadowEntries = VkDeviceSize(2);
    auto constexpr TriangleShadowEntries = VkDeviceSize(3);

    // The levels of an occlusion pyramid lie side by side in the image of its grid, see Shaders/RayTracingCommonCS.h. Each level has exactly
    // half the size of the level before.
    IntVector2D getPyramidImageSize(int gridImageSize)
    {
        return {gridImageSize + gridImageSize / 2, gridImageSize};
    }
    auto constexpr CoarsestPyramidLevelScale = 1 << (RayTracingLayout::OcclusionLevels - 1);
    static_assert(RayTracingLayout::MaxGasNodes % CoarsestPyramidLevelScale == 0 && RayTracingLayout::MaxShadowGridNodes % CoarsestPyramidLevelScale == 0);

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
        Binding_ShadowGridNodes,
        Binding_ShadowGridImage,
        Binding_ShadowGridSampler,
        Binding_ShadowCasters,
        Binding_OcclusionAccumulation,
        Binding_OcclusionImage,
        Binding_OcclusionSampler,
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
        case Binding_ShadowGridImage:
        case Binding_OcclusionImage:
            return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case Binding_PreviousAntiAliased:
        case Binding_GroundTexture:
        case Binding_GasSampler:
        case Binding_SceneColorSampler:
        case Binding_EnergyParticleTexture:
        case Binding_GasIrradianceSampler:
        case Binding_ShadowGridSampler:
        case Binding_OcclusionSampler:
            return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        default:
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
    }

    // Layouts of the uniform block FrameConstants and its struct SourceLight in Shaders/RayTracingCommonCS.h
    struct SourceLightConstants
    {
        float centerRangeShape[4];
        float color[4];
        float extent[4];
    };

    struct FrameConstants
    {
        float view[4];
        float rect[4];
        uint32_t counts[4];
        float tissue[4];
        float parameters[4];
        float gasGrid[4];
        int32_t gasSize[4];
        float gasLight[4];
        int32_t gasLightGrid[4];
        float gasLighting[4];
        float world[4];
        float shadowGrid[4];
        int32_t shadowGridSize[4];
        int32_t numSourceLights[4];
        SourceLightConstants sourceLights[MAX_SOURCES];
    };
}

RayTracingResources _RayTracingResources::create()
{
    return RayTracingResources(new _RayTracingResources());
}

_RayTracingResources::~_RayTracingResources()
{
    auto& context = VulkanContext::get();
    if (!context.isActive()) {
        return;
    }
    destroySceneResources(_scene, false);
    destroyViewResources(_view, false);
    for (auto buffer : {&_frameConstants, &_gasAccumulation, &_gasIntermediate, &_gasLightClusters, &_occlusionAccumulation}) {
        context.destroyBufferLater(*buffer);
    }
    context.destroyImageLater(_gasImage);
    context.destroyImageLater(_gasIrradiance);
    context.destroyImageLater(_shadowGridImage);
    context.destroyImageLater(_occlusionImage);
    context.destroyLater([device = context.getDevice(), pipelineLayout = _pipelineLayout, descriptorSetLayout = _descriptorSetLayout] {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    });
}

void _RayTracingResources::prepare(
    RenderView const& view,
    RayTracingLayout const& layout,
    NumRenderObjects const& numObjects,
    SimulationParameters const& parameters)
{
    CHECK(!layout.isEmpty());
    if (_frameConstants.buffer == VK_NULL_HANDLE) {
        createFrameResources();
    }

    // The descriptor set always refers to the scene, which only grows while it is needed
    auto sceneNeeded = isSceneNeeded(view, parameters);
    auto numSpheres = sceneNeeded ? numObjects.objects : 0;
    auto numCapsules = sceneNeeded ? numObjects.lineIndices / 2 : 0;
    auto numTriangles = sceneNeeded ? numObjects.triangleIndices / 3 : 0;
    if (_scene.objectCapacity == 0 || numSpheres > _scene.objectCapacity || numCapsules > _scene.capsuleCapacity || numTriangles > _scene.triangleCapacity) {
        auto grow = [](uint64_t capacity, uint64_t required) { return std::max({MinCapacity, capacity, required * 3 / 2}); };
        auto scene = createSceneResources(
            grow(_scene.objectCapacity, numSpheres), grow(_scene.capsuleCapacity, numCapsules), grow(_scene.triangleCapacity, numTriangles));
        destroySceneResources(_scene, false);
        _scene = scene;
    }

    _layout = layout;

    if (_view.size != view.viewSize) {
        destroyViewResources(_view, false);
        _view = createViewResources(view.viewSize);
        _current = 0;
        _numAveragedFrames = 0;
    }
}

void _RayTracingResources::setStillPicture(bool value)
{
    _stillPicture = value;
    _numAveragedFrames = 0;
    _lightComputed = false;
}

bool _RayTracingResources::isRepeatedStillPictureFrame() const
{
    return _stillPicture && _numAveragedFrames > 0;
}

bool _RayTracingResources::isLightReused() const
{
    return _lightReused;
}

bool _RayTracingResources::isTissueTraced() const
{
    return _tissueTraced;
}

bool _RayTracingResources::isAntiAliasingNeeded() const
{
    return _tissueTraced || _stillPicture;
}

void _RayTracingResources::resetImageStates()
{
    auto images = _view.getImages();
    images.emplace_back(&_gasImage);
    images.emplace_back(&_gasIrradiance);
    images.emplace_back(&_shadowGridImage);
    images.emplace_back(&_occlusionImage);
    for (auto image : images) {
        image->layout = VK_IMAGE_LAYOUT_UNDEFINED;
        image->lastStages = VK_PIPELINE_STAGE_2_NONE;
        image->lastAccesses = VK_ACCESS_2_NONE;
    }
    _numAveragedFrames = 0;
    _lightComputed = false;
}

// The sizes shared with the shaders and the work group size are declared here, so that they cannot diverge
RayTracingComputePipeline _RayTracingResources::createPipeline(std::string const& source, IntVector2D const& workGroupSize) const
{
    static_assert(sizeof(ShadowGridNodeData) == 6 * sizeof(uint32_t), "The struct ShadowGridNode in Shaders/RayTracingCommonCS.h mirrors ShadowGridNodeData");
    auto declarations = std::format(
        "#version 460\n"
        "#extension GL_EXT_ray_query : require\n"
        "#extension GL_KHR_shader_subgroup_clustered : require\n"
        "layout(local_size_x = {}, local_size_y = {}, local_size_z = 1) in;\n"
        "const int MaxSourceLights = {};\n"
        "const float CellRadius = {:.6f};\n"
        "const float FreeCellRadius = {:.6f};\n"
        "const float SolidRadius = {:.6f};\n"
        "const float StrandHalfWidth = {:.6f};\n"
        "const int GasImageSize = {};\n"
        "const int GasLayers = {};\n"
        "const uint GasAccumulationStride = {}u;\n"
        "const uint GasLightClusterStride = {}u;\n"
        "const uint OcclusionStride = {}u;\n"
        "const int OcclusionLevels = {};\n"
        "const int GasIrradianceImageSize = {};\n"
        "const float LightFalloffDistance = {:.6f};\n"
        "const float SourceLightSpread = {:.6f};\n"
        "const int ShadowGridImageSize = {};\n"
        "const float ShadowGridFixedPointScale = {:.6f};\n",
        workGroupSize.x,
        workGroupSize.y,
        MAX_SOURCES,
        RayTracedObjectSizes::CellRadius,
        RayTracedObjectSizes::FreeCellRadius,
        RayTracedObjectSizes::SolidRadius,
        RayTracedObjectSizes::StrandHalfWidth,
        RayTracingLayout::MaxGasNodes,
        GasLayers,
        GasAccumulationStride,
        GasLightClusterStride,
        OcclusionStride,
        RayTracingLayout::OcclusionLevels,
        RayTracingLayout::GasIrradianceImageSize,
        RayTracingLayout::LightFalloffDistance,
        RayTracingLayout::SourceLightSpread,
        RayTracingLayout::MaxShadowGridNodes,
        ShadowGridNodeData::FixedPointScale);
    return _RayTracingComputePipeline::create(declarations + std::string(Shaders::RayTracingCommonCS) + source, _pipelineLayout, workGroupSize);
}

void _RayTracingResources::beginFrame(ExecutionParameters const& parameters)
{
    auto const& view = parameters._view;
    CHECK(_frameConstants.buffer != VK_NULL_HANDLE && _view.size == view.viewSize);

    auto const& textures = parameters._textures;
    CHECK(textures.size() == 2);
    VulkanBarriers()
        .add(textures.at(0)->color, ImageUsage::ShaderRead)
        .add(textures.at(1)->color, ImageUsage::ShaderRead)
        .record(parameters._renderInfo.commandBuffer);

    auto const& geometryBuffers = parameters._geometryBuffers;
    _shadowGrid = geometryBuffers->getShadowGridLayout();
    _tissueTraced = getTissueVisibility(view) > 0.0f;
    updateFrameConstants(view, geometryBuffers->getNumObjects(), *parameters._simulationParameters, parameters._simulationFacade);
    writeDescriptorSet(geometryBuffers, textures);
}

void _RayTracingResources::finishFrame()
{
    _numAveragedFrames = _stillPicture ? _numAveragedFrames + 1 : 0;
    _current = getPrevious();
}

void _RayTracingResources::bindDescriptorSet(VkCommandBuffer commandBuffer) const
{
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &_descriptorSet, 0, nullptr);
}

void _RayTracingResources::access(VkCommandBuffer commandBuffer, RayTracingAccesses const& accesses)
{
    auto clears = accesses | std::views::filter([](auto const& access) { return access.second == RayTracingAccess::Clear; }) | std::views::keys;
    if (!clears.empty()) {
        VulkanBarriers barriers;
        for (auto resource : clears) {
            barriers.add(getBuffer(resource), BufferUsage::ClearDestination);
        }
        barriers.record(commandBuffer);
        for (auto resource : clears) {
            vkCmdFillBuffer(commandBuffer, getBuffer(resource).buffer, 0, getUsedSize(resource), 0);
        }
    }

    VulkanBarriers barriers;
    for (auto const& [resource, access] : accesses) {
        auto write = access != RayTracingAccess::Read;
        if (auto image = getImage(resource)) {
            CHECK(access != RayTracingAccess::Clear);
            barriers.add(*image, write ? ImageUsage::ComputeReadWrite : ImageUsage::ComputeRead);
        } else if (resource == RayTracingResource::Scene) {
            CHECK(!write);
            barriers.add(_scene.topLevelBuffer, BufferUsage::RayQuery);
        } else {
            barriers.add(getBuffer(resource), write ? BufferUsage::ComputeReadWrite : BufferUsage::ComputeRead);
        }
    }
    barriers.record(commandBuffer);
}

VulkanImage& _RayTracingResources::getOutput()
{
    return isAntiAliasingNeeded() ? _view.antiAliased.at(_current) : _view.sceneColor;
}

RayTracingLayout::GasGrid const& _RayTracingResources::getGasGrid() const
{
    return _layout.gasGrid;
}

ShadowGridLayout const& _RayTracingResources::getShadowGrid() const
{
    return _shadowGrid;
}

IntVector2D const& _RayTracingResources::getViewSize() const
{
    return _view.size;
}

_RayTracingResources::_RayTracingResources()
{
    auto device = VulkanContext::get().getDevice();
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
    auto result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &_pipelineLayout);
    if (result != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, _descriptorSetLayout, nullptr);
        checkVkResult(result, "vkCreatePipelineLayout");
    }
}

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

_RayTracingResources::SceneResources _RayTracingResources::createSceneResources(uint64_t objectCapacity, uint64_t capsuleCapacity, uint64_t triangleCapacity)
{
    auto& context = VulkanContext::get();
    auto device = context.getDevice();
    SceneResources result{.objectCapacity = objectCapacity, .capsuleCapacity = capsuleCapacity, .triangleCapacity = triangleCapacity};
    try {
        result.spheres = context.createBuffer(objectCapacity * SphereSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VulkanMemory::DeviceLocal);
        result.sphereAabbs = context.createBuffer(objectCapacity * sizeof(VkAabbPositionsKHR), AabbBufferUsage, VulkanMemory::DeviceLocal);
        result.capsuleAabbs = context.createBuffer(capsuleCapacity * sizeof(VkAabbPositionsKHR), AabbBufferUsage, VulkanMemory::DeviceLocal);
        result.triangleAabbs = context.createBuffer(triangleCapacity * sizeof(VkAabbPositionsKHR), AabbBufferUsage, VulkanMemory::DeviceLocal);
        result.shadowCasters = context.createBuffer(
            (capsuleCapacity * CapsuleShadowEntries + triangleCapacity * TriangleShadowEntries) * ShadowCasterEntrySize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VulkanMemory::DeviceLocal);

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

void _RayTracingResources::destroySceneResources(SceneResources& resources, bool immediately)
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
             &resources.shadowCasters,
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

std::vector<VulkanImage*> _RayTracingResources::ViewResources::getImages()
{
    return {&sceneColor, &antiAliased.at(0), &antiAliased.at(1)};
}

_RayTracingResources::ViewResources _RayTracingResources::createViewResources(IntVector2D const& size)
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

void _RayTracingResources::destroyViewResources(ViewResources& resources, bool immediately)
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

void _RayTracingResources::createFrameResources()
{
    auto& context = VulkanContext::get();
    for (auto buffer : {&_gasAccumulation, &_gasIntermediate, &_gasLightClusters, &_occlusionAccumulation}) {
        context.destroyBufferLater(*buffer);
    }
    context.destroyImageLater(_gasImage);
    context.destroyImageLater(_gasIrradiance);
    context.destroyImageLater(_shadowGridImage);
    context.destroyImageLater(_occlusionImage);
    auto maxGasNodes = RayTracingLayout::MaxGasNodes;
    auto numNodes = static_cast<VkDeviceSize>(maxGasNodes) * maxGasNodes;
    auto clearableUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    _gasAccumulation = context.createBuffer(numNodes * GasAccumulationStride * sizeof(uint32_t), clearableUsage, VulkanMemory::DeviceLocal);

    // The blurs of the gas grid and of the shadow grid pass their intermediate results through the same buffer
    auto maxShadowGridNodes = RayTracingLayout::MaxShadowGridNodes;
    auto numShadowGridNodes = static_cast<VkDeviceSize>(maxShadowGridNodes) * maxShadowGridNodes;
    auto numIntermediateEntries = std::max(numNodes * GasLayers, numShadowGridNodes);
    _gasIntermediate = context.createBuffer(numIntermediateEntries * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VulkanMemory::DeviceLocal);

    auto maxLightClusters = RayTracingLayout::MaxLightClusters;
    auto numClusters = static_cast<VkDeviceSize>(maxLightClusters) * maxLightClusters;
    _gasLightClusters = context.createBuffer(numClusters * GasLightClusterStride * sizeof(uint32_t), clearableUsage, VulkanMemory::DeviceLocal);
    _gasImage = context.createImage({GasLayers * maxGasNodes, maxGasNodes}, VK_FORMAT_R16G16B16A16_SFLOAT, ViewImageUsage);
    auto irradianceImageSize = RayTracingLayout::GasIrradianceImageSize;
    _gasIrradiance = context.createImage({2 * irradianceImageSize, irradianceImageSize}, VK_FORMAT_R16G16B16A16_SFLOAT, ViewImageUsage);
    _shadowGridImage = context.createImage(getPyramidImageSize(maxShadowGridNodes), VK_FORMAT_R16G16B16A16_SFLOAT, ViewImageUsage);
    _occlusionAccumulation = context.createBuffer(numNodes * OcclusionStride * sizeof(uint32_t), clearableUsage, VulkanMemory::DeviceLocal);
    _occlusionImage = context.createImage(getPyramidImageSize(maxGasNodes), VK_FORMAT_R16G16B16A16_SFLOAT, ViewImageUsage);
    _lightComputed = false;

    // Created last since it marks the resources as complete
    _frameConstants = context.createBuffer(sizeof(FrameConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VulkanMemory::HostVisible);
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

float _RayTracingResources::getTissueVisibility(RenderView const& view)
{
    auto transition = getSmoothTransition(view.getScreenZoomFactor(), TissueZoomStart, TissueZoomEnd);
    return std::max(0.0f, (transition - TissueVisibilityStart) / (1.0f - TissueVisibilityStart));
}

float _RayTracingResources::getBackgroundScale()
{
    return 1.0f / SimulationRenderer::getBloomSceneWeight();
}

bool _RayTracingResources::isSourceLightActive(SimulationParameters const& parameters)
{
    for (auto index : std::views::iota(0, parameters.numSources)) {
        if (parameters.sourceLightBrightness.sourceValues[index] > 0.0f) {
            return true;
        }
    }
    return false;
}

bool _RayTracingResources::isSceneNeeded(RenderView const& view, SimulationParameters const& parameters)
{
    return getTissueVisibility(view) > 0.0f || isSourceLightActive(parameters);
}

void _RayTracingResources::updateFrameConstants(
    RenderView const& view,
    NumRenderObjects const& numObjects,
    SimulationParameters const& parameters,
    SimulationFacade const& simulationFacade)
{
    auto worldSize = simulationFacade->getWorldSize();
    auto const& topLeft = view.visibleWorldRect.topLeft;
    RealVector2D jitter;
    if (_stillPicture) {
        auto jitterIndex = _numAveragedFrames % NumJitterPositions + 1;
        jitter = {getHaltonValue(jitterIndex, 2) - 0.5f, getHaltonValue(jitterIndex, 3) - 0.5f};
    }
    auto screenZoom = view.getScreenZoomFactor();
    auto closeUpWeight = getSmoothTransition(screenZoom, CloseUpZoomStart, CloseUpZoomEnd);
    auto detailVisibility = getSmoothTransition(screenZoom, DetailZoomStart, DetailZoomEnd);

    auto const& gasGrid = _layout.gasGrid;
    auto shadowGridBlur = gasGrid.cloudSpreadInNodes * gasGrid.nodeDistance / _shadowGrid.nodeDistance;
    auto shadowGridBlurRadius = std::min(toInt(std::ceil(3.0f * shadowGridBlur)), MaxShadowGridBlurRadius);

    FrameConstants constants{
        .view = {toFloat(view.viewSize.x), toFloat(view.viewSize.y), view.zoomFactor, screenZoom},
        .rect = {topLeft.x, topLeft.y, jitter.x, jitter.y},
        .counts =
            {static_cast<uint32_t>(numObjects.objects),
             static_cast<uint32_t>(numObjects.lineIndices / 2),
             static_cast<uint32_t>(numObjects.triangleIndices / 3),
             0},
        .tissue = {closeUpWeight, detailVisibility, 1.0f / view.zoomFactor, getTissueVisibility(view)},
        .parameters =
            {getBackgroundScale(), _stillPicture ? 0.0f : 1.0f, 1.0f / toFloat(_numAveragedFrames + 1), SimulationRenderer::getSolidDetailVisibility(view)},
        .gasGrid = {gasGrid.origin.x, gasGrid.origin.y, gasGrid.nodeDistance, gasGrid.cloudSpreadInNodes},
        .gasSize = {gasGrid.size.x, gasGrid.size.y, gasGrid.blurRadius, static_cast<int32_t>(numObjects.fluidParticles)},
        .gasLight = {gasGrid.lightClusterSize, gasGrid.lightRange, gasGrid.irradianceNodeDistance, gasGrid.lightFalloffDistance},
        .gasLightGrid = {gasGrid.numLightClusters.x, gasGrid.numLightClusters.y, gasGrid.irradianceSize.x, gasGrid.irradianceSize.y},
        .gasLighting =
            {gasGrid.lightClusterOrigin.x, gasGrid.lightClusterOrigin.y, MaxFluidIllumination * parameters.fluidIllumination.value, gasGrid.sourceLightNodeDistance},
        .world = {toFloat(worldSize.x), toFloat(worldSize.y), parameters.borderlessRendering.value ? 1.0f : 0.0f, 0.0f},
        .shadowGrid = {_shadowGrid.origin.x, _shadowGrid.origin.y, _shadowGrid.nodeDistance, shadowGridBlur},
        .shadowGridSize = {_shadowGrid.size.x, _shadowGrid.size.y, shadowGridBlurRadius, 0},
        .numSourceLights = {static_cast<int32_t>(_layout.sourceLights.size()), 0, gasGrid.sourceLightSize.x, gasGrid.sourceLightSize.y},
    };
    for (auto&& [light, lightConstants] : std::views::zip(_layout.sourceLights, constants.sourceLights)) {
        lightConstants = SourceLightConstants{
            .centerRangeShape = {light.center.x, light.center.y, light.range, light.circular ? 0.0f : 1.0f},
            .color = {light.color.r, light.color.g, light.color.b, 0.0f},
            .extent = {light.halfExtent.x, light.halfExtent.y, 0.0f, 0.0f},
        };
    }
    std::memcpy(_frameConstants.mapped, &constants, sizeof(constants));

    // Apart from the objects, the light in the gas depends on the constants without the numbers of the objects
    auto lightConstants = constants;
    std::ranges::fill(lightConstants.counts, 0u);
    lightConstants.gasSize[3] = 0;
    updateLightReuse(std::as_bytes(std::span(&lightConstants, 1)), simulationFacade);
}

void _RayTracingResources::updateLightReuse(std::span<std::byte const> lightConstants, SimulationFacade const& simulationFacade)
{
    auto timestep = simulationFacade->getCurrentTimestep();
    _lightReused = _lightComputed && !_lightReused && !_stillPicture && simulationFacade->isSimulationRunning() && timestep >= _lightTimestep
        && timestep - _lightTimestep <= MaxLightDelay && std::ranges::equal(lightConstants, _lightConstants);
    if (!_lightReused) {
        _lightComputed = true;
        _lightTimestep = timestep;
        _lightConstants.assign(lightConstants.begin(), lightConstants.end());
    }
}

void _RayTracingResources::writeDescriptorSet(VulkanGeometryBuffers const& geometryBuffers, std::vector<TextureTarget> const& textures)
{
    _descriptorSet = VulkanContext::get().allocateFrameDescriptorSet(_descriptorSetLayout);
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
        .storageBuffer(Binding_ShadowGridNodes, geometryBuffers->getBuffer(GeometryBufferType_ShadowGrid))
        .storageImage(Binding_ShadowGridImage, _shadowGridImage.view)
        .sampledImage(Binding_ShadowGridSampler, _shadowGridImage.view, General)
        .storageBuffer(Binding_ShadowCasters, _scene.shadowCasters.buffer)
        .storageBuffer(Binding_OcclusionAccumulation, _occlusionAccumulation.buffer)
        .storageImage(Binding_OcclusionImage, _occlusionImage.view)
        .sampledImage(Binding_OcclusionSampler, _occlusionImage.view, General)
        .write(_descriptorSet);
}

void _RayTracingResources::buildScene(VkCommandBuffer commandBuffer, NumRenderObjects const& numObjects)
{
    VulkanBarriers()
        .add(_scene.sphereAabbs, BufferUsage::BuildInput)
        .add(_scene.capsuleAabbs, BufferUsage::BuildInput)
        .add(_scene.triangleAabbs, BufferUsage::BuildInput)
        .add(_scene.bottomLevelBuffer, BufferUsage::BuildOutput)
        .add(_scene.scratchBuffer, BufferUsage::BuildOutput)
        .record(commandBuffer);
    auto bottomLevelGeometries = createBottomLevelGeometries(_scene.sphereAabbs, _scene.capsuleAabbs, _scene.triangleAabbs);
    auto bottomLevelBuildInfo = createBuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, bottomLevelGeometries);
    bottomLevelBuildInfo.dstAccelerationStructure = _scene.bottomLevel;
    bottomLevelBuildInfo.scratchData.deviceAddress = getAlignedScratchAddress(_scene.scratchBuffer);
    std::array bottomLevelRanges{
        VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = static_cast<uint32_t>(numObjects.objects)},
        VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = static_cast<uint32_t>(numObjects.lineIndices / 2)},
        VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = static_cast<uint32_t>(numObjects.triangleIndices / 3)},
    };
    auto bottomLevelRangesPointer = bottomLevelRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &bottomLevelBuildInfo, &bottomLevelRangesPointer);

    VulkanBarriers()
        .add(_scene.bottomLevelBuffer, BufferUsage::BuildInput)
        .add(_scene.topLevelBuffer, BufferUsage::BuildOutput)
        .add(_scene.scratchBuffer, BufferUsage::BuildOutput)
        .record(commandBuffer);
    std::vector<VkAccelerationStructureGeometryKHR> topLevelGeometries{createInstanceGeometry(VulkanContext::get().getDeviceAddress(_scene.instanceBuffer))};
    auto topLevelBuildInfo = createBuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, topLevelGeometries);
    topLevelBuildInfo.dstAccelerationStructure = _scene.topLevel;
    topLevelBuildInfo.scratchData.deviceAddress = getAlignedScratchAddress(_scene.scratchBuffer);
    VkAccelerationStructureBuildRangeInfoKHR topLevelRange{.primitiveCount = 1};
    auto topLevelRangePointer = &topLevelRange;
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &topLevelBuildInfo, &topLevelRangePointer);
}

VulkanBuffer& _RayTracingResources::getBuffer(RayTracingResource resource)
{
    switch (resource) {
    case RayTracingResource::Spheres:
        return _scene.spheres;
    case RayTracingResource::SphereAabbs:
        return _scene.sphereAabbs;
    case RayTracingResource::CapsuleAabbs:
        return _scene.capsuleAabbs;
    case RayTracingResource::TriangleAabbs:
        return _scene.triangleAabbs;
    case RayTracingResource::ShadowCasters:
        return _scene.shadowCasters;
    case RayTracingResource::GasAccumulation:
        return _gasAccumulation;
    case RayTracingResource::GasIntermediate:
        return _gasIntermediate;
    case RayTracingResource::GasLightClusters:
        return _gasLightClusters;
    case RayTracingResource::OcclusionAccumulation:
        return _occlusionAccumulation;
    default:
        THROW_NOT_IMPLEMENTED();
    }
}

VulkanImage* _RayTracingResources::getImage(RayTracingResource resource)
{
    switch (resource) {
    case RayTracingResource::GasImage:
        return &_gasImage;
    case RayTracingResource::GasIrradiance:
        return &_gasIrradiance;
    case RayTracingResource::ShadowGridImage:
        return &_shadowGridImage;
    case RayTracingResource::OcclusionImage:
        return &_occlusionImage;
    case RayTracingResource::SceneColor:
        return &_view.sceneColor;
    case RayTracingResource::AntiAliased:
        return &_view.antiAliased.at(_current);
    case RayTracingResource::PreviousAntiAliased:
        return &_view.antiAliased.at(getPrevious());
    default:
        return nullptr;
    }
}

VkDeviceSize _RayTracingResources::getUsedSize(RayTracingResource resource) const
{
    switch (resource) {
    case RayTracingResource::GasAccumulation:
        return static_cast<VkDeviceSize>(_layout.gasGrid.size.x) * _layout.gasGrid.size.y * GasAccumulationStride * sizeof(uint32_t);
    case RayTracingResource::GasLightClusters:
        return static_cast<VkDeviceSize>(_layout.gasGrid.numLightClusters.x) * _layout.gasGrid.numLightClusters.y * GasLightClusterStride * sizeof(uint32_t);
    case RayTracingResource::OcclusionAccumulation:
        return static_cast<VkDeviceSize>(_layout.gasGrid.size.x) * _layout.gasGrid.size.y * OcclusionStride * sizeof(uint32_t);
    default:
        THROW_NOT_IMPLEMENTED();
    }
}

int _RayTracingResources::getPrevious() const
{
    return 1 - _current;
}
