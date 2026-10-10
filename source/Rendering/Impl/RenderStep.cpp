#include "RenderStep.h"

#include <cstddef>
#include <cstring>
#include <ranges>

#include <Base/Interface/Math.h>

#include <Data/Interface/CellTypeConstants.h>

#include <Engine/Interface/GeometryBuffers.h>
#include <Engine/Interface/SimulationFacade.h>

#include "RenderGraph.h"
#include "Shader.h"

namespace
{
    auto constexpr ZoomFactorForCellDetails = 25.0f;  // Cell type strings and arrows
    auto constexpr CellTypeLabelShadowOffset = 1.0f;
    auto constexpr CellTypeLabelShadowAlpha = 0.8f;
    auto constexpr DetonationLifetime = 0.1f;  // In seconds
    auto constexpr DepthFormat = VK_FORMAT_D32_SFLOAT;
}

TextureTarget _TextureTarget::create()
{
    return TextureTarget(new _TextureTarget());
}

_TextureTarget::~_TextureTarget()
{
    destroyImages();
}

void _TextureTarget::resize(IntVector2D const& size, VkFormat colorFormat)
{
    // A used depth image is recreated right away, so that a lack of memory shows up before any commands are recorded
    auto withDepth = _depth.image != VK_NULL_HANDLE;
    destroyImages();
    color =
        VulkanContext::get().createImage(size, colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (withDepth) {
        getDepth();
    }
}

VulkanImage& _TextureTarget::getDepth()
{
    if (_depth.image == VK_NULL_HANDLE) {
        _depth = VulkanContext::get().createImage(color.size, DepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    return _depth;
}

void _TextureTarget::resetImageStates()
{
    for (auto image : {&color, &_depth}) {
        image->layout = VK_IMAGE_LAYOUT_UNDEFINED;
        image->lastStages = VK_PIPELINE_STAGE_2_NONE;
        image->lastAccesses = VK_ACCESS_2_NONE;
    }
}

void _TextureTarget::destroyImages()
{
    VulkanContext::get().destroyImageLater(color);
    VulkanContext::get().destroyImageLater(_depth);
}

_RenderStep::_RenderStep(StepParameters const& parameters, DepthTest depthTest)
    : _previousTargetSelection(parameters._previousTargetSelection)
    , _textureScale(parameters._textureScale)
    , _uniforms(parameters._uniforms)
    , _uniformFunc(parameters._uniformFunc)
    , _depthTest(depthTest)
{
    if (!parameters._shader.vertex.empty()) {
        _shader = _Shader::createFromSource(parameters._shader.vertex, parameters._shader.fragment, parameters._shader.geometry);
    }
}

StepParameters& StepParameters::addUniform(std::string const& key, UniformValueType const& value)
{
    _uniforms.emplace(key, value);
    return *this;
}

std::optional<int> const& _RenderStep::getPreviousTargetSelection() const
{
    return _previousTargetSelection;
}

void _RenderStep::prepareExecution(ExecutionParameters const& parameters, std::vector<TextureTarget> const& sampledTextures)
{
    auto worldSize = parameters._simulationFacade->getWorldSize();
    auto const& view = parameters._view;
    auto const& worldRect = view.visibleWorldRect;
    auto viewSize = view.viewSize;
    auto zoom = view.zoomFactor;
    auto renderScale = view.renderScale;

    _shader->setFloat("zoom", zoom);
    _shader->setFloat("renderScale", renderScale);
    _shader->setFloat("radius", std::max(parameters._minBallRadius * renderScale, zoom));
    _shader->setVec2("worldSize", toRealVector2D(worldSize));
    _shader->setVec2("rectUpperLeft", worldRect.topLeft);
    _shader->setVec2("rectLowerRight", worldRect.bottomRight);
    _shader->setVec2("viewportSize", toRealVector2D(viewSize));

    auto uniforms = _uniforms;
    if (_uniformFunc) {
        auto uniformFunc = _uniformFunc(*parameters._simulationParameters, view);
        uniforms.insert(uniformFunc.begin(), uniformFunc.end());
    }
    for (auto const& [key, value] : uniforms) {
        if (std::holds_alternative<int>(value)) {
            _shader->setInt(key, std::get<int>(value));
        }
        if (std::holds_alternative<float>(value)) {
            _shader->setFloat(key, std::get<float>(value));
        }
        if (std::holds_alternative<FloatColorRGB>(value)) {
            _shader->setVec3(key, std::get<FloatColorRGB>(value));
        }
    }

    // Barriers are not allowed during rendering
    auto commandBuffer = parameters._renderInfo.commandBuffer;
    auto const& target = parameters._target;
    auto withDepth = _depthTest != DepthTest::None;
    VulkanImageBarriers barriers;
    for (auto const& texture : sampledTextures) {
        barriers.add(texture->color, ImageUsage::ShaderRead);
    }
    barriers.add(target->color, ImageUsage::ColorAttachment);
    if (withDepth) {
        barriers.add(target->getDepth(), ImageUsage::DepthAttachment);
    }
    barriers.record(commandBuffer);

    auto loadOp = parameters._clearBackground ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = target->color.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = loadOp,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
    };
    VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = withDepth ? target->getDepth().view : VK_NULL_HANDLE,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = loadOp,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.depthStencil = {1.0f, 0}},
    };
    auto targetExtent = VkExtent2D{static_cast<uint32_t>(target->color.size.x), static_cast<uint32_t>(target->color.size.y)};
    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, targetExtent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = withDepth ? &depthAttachment : nullptr,
    };
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{0, 0, toFloat(viewSize.x) * _textureScale, toFloat(viewSize.y) * _textureScale, 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, targetExtent};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

void _RenderStep::finishExecution(ExecutionParameters const& parameters)
{
    vkCmdEndRendering(parameters._renderInfo.commandBuffer);
}

void _RenderStep::draw(ExecutionParameters const& parameters, PipelineState state, VkBuffer vertexBuffer, uint64_t numElements, VkBuffer indexBuffer)
{
    if (numElements == 0) {
        return;
    }
    auto commandBuffer = parameters._renderInfo.commandBuffer;
    state.depthTest = _depthTest;
    state.colorFormat = parameters._target->color.format;
    state.depthFormat = _depthTest != DepthTest::None ? DepthFormat : VK_FORMAT_UNDEFINED;
    _shader->bind(commandBuffer, state);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);
    if (indexBuffer != VK_NULL_HANDLE) {
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(numElements), 1, 0, 0, 0);
    } else {
        vkCmdDraw(commandBuffer, static_cast<uint32_t>(numElements), 1, 0, 0);
    }
}

CellRenderStep _NonFluidObjectRenderStep::create(StepParameters const& parameters)
{
    return CellRenderStep(new _NonFluidObjectRenderStep(parameters));
}

void _NonFluidObjectRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, .vertexLayout = VertexLayout::Objects, .blendMode = BlendMode::AlphaAdditive},
        geometryBuffers->getBuffer(GeometryBufferType_Objects),
        geometryBuffers->getNumObjects().objects);

    finishExecution(parameters);
}

_NonFluidObjectRenderStep::_NonFluidObjectRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

LineRenderStep _LineRenderStep::create(StepParameters const& parameters)
{
    return LineRenderStep(new _LineRenderStep(parameters));
}

void _LineRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // The geometry shader converts the lines to quads with proper width
    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST, .vertexLayout = VertexLayout::Objects, .blendMode = BlendMode::AlphaBlend},
        geometryBuffers->getBuffer(GeometryBufferType_Objects),
        geometryBuffers->getNumObjects().lineIndices,
        geometryBuffers->getBuffer(GeometryBufferType_LineIndices));

    finishExecution(parameters);
}

_LineRenderStep::_LineRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters, DepthTest::LessOrEqual)
{}

TriangleRenderStep _TriangleRenderStep::create(StepParameters const& parameters)
{
    return TriangleRenderStep(new _TriangleRenderStep(parameters));
}

void _TriangleRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .vertexLayout = VertexLayout::Objects},
        geometryBuffers->getBuffer(GeometryBufferType_Objects),
        geometryBuffers->getNumObjects().triangleIndices,
        geometryBuffers->getBuffer(GeometryBufferType_TriangleIndices));

    finishExecution(parameters);
}

_TriangleRenderStep::_TriangleRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters, DepthTest::Less)
{}

struct FullscreenQuad
{
    VulkanBuffer vertices;
    VulkanBuffer indices;
};

namespace
{
    float const QuadVertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // Top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // Bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // Bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // Top left
    };
    unsigned int const QuadIndices[] = {0, 1, 3, 1, 2, 3};

    VulkanBuffer createHostBuffer(void const* data, VkDeviceSize size, VkBufferUsageFlags usage)
    {
        auto result = VulkanContext::get().createBuffer(size, usage, VulkanMemory::HostVisible);
        std::memcpy(result.mapped, data, size);
        return result;
    }

    // Shared by all post-processing steps and released with the last one
    std::weak_ptr<FullscreenQuad> sharedFullscreenQuad;

    std::shared_ptr<FullscreenQuad> getFullscreenQuad()
    {
        if (auto result = sharedFullscreenQuad.lock()) {
            return result;
        }
        auto result = std::shared_ptr<FullscreenQuad>(
            new FullscreenQuad{
                .vertices = createHostBuffer(QuadVertices, sizeof(QuadVertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT),
                .indices = createHostBuffer(QuadIndices, sizeof(QuadIndices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT),
            },
            [](FullscreenQuad* quad) {
                VulkanContext::get().destroyBufferLater(quad->vertices);
                VulkanContext::get().destroyBufferLater(quad->indices);
                delete quad;
            });
        sharedFullscreenQuad = result;
        return result;
    }
}

PostProcessingRenderStep _PostProcessingRenderStep::create(StepParameters const& parameters)
{
    return PostProcessingRenderStep(new _PostProcessingRenderStep(parameters));
}

void _PostProcessingRenderStep::execute(ExecutionParameters parameters)
{
    parameters._clearBackground = false;

    auto const& textures = parameters._textures;
    auto numTextures = textures.size();
    CHECK(numTextures <= 3);
    prepareExecution(parameters, textures);

    _shader->setInt("numTextures", toInt(numTextures));
    for (auto const& [number, texture] : std::views::zip(std::views::iota(1), textures)) {
        _shader->setTexture("inputTexture" + std::to_string(number), texture->color);
    }

    draw(parameters, {.vertexLayout = VertexLayout::FullscreenQuad}, _fullscreenQuad->vertices.buffer, 6, _fullscreenQuad->indices.buffer);

    finishExecution(parameters);
}

_PostProcessingRenderStep::_PostProcessingRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
    , _fullscreenQuad(getFullscreenQuad())
{}

ForwardRenderStep _ForwardRenderStep::create(StepParameters const& parameters)
{
    return ForwardRenderStep(new _ForwardRenderStep(parameters));
}

void _ForwardRenderStep::execute(ExecutionParameters parameters)
{
    // Do nothing
}

_ForwardRenderStep::_ForwardRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

FluidParticleRenderStep _FluidParticleRenderStep::create(StepParameters const& parameters)
{
    return FluidParticleRenderStep(new _FluidParticleRenderStep(parameters));
}

void _FluidParticleRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    parameters._minBallRadius = 0.0f;
    prepareExecution(parameters);

    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, .vertexLayout = VertexLayout::FluidParticles, .blendMode = BlendMode::AlphaAdditive},
        geometryBuffers->getBuffer(GeometryBufferType_FluidParticles),
        geometryBuffers->getNumObjects().fluidParticles);

    finishExecution(parameters);
}

_FluidParticleRenderStep::_FluidParticleRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

LocationRenderStep _LocationRenderStep::create(StepParameters const& parameters)
{
    return LocationRenderStep(new _LocationRenderStep(parameters));
}

void _LocationRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);
    _shader->setBool("borderlessRendering", parameters._simulationParameters->borderlessRendering.value);

    // The geometry shader converts the location points to quads
    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, .vertexLayout = VertexLayout::Locations, .blendMode = BlendMode::AlphaBlend},
        geometryBuffers->getBuffer(GeometryBufferType_Locations),
        geometryBuffers->getNumObjects().locations);

    finishExecution(parameters);
}

_LocationRenderStep::_LocationRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

SelectedObjectRenderStep _SelectedObjectRenderStep::create(StepParameters const& parameters)
{
    return SelectedObjectRenderStep(new _SelectedObjectRenderStep(parameters));
}

void _SelectedObjectRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // The geometry shader converts the selected objects to quads
    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, .vertexLayout = VertexLayout::SelectedObjects, .blendMode = BlendMode::AlphaBlend},
        geometryBuffers->getBuffer(GeometryBufferType_SelectedObjects),
        geometryBuffers->getNumObjects().selectedObjects);

    finishExecution(parameters);
}

_SelectedObjectRenderStep::_SelectedObjectRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

CellTypeOverlayRenderStep _CellTypeOverlayRenderStep::create(StepParameters const& parameters, ImFont* labelFont)
{
    return CellTypeOverlayRenderStep(new _CellTypeOverlayRenderStep(parameters, labelFont));
}

_CellTypeOverlayRenderStep::~_CellTypeOverlayRenderStep()
{
    VulkanContext::get().destroyImageLater(_cellTypeTextureAtlas);
}

void _CellTypeOverlayRenderStep::execute(ExecutionParameters parameters)
{
    // Only render if zoom exceeds threshold and overlay is active
    if (parameters._view.getScreenZoomFactor() <= ZoomFactorForCellDetails || !parameters._view.cellDetailOverlay) {
        return;
    }

    // Don't clear background - we want to composite on top of existing rendering
    parameters._clearBackground = false;
    prepareExecution(parameters);

    // The geometry shader converts the points to textured quads
    _shader->setTexture("overlayTexture", _cellTypeTextureAtlas);
    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, .vertexLayout = VertexLayout::Objects, .blendMode = BlendMode::AlphaBlend},
        geometryBuffers->getBuffer(GeometryBufferType_Objects),
        geometryBuffers->getNumObjects().objects);

    finishExecution(parameters);
}

_CellTypeOverlayRenderStep::_CellTypeOverlayRenderStep(StepParameters const& parameters, ImFont* labelFont)
    : _RenderStep(parameters)
{
    createCellTypeTextureAtlas(labelFont);
}

namespace
{
    void blendPixel(std::vector<uint8_t>& pixels, int index, float brightness, float alpha)
    {
        auto backgroundAlpha = toFloat(pixels.at(index + 3)) / 255.0f;
        auto resultAlpha = alpha + backgroundAlpha * (1.0f - alpha);
        if (resultAlpha == 0) {
            return;
        }
        for (auto channel : std::views::iota(0, 3)) {
            auto backgroundColor = toFloat(pixels.at(index + channel)) / 255.0f;
            auto resultColor = (brightness * alpha + backgroundColor * backgroundAlpha * (1.0f - alpha)) / resultAlpha;
            pixels.at(index + channel) = static_cast<uint8_t>(resultColor * 255.0f);
        }
        pixels.at(index + 3) = static_cast<uint8_t>(resultAlpha * 255.0f);
    }
}

void _CellTypeOverlayRenderStep::createCellTypeTextureAtlas(ImFont* labelFont)
{
    // Create a texture atlas containing all cell type strings and object type strings
    // We'll arrange them in a vertical strip, one per row
    // Rows 0-12: Cell types (Base, Depot, Sensor, etc.)
    // Row 13: "Solid" (for ObjectType_Solid)
    // Row 14: "Fluid" (for ObjectType_Fluid)
    // Row 15: "Free Cell" (for ObjectType_FreeCell)
    float fontSize = 16.0f;  // Base font size for rendering

    // Build combined list of labels: cell types + object types (Solid, Fluid, Free Cell)
    std::vector<std::string> allLabels;
    for (auto const& cellTypeStr : Const::CellTypeStrings) {
        allLabels.push_back(cellTypeStr);
    }
    // Add object type labels (only Solid, Fluid, and Free Cell, as Cell uses cell type names)
    allLabels.push_back(Const::ObjectTypeStrings[ObjectType_Solid]);
    allLabels.push_back(Const::ObjectTypeStrings[ObjectType_Fluid]);
    allLabels.push_back(Const::ObjectTypeStrings[ObjectType_FreeCell]);

    int textureWidth = 512;   // Fixed width
    int textureHeight = 512;  // Fixed height, should be enough for all labels

    // Create pixel buffer (clear to transparent)
    std::vector<uint8_t> pixels(textureWidth * textureHeight * 4, 0);  // RGBA

    // Get font atlas data
    int atlasWidth, atlasHeight;
    unsigned char* atlasData;
    labelFont->ContainerAtlas->GetTexDataAsAlpha8(&atlasData, &atlasWidth, &atlasHeight);

    // Render each label string to the buffer using ImGui font
    int rowHeight = 20;
    float scale = fontSize / labelFont->FontSize;

    auto renderLabel = [&](std::string const& label, float startPosX, float posY, float brightness, float alphaFactor) {
        auto posX = startPosX;
        for (auto const& character : label) {
            auto glyph = labelFont->FindGlyph(static_cast<ImWchar>(character));
            CHECK(glyph);

            // Calculate glyph position and size
            auto x0 = posX + glyph->X0 * scale;
            auto y0 = posY + glyph->Y0 * scale;
            auto x1 = posX + glyph->X1 * scale;
            auto y1 = posY + glyph->Y1 * scale;

            // Render glyph to our texture buffer
            for (auto py = toInt(y0); py <= toInt(y1); ++py) {
                for (auto px = toInt(x0); px <= toInt(x1); ++px) {
                    // Calculate texture coordinate in font atlas
                    auto tu = glyph->U0 + (glyph->U1 - glyph->U0) * ((px - x0) / (x1 - x0));
                    auto tv = glyph->V0 + (glyph->V1 - glyph->V0) * ((py - y0) / (y1 - y0));

                    auto atlasX = toInt(tu * atlasWidth);
                    auto atlasY = toInt(tv * atlasHeight);
                    if (atlasX < 0 || atlasX >= atlasWidth || atlasY < 0 || atlasY >= atlasHeight) {
                        continue;
                    }

                    auto alpha = toFloat(atlasData[atlasY * atlasWidth + atlasX]) / 255.0f;
                    if (alpha > 0) {
                        blendPixel(pixels, (py * textureWidth + px) * 4, brightness, alpha * alphaFactor);
                    }
                }
            }

            // Advance position for next character
            posX += glyph->AdvanceX * scale;
        }
    };

    auto rowIndex = 0;
    for (auto const& label : allLabels) {
        auto posX = 5.0f;
        auto posY = toFloat(rowIndex * rowHeight) + 2.0f;

        // A dark copy behind the white text keeps the labels readable on bright backgrounds
        renderLabel(label, posX + CellTypeLabelShadowOffset, posY + CellTypeLabelShadowOffset, 0.0f, CellTypeLabelShadowAlpha);
        renderLabel(label, posX, posY, 1.0f, 1.0f);
        ++rowIndex;
    }

    _cellTypeTextureAtlas = VulkanContext::get().createSampledImage(pixels.data(), {textureWidth, textureHeight});
}

SelectedConnectionRenderStep _SelectedConnectionRenderStep::create(StepParameters const& parameters)
{
    return SelectedConnectionRenderStep(new _SelectedConnectionRenderStep(parameters));
}

void _SelectedConnectionRenderStep::execute(ExecutionParameters parameters)
{
    if (parameters._view.getScreenZoomFactor() <= ZoomFactorForCellDetails) {
        return;
    }

    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // The geometry shader converts the connections to lines with arrows
    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST, .vertexLayout = VertexLayout::SelectedConnections, .blendMode = BlendMode::AlphaAdditive},
        geometryBuffers->getBuffer(GeometryBufferType_SelectedConnections),
        geometryBuffers->getNumObjects().connectionArrowVertices);

    finishExecution(parameters);
}

_SelectedConnectionRenderStep::_SelectedConnectionRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

AttackEventRenderStep _AttackEventRenderStep::create(StepParameters const& parameters)
{
    return AttackEventRenderStep(new _AttackEventRenderStep(parameters));
}

void _AttackEventRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // The geometry shader converts the attack event lines to dashed quads
    auto const& geometryBuffers = parameters._geometryBuffers;
    draw(
        parameters,
        {.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST, .vertexLayout = VertexLayout::AttackEvents, .blendMode = BlendMode::AlphaAdditive},
        geometryBuffers->getBuffer(GeometryBufferType_AttackEvents),
        geometryBuffers->getNumObjects().attackEventVertices);

    finishExecution(parameters);
}

_AttackEventRenderStep::_AttackEventRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

DetonationEventRenderStep _DetonationEventRenderStep::create(StepParameters const& parameters)
{
    return DetonationEventRenderStep(new _DetonationEventRenderStep(parameters));
}

_DetonationEventRenderStep::~_DetonationEventRenderStep()
{
    VulkanContext::get().destroyBufferLater(_instanceBuffer);
}

namespace
{
    struct DetonationInstance
    {
        float pos[2];
        float radius;
        float age;
    };
}

void _DetonationEventRenderStep::execute(ExecutionParameters parameters)
{
    auto now = std::chrono::steady_clock::now();
    updateDetonations(parameters._geometryBuffers, now);

    std::vector<DetonationInstance> instances;
    for (auto const& detonation : _detonations | std::views::values) {
        auto age = std::chrono::duration<float>(now - detonation.startTime).count();
        if (age < DetonationLifetime) {
            instances.emplace_back(DetonationInstance{
                .pos = {detonation.pos.x, detonation.pos.y},
                .radius = detonation.radius,
                .age = age,
            });
        }
    }

    parameters._clearBackground = true;
    auto const& textures = parameters._textures;
    prepareExecution(parameters, textures);

    if (!instances.empty()) {
        auto sizeInBytes = instances.size() * sizeof(DetonationInstance);
        if (_instanceBuffer.size < sizeInBytes) {
            VulkanContext::get().destroyBufferLater(_instanceBuffer);
            _instanceBuffer = VulkanContext::get().createBuffer(sizeInBytes * 2, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VulkanMemory::HostVisible);
        }
        std::memcpy(_instanceBuffer.mapped, instances.data(), sizeInBytes);

        _shader->setFloat("lifetime", DetonationLifetime);
        _shader->setTexture("inputTexture1", textures.at(0)->color);
        draw(
            parameters,
            {.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST, .vertexLayout = VertexLayout::DetonationInstances, .blendMode = BlendMode::Additive},
            _instanceBuffer.buffer,
            instances.size());
    }

    finishExecution(parameters);
}

void _DetonationEventRenderStep::updateDetonations(GeometryBuffers const& geometryBuffers, std::chrono::steady_clock::time_point now)
{
    for (auto& detonation : _detonations | std::views::values) {
        detonation.reported = false;
    }

    // An event lasts several timesteps and is reported in each of them, but the animation starts only once
    for (auto const& event : geometryBuffers->getDetonationEventData()) {
        auto newDetonation = Detonation{.pos = {event.pos[0], event.pos[1]}, .radius = event.radius, .startTime = now};
        _detonations.try_emplace(event.objectId, newDetonation).first->second.reported = true;
    }

    std::erase_if(_detonations, [&](auto const& entry) {
        auto const& detonation = entry.second;
        return !detonation.reported && now - detonation.startTime > std::chrono::duration<float>(DetonationLifetime);
    });
}

_DetonationEventRenderStep::_DetonationEventRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}
