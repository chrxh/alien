#include "SelectionHud.h"

#include <algorithm>
#include <cmath>

#include <imgui.h>

#include <Fonts/AlienIconFont.h>
#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "MultiplierTool.h"
#include "SimulationInteractionController.h"
#include "SimulationView.h"
#include "StyleService.h"
#include "Viewport.h"

namespace
{
    auto constexpr FramePadding = 10.0f;
    auto constexpr FrameThickness = 1.0f;
    auto constexpr CornerHandleSize = 7.0f;
    auto constexpr HandleRadius = 7.0f;
    auto constexpr HandleHitMargin = 4.0f;
    auto constexpr HandleThickness = 1.5f;
    auto constexpr RotationHandleDistance = 30.0f;
    auto constexpr SummarySpacing = 8.0f;
    auto constexpr SummaryPaddingX = 8.0f;
    auto constexpr SummaryPaddingY = 3.0f;
    auto constexpr SummaryRounding = 10.0f;
    auto constexpr ActionBarSpacing = 10.0f;
    auto constexpr ActionBarPadding = 5.0f;
    auto constexpr ActionBarMargin = 10.0f;
    auto constexpr ColorFieldSize = 24.0f;
    auto constexpr MultiplyPopupWidth = 470.0f;
    auto constexpr MorePopupWidth = 360.0f;
    auto constexpr TransformTextWidth = 130.0f;
}

void SelectionHud::process()
{
    auto const& model = EditorModel::get();
    if (!SimulationInteractionController::get().isEditMode() || !SimulationView::get().isRenderSimulation() || model.getTool() != EditTool_Select
        || model.isSelectionEmpty()) {
        _lastRotationAngle.reset();
        return;
    }

    auto bounds = calcViewBounds();
    processFrame(bounds);
    processRotationHandle(bounds);
    processSummary(bounds);
    processActionBar(bounds);
}

SelectionHud::ViewBounds SelectionHud::calcViewBounds() const
{
    auto const& model = EditorModel::get();
    auto selectionBounds = model.getSelectionBounds(model.isApplyToNetworks());
    auto zoom = Viewport::get().getZoomFactor();
    auto borderlessRendering = _SimulationFacade::get()->getSimulationParameters().borderlessRendering.value;
    auto center = Viewport::get().mapWorldToViewPosition(selectionBounds.center, borderlessRendering);
    auto padding = RealVector2D{scale(FramePadding), scale(FramePadding)};
    return ViewBounds{
        .center = center,
        .topLeft = center + (selectionBounds.topLeft - selectionBounds.center) * zoom - padding,
        .bottomRight = center + (selectionBounds.bottomRight - selectionBounds.center) * zoom + padding,
    };
}

void SelectionHud::processFrame(ViewBounds const& bounds) const
{
    auto drawList = ImGui::GetBackgroundDrawList();
    drawList->AddRect(
        {bounds.topLeft.x, bounds.topLeft.y}, {bounds.bottomRight.x, bounds.bottomRight.y}, Const::SelectionFrameColor, 0, 0, scale(FrameThickness));

    auto halfSize = scale(CornerHandleSize) / 2;
    for (auto const& corner : {
             bounds.topLeft,
             RealVector2D{bounds.bottomRight.x, bounds.topLeft.y},
             RealVector2D{bounds.topLeft.x, bounds.bottomRight.y},
             bounds.bottomRight,
         }) {
        drawList->AddRectFilled({corner.x - halfSize, corner.y - halfSize}, {corner.x + halfSize, corner.y + halfSize}, Const::SelectionHandleFillColor);
        drawList->AddRect(
            {corner.x - halfSize, corner.y - halfSize}, {corner.x + halfSize, corner.y + halfSize}, Const::SelectionHandleColor, 0, 0, scale(HandleThickness));
    }
}

namespace
{
    struct HandleState
    {
        bool hovered = false;
        bool active = false;
    };

    HandleState processHandle(char const* id, RealVector2D const& pos, std::string const& tooltip)
    {
        auto size = 2 * scale(HandleRadius + HandleHitMargin);
        ImGui::SetNextWindowPos({pos.x - size / 2, pos.y - size / 2}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({size, size}, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(1.0f, 1.0f));
        HandleState result;
        auto flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove
            | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
        if (ImGui::Begin(id, nullptr, flags)) {
            ImGui::InvisibleButton("##handle", {size, size});
            result.hovered = ImGui::IsItemHovered();
            result.active = ImGui::IsItemActive();
            if (!result.active) {
                AlienGui::Tooltip(tooltip, false);
            }
        }
        ImGui::End();
        ImGui::PopStyleVar(3);
        return result;
    }

    void drawHandle(RealVector2D const& pos, HandleState const& state, ImColor const& color)
    {
        auto drawList = ImGui::GetBackgroundDrawList();
        auto borderColor = state.hovered || state.active ? Const::SelectionHandleHoveredColor : color;
        drawList->AddCircleFilled({pos.x, pos.y}, scale(HandleRadius), Const::SelectionHandleFillColor);
        drawList->AddCircle({pos.x, pos.y}, scale(HandleRadius), borderColor, 0, scale(HandleThickness));
    }

    float calcAngle(RealVector2D const& from, ImVec2 const& to)
    {
        return toFloat(std::atan2(to.y - from.y, to.x - from.x) * Const::RadToDeg);
    }
}

void SelectionHud::processRotationHandle(ViewBounds const& bounds)
{
    auto frameTop = RealVector2D{(bounds.topLeft.x + bounds.bottomRight.x) / 2, bounds.topLeft.y};
    auto handlePos = RealVector2D{frameTop.x, frameTop.y - scale(RotationHandleDistance)};

    ImGui::GetBackgroundDrawList()->AddLine(
        {frameTop.x, frameTop.y}, {handlePos.x, handlePos.y + scale(HandleRadius)}, Const::SelectionFrameColor, scale(FrameThickness));

    auto state = processHandle("##selectionRotationHandle", handlePos, "Drag to rotate the selection");
    drawHandle(handlePos, state, Const::SelectionHandleColor);

    if (!state.active) {
        _lastRotationAngle.reset();
        return;
    }
    auto angle = calcAngle(bounds.center, ImGui::GetMousePos());
    if (_lastRotationAngle.has_value()) {
        auto angleDelta = Math::getNormalizedAngle(angle - *_lastRotationAngle, -180.0f);
        if (std::abs(angleDelta) > NEAR_ZERO) {
            EditorController::get().onRotateSelectedObjects(angleDelta);
        }
    }
    _lastRotationAngle = angle;
}

void SelectionHud::processSummary(ViewBounds const& bounds) const
{
    auto const& model = EditorModel::get();
    auto const& selection = model.getSelectionShallowData();
    auto numObjects = model.isApplyToNetworks() ? selection.numClusterCells : selection.numObjects;

    std::vector<std::string> parts;
    if (numObjects > 0) {
        parts.emplace_back(StringHelper::format(static_cast<uint64_t>(numObjects)) + (numObjects == 1 ? " object" : " objects"));
    }
    if (selection.numCreatures > 0) {
        parts.emplace_back(StringHelper::format(static_cast<uint64_t>(selection.numCreatures)) + (selection.numCreatures == 1 ? " creature" : " creatures"));
    }
    if (selection.numEnergyParticles > 0) {
        parts.emplace_back(
            StringHelper::format(static_cast<uint64_t>(selection.numEnergyParticles))
            + (selection.numEnergyParticles == 1 ? " energy particle" : " energy particles"));
    }
    std::string text;
    for (auto const& part : parts) {
        text += text.empty() ? part : "  \xC2\xB7  " + part;
    }

    auto textSize = ImGui::CalcTextSize(text.c_str());
    auto topLeft = ImVec2{bounds.topLeft.x, bounds.bottomRight.y + scale(SummarySpacing)};
    auto bottomRight = ImVec2{topLeft.x + textSize.x + 2 * scale(SummaryPaddingX), topLeft.y + textSize.y + 2 * scale(SummaryPaddingY)};
    auto drawList = ImGui::GetBackgroundDrawList();
    drawList->AddRectFilled(topLeft, bottomRight, Const::FloatingCardBackgroundColor, scale(SummaryRounding));
    drawList->AddRect(topLeft, bottomRight, Const::FloatingCardBorderColor, scale(SummaryRounding));
    drawList->AddText({topLeft.x + scale(SummaryPaddingX), topLeft.y + scale(SummaryPaddingY)}, Const::SelectionChipTextColor, text.c_str());
}

void SelectionHud::processActionBar(ViewBounds const& bounds)
{
    if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::GetIO().WantCaptureMouse) {
        return;
    }

    auto& controller = EditorController::get();
    auto const& model = EditorModel::get();
    auto popupButton = [](std::string const& icon, std::string const& name, char const* popupId) {
        return AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters().icon(icon).name(name).action([popupId] { ImGui::OpenPopup(popupId); }));
    };
    auto actionButton = [](std::string const& icon, std::string const& name, std::function<void()> const& action, bool disabled = false) {
        return AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters().icon(icon).name(name).disabled(disabled).action(action));
    };
    std::vector<AlienGui::ToolbarItem> items = {
        popupButton(ICON_COLOR, "Color", "##hudColor"),
        popupButton(ICON_STICKY, "Stickiness", "##hudSticky"),
        popupButton(ICON_FIXED, "Static", "##hudStatic"),
        AlienGui::ToolbarItem::createSeparator(),
        actionButton(ICON_UNIFORM_VELOCITY, "Make uniform velocities", [&controller] { controller.onUniformVelocities(); }),
        actionButton(
            ICON_RELEASE_STRESSES, "Release stresses", [&controller] { controller.onReleaseStresses(); }, model.isCellSelectionEmpty()),
        actionButton(
            ICON_GLUE_SELECTION,
            "Glue selection: connects neighboring objects within the selection",
            [&controller] { controller.onGlueSelectedObjects(); },
            model.isCellSelectionEmpty()),
        AlienGui::ToolbarItem::createSeparator(),
        popupButton(ICON_MULTIPLY, "Multiply", "##hudMultiply"),
        AlienGui::ToolbarItem::createSeparator(),
        actionButton(ICON_FA_COPY, "Copy (CTRL+C)", [&controller] { controller.onCopy(); }),
        actionButton(
            ICON_FA_TRASH, "Delete (DEL)", [&controller] { controller.onDelete(); }, !controller.isDeletingPossible()),
        AlienGui::ToolbarItem::createSeparator(),
        popupButton(ICON_INSPECT, "Inspect", "##hudInspect"),
        popupButton(ICON_MORE, "Transform: position, velocity and rotation", "##hudMore"),
    };

    auto viewport = ImGui::GetMainViewport();
    auto width = AlienGui::CalcToolbarWidth(items) + 2 * scale(ActionBarPadding) + 1.0f;
    auto height = AlienGui::CalcToolbarHeight() + 2 * scale(ActionBarPadding);
    auto margin = scale(ActionBarMargin);
    auto posX = (bounds.topLeft.x + bounds.bottomRight.x) / 2 - width / 2;
    posX = std::clamp(posX, viewport->WorkPos.x + margin, std::max(viewport->WorkPos.x + margin, viewport->WorkPos.x + viewport->WorkSize.x - width - margin));
    auto posY = bounds.topLeft.y - scale(RotationHandleDistance + HandleRadius + ActionBarSpacing) - height;
    if (posY < viewport->WorkPos.y + margin) {
        posY = bounds.bottomRight.y + scale(SummarySpacing) + ImGui::GetTextLineHeight() + 2 * scale(SummaryPaddingY + ActionBarSpacing);
    }
    posY = std::clamp(posY, viewport->WorkPos.y + margin, std::max(viewport->WorkPos.y + margin, viewport->WorkPos.y + viewport->WorkSize.y - height - margin));

    ImGui::SetNextWindowPos({posX, posY}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Always);
    AlienGui::PushFloatingCardStyle(ActionBarPadding);
    auto flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoFocusOnAppearing;
    if (ImGui::Begin("##selectionActionBar", nullptr, flags)) {
        AlienGui::PopFloatingCardStyle();
        AlienGui::Toolbar(AlienGui::ToolbarParameters().id("SelectionActions").bottomSeparator(false), items);
        processColorPopup();
        processStickyPopup();
        processStaticPopup();
        processMultiplyPopup();
        processInspectPopup();
        processMorePopup();
    } else {
        AlienGui::PopFloatingCardStyle();
    }
    ImGui::End();
}

void SelectionHud::processColorPopup()
{
    if (!ImGui::BeginPopup("##hudColor")) {
        return;
    }
    auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;
    for (int i = 0; i < MAX_COLORS; ++i) {
        if (i > 0) {
            ImGui::SameLine();
        }
        ImGui::PushID(i);
        if (AlienGui::ColorField(customizationColors.values[i].toRgbColor(), ColorFieldSize, ColorFieldSize)) {
            EditorController::get().onColorSelectedObjects(i);
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopID();
    }
    ImGui::EndPopup();
}

void SelectionHud::processStickyPopup()
{
    if (!ImGui::BeginPopup("##hudSticky")) {
        return;
    }
    if (ImGui::Selectable("Make sticky")) {
        EditorController::get().onSetSticky(true);
    }
    if (ImGui::Selectable("Make unsticky")) {
        EditorController::get().onSetSticky(false);
    }
    ImGui::EndPopup();
}

void SelectionHud::processStaticPopup()
{
    if (!ImGui::BeginPopup("##hudStatic")) {
        return;
    }
    if (ImGui::Selectable("Make static")) {
        EditorController::get().onSetStatic(true);
    }
    if (ImGui::Selectable("Make non-static")) {
        EditorController::get().onSetStatic(false);
    }
    ImGui::EndPopup();
}

void SelectionHud::processMultiplyPopup()
{
    ImGui::SetNextWindowSize({scale(MultiplyPopupWidth), 0.0f});
    if (!ImGui::BeginPopup("##hudMultiply")) {
        return;
    }
    MultiplierTool::get().processContent();
    ImGui::EndPopup();
}

void SelectionHud::processInspectPopup()
{
    if (!ImGui::BeginPopup("##hudInspect")) {
        return;
    }
    auto& controller = EditorController::get();
    if (ImGui::Selectable("Objects (ALT+N)", false, controller.isObjectInspectionPossible() ? 0 : ImGuiSelectableFlags_Disabled)) {
        controller.onInspectSelectedObjects();
    }
    if (ImGui::Selectable("Genomes (ALT+F)", false, controller.isGenomeInspectionPossible() ? 0 : ImGuiSelectableFlags_Disabled)) {
        controller.onInspectSelectedGenomes();
    }
    if (ImGui::Selectable("Creatures (ALT+P)", false, controller.isCreatureInspectionPossible() ? 0 : ImGuiSelectableFlags_Disabled)) {
        controller.onInspectSelectedCreatures();
    }
    ImGui::EndPopup();
}

void SelectionHud::processMorePopup()
{
    auto const& model = EditorModel::get();
    auto const& selection = model.getSelectionShallowData();
    if (_lastSelection.has_value()
        && (_lastSelection->numObjects != selection.numObjects || _lastSelection->numEnergyParticles != selection.numEnergyParticles)) {
        _angle = 0;
        _angularVelocity = 0;
    }
    _lastSelection = selection;

    ImGui::SetNextWindowSize({scale(MorePopupWidth), 0.0f});
    if (!ImGui::BeginPopup("##hudMore")) {
        return;
    }
    auto& controller = EditorController::get();
    auto bounds = model.getSelectionBounds(model.isApplyToNetworks());

    AlienGui::Group(AlienGui::GroupParameters().text("Center position and velocity"));
    auto position = bounds.center;
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Position X").textWidth(TransformTextWidth).format("%.3f"), position.x);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Position Y").textWidth(TransformTextWidth).format("%.3f"), position.y);
    if (position != bounds.center) {
        controller.onMoveSelectedObjectsBy(position - bounds.center);
    }

    auto velocity = bounds.velocity;
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Velocity X").textWidth(TransformTextWidth).step(0.1f).format("%.3f"), velocity.x);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Velocity Y").textWidth(TransformTextWidth).step(0.1f).format("%.3f"), velocity.y);
    if (velocity != bounds.velocity) {
        controller.onSetVelocityOfSelectedObjects(velocity);
    }

    AlienGui::Group(AlienGui::GroupParameters().text("Center rotation"));
    auto origAngle = _angle;
    AlienGui::SliderInputFloat(
        AlienGui::SliderInputFloatParameters().name("Angle").textWidth(TransformTextWidth).inputWidth(50.0f).min(-180.0f).max(180.0f).format("%.1f"), _angle);
    if (_angle != origAngle) {
        controller.onRotateSelectedObjects(_angle - origAngle);
    }

    auto origAngularVelocity = _angularVelocity;
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Angular velocity").textWidth(TransformTextWidth).step(0.01f).format("%.2f"), _angularVelocity);
    if (_angularVelocity != origAngularVelocity) {
        controller.onSetAngularVelocityOfSelectedObjects(_angularVelocity);
    }
    ImGui::EndPopup();
}
