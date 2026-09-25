#include "EditToolbar.h"

#include <imgui.h>

#include <Fonts/AlienIconFont.h>
#include <Fonts/IconsFontAwesome5.h>

#include "AlienGui.h"
#include "CreatorTool.h"
#include "ImageToPatternDialog.h"
#include "SimulationInteractionController.h"
#include "SimulationView.h"
#include "StyleService.h"

namespace
{
    auto constexpr DockSpacing = 14.0f;
    auto constexpr DockPadding = 6.0f;
    auto constexpr OptionsSpacing = 8.0f;
    auto constexpr OptionsPadding = 10.0f;
    auto constexpr OptionsWidth = 380.0f;
    auto constexpr OptionsTextWidth = 140.0f;
    auto constexpr FloatingCardFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoFocusOnAppearing;

    struct ToolDefinition
    {
        EditTool tool;
        std::string icon;
        std::string name;
        std::string shortcut;
        ImGuiKey key;
    };
    std::vector<ToolDefinition> const ToolDefinitions = {
        {EditTool_Select, ICON_FA_MOUSE_POINTER, "Select and move", "V", ImGuiKey_V},
        {EditTool_Scissors, ICON_FA_CUT, "Scissors", "X", ImGuiKey_X},
        {EditTool_Object, ICON_DOT, "Single object", "O", ImGuiKey_O},
        {EditTool_Rectangle, ICON_RECTANGLE, "Rectangular object network", "R", ImGuiKey_R},
        {EditTool_Hexagon, ICON_HEXAGON, "Hexagonal object network", "H", ImGuiKey_H},
        {EditTool_Disc, ICON_DISC, "Disc-shaped object network", "D", ImGuiKey_D},
        {EditTool_Line, ICON_FA_SLASH, "Object network along a line", "L", ImGuiKey_L},
        {EditTool_Curve, ICON_FA_BEZIER_CURVE, "Object network along a Bezier curve", "C", ImGuiKey_C},
        {EditTool_Polygon, ICON_FA_DRAW_POLYGON, "Polygon-shaped object network", "P", ImGuiKey_P},
        {EditTool_Freehand, ICON_FA_PAINT_BRUSH, "Draw freehand", "B", ImGuiKey_B},
    };
}

void EditToolbar::process()
{
    if (!SimulationInteractionController::get().isEditMode() || !SimulationView::get().isRenderSimulation()) {
        return;
    }
    processDock();
    processToolOptions();
    processShortcuts();
}

void EditToolbar::processDock()
{
    auto& model = EditorModel::get();

    std::vector<AlienGui::ToolbarItem> items;
    for (auto const& definition : ToolDefinitions) {
        items.emplace_back(AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                                   .icon(definition.icon)
                                                                   .name(definition.name)
                                                                   .tooltip(definition.name + " (" + definition.shortcut + ")")
                                                                   .selected(model.getTool() == definition.tool)
                                                                   .action([this, tool = definition.tool] { selectTool(tool); })));
        if (definition.tool == EditTool_Scissors) {
            items.emplace_back(AlienGui::ToolbarItem::createSeparator());
        }
    }
    items.emplace_back(AlienGui::ToolbarItem::createSeparator());
    items.emplace_back(AlienGui::ToolbarItem::createButton(
        AlienGui::ToolbarItemParameters().icon(ICON_FA_IMAGE).name("Pattern from image").tooltip("Create a pattern from an image (I)").action([] {
            ImageToPatternDialog::get().show();
        })));
    items.emplace_back(AlienGui::ToolbarItem::createSeparator());
    items.emplace_back(AlienGui::ToolbarItem::createButton(
        AlienGui::ToolbarItemParameters()
            .icon(ICON_FA_DOT_CIRCLE)
            .name("Apply to selected objects")
            .tooltip("Apply to selected objects\n\nEdits only affect the selected objects. When they are moved, their connections to unselected "
                     "objects can only tear.\nHold SHIFT to switch to entire networks temporarily.")
            .selected(!model.isApplyToNetworks())
            .action([&model] { model.setApplyToNetworks(false); })));
    items.emplace_back(AlienGui::ToolbarItem::createButton(
        AlienGui::ToolbarItemParameters()
            .icon(ICON_FA_PROJECT_DIAGRAM)
            .name("Apply to entire networks")
            .tooltip("Apply to entire networks\n\nEdits affect the whole object networks of the selected objects.\nHold SHIFT to switch to selected "
                     "objects temporarily.")
            .selected(model.isApplyToNetworks())
            .action([&model] { model.setApplyToNetworks(true); })));
    items.emplace_back(AlienGui::ToolbarItem::createSeparator());
    items.emplace_back(
        AlienGui::ToolbarItem::createButton(AlienGui::ToolbarItemParameters()
                                                .icon(ICON_FA_MAGNET)
                                                .name("Glue on contact")
                                                .tooltip("Glue on contact\n\nIf enabled, moved or rotated objects connect to the objects they touch.")
                                                .selected(model.isGlueOnContact())
                                                .action([&model] { model.setGlueOnContact(!model.isGlueOnContact()); })));

    auto anchor = SimulationInteractionController::get().getEditToggleAnchor();
    auto width = AlienGui::CalcToolbarWidth(items) + 2 * scale(DockPadding) + 1.0f;
    auto height = AlienGui::CalcToolbarHeight() + 2 * scale(DockPadding);
    _dockTop = anchor.y - height / 2;

    ImGui::SetNextWindowPos({anchor.x + scale(DockSpacing), _dockTop}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Always);
    AlienGui::PushFloatingCardStyle(DockPadding);
    if (ImGui::Begin("##editDock", nullptr, FloatingCardFlags)) {
        AlienGui::Toolbar(AlienGui::ToolbarParameters().id("EditDock").bottomSeparator(false), items);
    }
    ImGui::End();
    AlienGui::PopFloatingCardStyle();
}

namespace
{
    ToolDefinition const& getToolDefinition(EditTool tool)
    {
        for (auto const& definition : ToolDefinitions) {
            if (definition.tool == tool) {
                return definition;
            }
        }
        return ToolDefinitions.front();
    }
}

void EditToolbar::processToolOptions()
{
    auto const& model = EditorModel::get();
    auto tool = model.getTool();
    if (tool == EditTool_Select) {
        return;
    }

    auto anchor = SimulationInteractionController::get().getEditToggleAnchor();
    ImGui::SetNextWindowPos({anchor.x + scale(DockSpacing), _dockTop - scale(OptionsSpacing)}, ImGuiCond_Always, {0.0f, 1.0f});
    ImGui::SetNextWindowSizeConstraints({scale(OptionsWidth), 0.0f}, {scale(OptionsWidth), FLT_MAX});
    AlienGui::PushFloatingCardStyle(OptionsPadding);
    if (ImGui::Begin("##editToolOptions", nullptr, FloatingCardFlags | ImGuiWindowFlags_AlwaysAutoResize)) {
        AlienGui::Group(AlienGui::GroupParameters().text(getToolDefinition(tool).name));

        std::string hint;
        if (tool == EditTool_Scissors) {
            auto cutOnlyInSelection = model.isCutOnlyInSelection();
            AlienGui::Checkbox(
                AlienGui::CheckboxParameters()
                    .name("Only in selection")
                    .textWidth(OptionsTextWidth)
                    .tooltip("If enabled, only connections between selected objects are cut."),
                &cutOnlyInSelection);
            EditorModel::get().setCutOnlyInSelection(cutOnlyInSelection);
            hint = "Hold the left mouse button and drag across connections to cut them.";
        } else {
            CreatorTool::get().processOptions();
            if (tool == EditTool_Line || tool == EditTool_Curve || tool == EditTool_Polygon) {
                hint = "Left click: add point, right click: remove last point, ENTER: finish, ESC: abort";
            } else if (tool == EditTool_Freehand) {
                hint = "Hold the left mouse button to draw.";
            } else {
                hint = "Left click in the simulation to place.";
            }
        }
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(hint.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }
    ImGui::End();
    AlienGui::PopFloatingCardStyle();
}

void EditToolbar::processShortcuts()
{
    auto const& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || io.WantTextInput || io.KeyCtrl || io.KeyAlt || io.KeySuper) {
        return;
    }
    if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
        return;
    }

    for (auto const& definition : ToolDefinitions) {
        if (ImGui::IsKeyPressed(definition.key, false)) {
            selectTool(definition.tool);
        }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_I, false)) {
        ImageToPatternDialog::get().show();
    }

    auto& creatorTool = CreatorTool::get();
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) {
        creatorTool.onFinishPoints();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) && creatorTool.hasPoints()) {
        creatorTool.onAbortPoints();
    }
}

void EditToolbar::selectTool(EditTool tool)
{
    EditorModel::get().setTool(tool);
}
