#include "SymbolsWindow.h"

#include <algorithm>
#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/StringHelper.h"
#include "EngineInterface/CellComputationCompiler.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"
#include "StyleRepository.h"
#include "OpenSymbolsDialog.h"
#include "SaveSymbolsDialog.h"

namespace
{
    auto const ContentTextInputWidth = 60.0f;
}

_SymbolsWindow::_SymbolsWindow(SimulationController const& simController)
    : _AlienWindow("Symbols", "editor.symbols", false)
    , _simController(simController)
{
    _openSymbolsDialog = std::make_shared<_OpenSymbolsDialog>(simController);
    _saveSymbolsDialog = std::make_shared<_SaveSymbolsDialog>(simController);
    onClearEditFields();
}

void _SymbolsWindow::processIntern()
{
    auto entries = getEntriesFromSymbolMap(_simController->getSymbolMap());

    //new button
    if (AlienImGui::ToolbarButton(ICON_FA_ERASER)) {
        entries.clear();
    }

    //load button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
        _openSymbolsDialog->show();
    }

    //save button
    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
        _saveSymbolsDialog->show();
    }

    //undo button
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSymbolMapChanged());
    if (AlienImGui::ToolbarButton(ICON_FA_UNDO)) {
        entries = getEntriesFromSymbolMap(_simController->getOriginalSymbolMap());
    }
    ImGui::EndDisabled();

    AlienImGui::Separator();

    if (ImGui::BeginTable(
            "Symbols",
            4,
            ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable
                | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_ScrollY,
            ImVec2(0.0f, ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(135)))) {

        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoHide, 0.0f);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoHide, 0.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoHide, 0.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoHide, 0.0f);
        ImGui::TableHeadersRow();

        int i = 0;
        for(auto const& entry : entries) {
            ImGui::PushID(i++);

            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);

            if (AlienImGui::Button(ICON_FA_PEN)) {
                onEditEntry(entry);
            }
            ImGui::SameLine();
            if (AlienImGui::Button(ICON_FA_TRASH)) {
                onDeleteEntry(entries, entry.name);
            }

            ImGui::TableSetColumnIndex(1);
            ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
            AlienImGui::Text(entry.name);
            
            ImGui::TableSetColumnIndex(2);
            if (entry.type == SymbolType::Constant) {
                AlienImGui::Text("Constant");
            } else {
                AlienImGui::Text("Variable");
            }
            
            ImGui::TableSetColumnIndex(3);
            AlienImGui::Text(entry.value);
            ImGui::PopFont();
            ImGui::PopID();
        }

        ImGui::EndTable();
    }
    if (_mode == Mode::Create) {
        AlienImGui::Group("Create symbol");
    }
    if (_mode == Mode::Edit) {
        AlienImGui::Group("Edit symbol");
    }
    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name("Name").textWidth(ContentTextInputWidth).monospaceFont(true), _symbolName, IM_ARRAYSIZE(_symbolName));
    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name("Value").textWidth(ContentTextInputWidth).monospaceFont(true), _symbolValue, IM_ARRAYSIZE(_symbolValue));
    AlienImGui::Separator();
    if (_mode == Mode::Create) {
        ImGui::BeginDisabled(!isEditValid());
        if (AlienImGui::Button("Add")) {
            onAddEntry(entries, std::string(_symbolName), std::string(_symbolValue));
            onClearEditFields();
        }
        ImGui::EndDisabled();
    }
    if (_mode == Mode::Edit) {
        ImGui::BeginDisabled(!isEditValid());
        if (AlienImGui::Button("Update")) {
            onUpdateEntry(entries, std::string(_symbolName), std::string(_symbolValue));
            onClearEditFields();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (AlienImGui::Button("Cancel")) {
            _mode = Mode::Create;
            onClearEditFields();
        }
    }
    updateSymbolMapFromEntries(entries);

    _openSymbolsDialog->process();
    _saveSymbolsDialog->process();
}

auto _SymbolsWindow::getEntriesFromSymbolMap(SymbolMap const& symbolMap) const -> std::vector<Entry>
{
    std::vector<Entry> result;
    for (auto const& [key, value] : symbolMap) {
        Entry entry;
        entry.name = key;
        entry.type = getSymbolType(value);
        entry.value = value;
        result.emplace_back(entry);
    }
    return result;
}

void _SymbolsWindow::updateSymbolMapFromEntries(std::vector<Entry> const& entries)
{
    SymbolMap symbolMap;
    for (auto const& entry : entries) {
        symbolMap.insert_or_assign(entry.name, entry.value);
    }
    _simController->setSymbolMap(symbolMap);
}

bool _SymbolsWindow::hasSymbolMapChanged() const
{
    return _simController->getSymbolMap() != _simController->getOriginalSymbolMap();
}

bool _SymbolsWindow::isEditValid() const
{
    return _symbolName[0] != '\0' && _symbolValue[0] != '\0';
}

auto _SymbolsWindow::getSymbolType(std::string const& value) const -> SymbolType
{
    return CellComputationCompiler::extractAddress(value).has_value() ? SymbolType::Variable : SymbolType::Constant;
}

void _SymbolsWindow::onClearEditFields()
{
    _mode = Mode::Create;
    _symbolName[0] = 0;
    _symbolValue[0] = 0;
}

void _SymbolsWindow::onEditEntry(Entry const& entry)
{
    _mode = Mode::Edit;
    _origSymbolName = entry.name;
    StringHelper::copy(_symbolName, IM_ARRAYSIZE(_symbolName), entry.name);
    StringHelper::copy(_symbolValue, IM_ARRAYSIZE(_symbolValue), entry.value);
}

void _SymbolsWindow::onAddEntry(std::vector<Entry>& entries, std::string const& name, std::string const& value) const
{
    Entry newEntry;
    newEntry.name = std::string(_symbolName);
    newEntry.value = std::string(_symbolValue);
    newEntry.type = getSymbolType(newEntry.value);
    entries.emplace_back(newEntry);
}

void _SymbolsWindow::onUpdateEntry(std::vector<Entry>& entries, std::string const& name, std::string const& value) const
{
    onDeleteEntry(entries, _origSymbolName);
    onAddEntry(entries, name, value);
}

void _SymbolsWindow::onDeleteEntry(std::vector<Entry>& entries, std::string const& name) const
{
    auto removeResult = std::remove_if(entries.begin(), entries.end(), [&](Entry const& entry) { return entry.name == name; });
    if (removeResult != entries.end()) {
        entries.erase(removeResult);
    }
}
