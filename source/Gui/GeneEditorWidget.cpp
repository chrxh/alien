#include "GeneEditorWidget.h"

#include <imgui.h>

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr HeaderMinRightColumnWidth = 160.0f;
    auto constexpr HeaderMaxLeftColumnWidth = 200.0f;
    auto constexpr HeaderMinColumnWidth = 300.0f;
}


GeneEditorWidget _GeneEditorWidget::create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
{
    return GeneEditorWidget(new _GeneEditorWidget(editData, layoutData));
}

void _GeneEditorWidget::process()
{
    if (ImGui::BeginChild("GeneEditor", ImVec2(0, 0))) {
        if (_editData->selectedGeneIndex.has_value()) {
            ImGui::PushID(_editData->selectedGeneIndex.value());
            processHeaderData();
            ImGui::PopID();
        } else {
            processNoSelection();
        }
    }
    ImGui::EndChild();
}

_GeneEditorWidget::_GeneEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
    : _editData(editData)
    , _layoutData(layoutData)
{}

void _GeneEditorWidget::processNoSelection()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Gene").highlighted(true));
    if (ImGui::BeginChild("overlay", ImVec2(0, 0), 0)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        AlienGui::DisabledField();
        auto text = "No gene is selected";
        auto textSize = ImGui::CalcTextSize(text);
        ImVec2 textPos(startPos.x + size.x / 2 - textSize.x / 2, startPos.y + size.y / 2 - textSize.y / 2);
        ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }
    ImGui::EndChild();
}

void _GeneEditorWidget::processHeaderData()
{
    auto const& selectedGene = _editData->getSelectedGeneRef();
    auto title = selectedGene._name.empty() ? "Gene " + std::to_string(_editData->selectedGeneIndex.value()) : selectedGene._name;
    AlienGui::Group(AlienGui::GroupParameters().text(title).highlighted(true));

    if (ImGui::BeginChild("GeneHeader", ImVec2(0, 0), 0, 0)) {
        auto& gene = _editData->getSelectedGeneRef();

        _editData->updateGeometry();  // Do it every time in order to avoid check for changes

        AlienGui::DynamicTableLayout table(HeaderMinColumnWidth);
        if (table.begin()) {

            auto rightColumnWidth = std::max(HeaderMinRightColumnWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(HeaderMaxLeftColumnWidth)));

            AlienGui::Group(AlienGui::GroupParameters().text("Base properties"));

            // Gene name
            AlienGui::InputText(AlienGui::InputTextParameters().name("Gene name").textWidth(rightColumnWidth), gene._name);

            // Shape
            if (AlienGui::Combo(
                    AlienGui::ComboParameters().name("Shape generator").values(Const::ConstructorShapeStrings).textWidth(rightColumnWidth),
                    gene._shape)) {
                {
                    ShapeGenerator shapeGenerator;
                    if (_editData->selectedGeneIndex.value() == 0) {
                        _editData->genome._frontAngle = shapeGenerator.getPreferredFrontAngle(gene._shape);
                    }
                }
            }

            // Connection distance
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Connection distance").format("%.2f").step(0.05f).textWidth(rightColumnWidth), gene._connectionDistance);

            // Stiffness
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Stiffness").format("%.2f").step(0.05f).textWidth(rightColumnWidth), gene._stiffness);

            // Homogeneous cell type
            AlienGui::Checkbox(
                AlienGui::CheckboxParameters()
                    .name("Homogeneous cell type")
                    .textWidth(rightColumnWidth)
                    .tooltip("If enabled, every constructed cell of this gene uses the cell type and its properties of the first node."),
                gene._homogeneousCellType);

            table.next();
            table.end();
        }
    }
    ImGui::EndChild();
}
