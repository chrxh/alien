#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include "ImguiMemoryEditor/imgui_memory_editor.h"
#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "EngineInterface/CellComputationCompiler.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"

using namespace std::string_literals;

namespace
{
    auto const MaxCellContentTextWidth = 110.0f;
    auto const MaxParticleContentTextWidth = 80.0f;
    auto const CellFunctions =
        std::vector{"Computation"s, "Propulsion"s, "Scanner"s, "Digestion"s, "Constructor"s, "Sensor"s, "Muscle"s};
}

_InspectorWindow::_InspectorWindow(
    SimulationController const& simController,
    Viewport const& viewport,
    EditorModel const& editorModel,
    uint64_t entityId,
    RealVector2D const& initialPos)
    : _entityId(entityId)
    , _initialPos(initialPos)
    , _viewport(viewport)
    , _editorModel(editorModel)
    , _simController(simController)
{
    _cellInstructionMemoryEdit = boost::make_shared<MemoryEditor>();
    _cellInstructionMemoryEdit->OptShowOptions = false;
    _cellInstructionMemoryEdit->OptShowAscii = false;
    _cellInstructionMemoryEdit->OptMidColsCount = 0;
    _cellInstructionMemoryEdit->Cols = 8;

    _cellDataMemoryEdit = boost::make_shared<MemoryEditor>();
    _cellDataMemoryEdit->OptShowOptions = false;
    _cellDataMemoryEdit->OptShowAscii = false;
    _cellDataMemoryEdit->OptMidColsCount = 0;
    _cellDataMemoryEdit->Cols = 8;

    auto const& parameters = _simController->getSimulationParameters();
    for (int i = 0; i < parameters.tokenMemorySize; ++i) {
        auto tokenMemoryEdit = boost::make_shared<MemoryEditor>();
        tokenMemoryEdit->OptShowOptions = false;
        tokenMemoryEdit->OptShowAscii = false;
        tokenMemoryEdit->OptMidColsCount = 0;
        tokenMemoryEdit->Cols = 8;
        _tokenMemoryEdits.emplace_back(tokenMemoryEdit);
    }
}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    if (!_on) {
        return;
    }
    auto width = calcWindowWidth();
    auto height = isCell() ? StyleRepository::getInstance().scaleContent(330.0f)
                           : StyleRepository::getInstance().scaleContent(70.0f);
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    auto entity = _editorModel->getInspectedEntity(_entityId);
    if (ImGui::Begin(generateTitle().c_str(), &_on)) {
        auto windowPos = ImGui::GetWindowPos();
        if (isCell()) {
            processCell(std::get<CellDescription>(entity));
        } else {
            processParticle(std::get<ParticleDescription>(entity));
        }
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(entity));
        auto factor = StyleRepository::getInstance().scaleContent(1);

        drawList->AddLine(
            {windowPos.x + 15.0f * factor, windowPos.y - 5.0f * factor},
            {entityPos.x, entityPos.y},
            Const::InspectorLineColor,
            1.5f);
        drawList->AddRectFilled(
            {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor},
            {windowPos.x + 25.0f * factor, windowPos.y},
            Const::InspectorRectColor,
            1.0,
            0);
        drawList->AddRect(
            {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor},
            {windowPos.x + 25.0f * factor, windowPos.y},
            Const::InspectorLineColor,
            1.0,
            0,
            2.0f);
    }
    ImGui::End();
}

bool _InspectorWindow::isClosed() const
{
    return !_on;
}

uint64_t _InspectorWindow::getId() const
{
    return _entityId;
}

bool _InspectorWindow::isCell() const
{
    auto entity = _editorModel->getInspectedEntity(_entityId);
    return std::holds_alternative<CellDescription>(entity);
}

std::string _InspectorWindow::generateTitle() const
{
    auto entity = _editorModel->getInspectedEntity(_entityId);
    std::stringstream ss;
    if (isCell()) {
        ss << "Cell #" << std::hex << _entityId;
    } else {
        ss << "Energy particle #" << std::hex << _entityId;
    }
    return ss.str();
}

namespace
{
    bool hasChanges(CellDescription const& left, CellDescription const& right)
    {
        return left.energy != right.energy || left.maxConnections != right.maxConnections || left.tokenBlocked != right.tokenBlocked
            || left.tokenBranchNumber != right.tokenBranchNumber || left.cellFeature.getType() != right.cellFeature.getType()
            || left.cellFeature.constData != right.cellFeature.constData || left.cellFeature.volatileData != right.cellFeature.volatileData
            || left.metadata.computerSourcecode != right.metadata.computerSourcecode
            || left.metadata.name != right.metadata.name || left.metadata.description != right.metadata.description
            || left.tokens != right.tokens;
    }
}

void _InspectorWindow::processCell(CellDescription cell)
{
    if (ImGui::BeginTabBar(
            "##CellInspect", /*ImGuiTabBarFlags_AutoSelectNewTabs | */ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        auto origCell = cell;
        showCellGeneralTab(cell);
        if (cell.cellFeature.getType() == Enums::CellFunction::COMPUTATION) {
            showCellCodeTab(cell);
            showCellMemoryTab(cell);
        } else {
            showCellInOutTab(cell);
        }
        for (int i = 0; i < cell.tokens.size(); ++i) {
            showTokenTab(cell.tokens.at(i), i);
        }
        auto const& parameters = _simController->getSimulationParameters();
        if (cell.tokens.size() < parameters.cellMaxToken) {
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_SetSelected)) {
                addToken(cell);
            }
            AlienImGui::Tooltip("Add token");
        }
        ImGui::EndTabBar();

        if (hasChanges(cell, origCell)) {
            if (cell.cellFeature != origCell.cellFeature) {
                cell.metadata.computerSourcecode.clear();
            }
            _simController->changeCell(cell);
        }
    }
}

void _InspectorWindow::showCellGeneralTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("General", nullptr, ImGuiTabItemFlags_None)) {
        AlienImGui::Group("Properties");
        auto const& parameters = _simController->getSimulationParameters();
        int type = static_cast<int>(cell.cellFeature.getType());
        AlienImGui::Combo(
            AlienImGui::ComboParameters()
                              .name("Specialization")
                              .values(CellFunctions)
                              .textWidth(MaxCellContentTextWidth), type);
        cell.cellFeature.setType(static_cast<Enums::CellFunction::Type>(type));

        auto energy = toFloat(cell.energy);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(MaxCellContentTextWidth), energy);
        cell.energy = energy;

        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Max connections")
                .textWidth(MaxCellContentTextWidth)
                .max(parameters.cellMaxBonds)
                .min(0),
            cell.maxConnections);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Branch number")
                .textWidth(MaxCellContentTextWidth)
                .max(parameters.cellMaxTokenBranchNumber - 1)
                .min(0),
            cell.tokenBranchNumber);
        AlienImGui::Checkbox(
            AlienImGui::CheckBoxParameters().name("Block token").textWidth(MaxCellContentTextWidth), cell.tokenBlocked);

        AlienImGui::Group("Metadata");

        cell.metadata.name.copy(_cellName, cell.metadata.name.size());
        _cellName[cell.metadata.name.size()] = 0;
        AlienImGui::InputText(
            AlienImGui::InputTextParameters().name("Name").textWidth(MaxCellContentTextWidth),
            _cellName,
            IM_ARRAYSIZE(_cellName));
        cell.metadata.name = std::string(_cellName);

        cell.metadata.description.copy(_cellDescription, cell.metadata.description.size());
        _cellDescription[cell.metadata.description.size()] = 0;
        AlienImGui::InputTextMultiline(
            AlienImGui::InputTextMultilineParameters().name("Notes").textWidth(MaxCellContentTextWidth).height(0),
            _cellDescription,
            IM_ARRAYSIZE(_cellDescription));
        cell.metadata.description = std::string(_cellDescription);

        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellCodeTab(CellDescription& cell)
{
    ImGuiTabItemFlags flags = 0;
    if (ImGui::BeginTabItem("Code", nullptr, flags)) {
        auto origSourcecode = [&] {
            if (cell.metadata.computerSourcecode.empty()) {
                return CellComputationCompiler::decompileSourceCode(
                    cell.cellFeature.constData,
                    _simController->getSymbolMap(),
                    _simController->getSimulationParameters());
            }
            return cell.metadata.computerSourcecode;
        }();
        origSourcecode.copy(_cellCode, std::min(toInt(origSourcecode.length()), IM_ARRAYSIZE(_cellCode) - 1), 0);
        _cellCode[origSourcecode.length()] = '\0';
        ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
        ImGui::InputTextMultiline(
            "##source",
            _cellCode,
            IM_ARRAYSIZE(_cellCode),
            {ImGui::GetContentRegionAvail().x,
             ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(20)},
            ImGuiInputTextFlags_AllowTabInput);
        ImGui::PopFont();

        //compilation state
        auto sourcecode = std::string(_cellCode);
        if (sourcecode != origSourcecode || !_lastCompilationResult) {
            _lastCompilationResult = boost::make_shared<CompilationResult>(
                CellComputationCompiler::compileSourceCode(sourcecode, _simController->getSymbolMap()));
            if (_lastCompilationResult->compilationOk) {
                cell.cellFeature.constData = _lastCompilationResult->compilation;
            }
            cell.metadata.computerSourcecode = sourcecode;
        }
        showCompilationResult(*_lastCompilationResult);
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellMemoryTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Memory", nullptr, ImGuiTabItemFlags_None)) {
        auto parameters = _simController->getSimulationParameters();
        if (ImGui::BeginChild(
                "##1",
                ImVec2(0, ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(90)),
                false,
                0)) {
            AlienImGui::Group("Instruction section");
            ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());

            auto dataSize = cell.cellFeature.constData.size();
            cell.cellFeature.constData.copy(_cellMemory, dataSize);
            auto maxDataSize = CellComputationCompiler::getMaxBytes(parameters);
            for (int i = dataSize; i < maxDataSize; ++i) {
                _cellMemory[i] = 0;
            }
             _cellDataMemoryEdit->DrawContents(reinterpret_cast<void*>(_cellMemory), maxDataSize);

            cell.cellFeature.constData = std::string(_cellMemory, maxDataSize);
            ImGui::PopFont();
        }
        ImGui::EndChild();
        if (ImGui::BeginChild("##2", ImVec2(0, 0), false, 0)) {
            AlienImGui::Group("Data section");
            ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
            auto dataSize = cell.cellFeature.volatileData.size();
            cell.cellFeature.volatileData.copy(_cellMemory, dataSize);
            _cellInstructionMemoryEdit->DrawContents(reinterpret_cast<void*>(_cellMemory), dataSize);
            cell.cellFeature.volatileData = std::string(_cellMemory, dataSize);
            ImGui::PopFont();
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellInOutTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem(ICON_FA_EXCHANGE_ALT " In/out channels", nullptr, ImGuiTabItemFlags_None)) {

        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showTokenTab(TokenDescription& token, int index)
{
    if (ImGui::BeginTabItem(("Token " + std::to_string(index + 1)).c_str(), nullptr, ImGuiTabItemFlags_None)) {
        auto parameters = _simController->getSimulationParameters();

        AlienImGui::Group("Properties");
        auto energy = toFloat(token.energy);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(MaxCellContentTextWidth), energy);
        token.energy = energy;

        AlienImGui::Group("Memory");
        auto dataSize = token.data.size();
        token.data.copy(_tokenMemory, dataSize);

        std::map<int, std::vector<std::string>> addressToSymbolNamesMap;
        auto const& symbolMap = _simController->getSymbolMap();
        for (auto const& [key, value] : symbolMap) {
            if (auto address = CellComputationCompiler::extractAddress(key)) {
                addressToSymbolNamesMap[*address].emplace_back(value);
            }
        }
        int currentMemoryEditIndex = 0;
        if (addressToSymbolNamesMap.empty() || addressToSymbolNamesMap.begin()->first != 0) {
            int numBytes = addressToSymbolNamesMap.empty() ? 256 : addressToSymbolNamesMap.begin()->first;
            showTokenMemorySection(0, numBytes, currentMemoryEditIndex);
        }

        boost::optional<int> lastAddress;
        for (auto const& [address, symbolNames] : addressToSymbolNamesMap) {
            if (lastAddress) {
                showTokenMemorySection(*lastAddress + 1, address - *lastAddress - 1, currentMemoryEditIndex);
            }
            showTokenMemorySection(address, 1, currentMemoryEditIndex);
            lastAddress = address;
        }

        if (!addressToSymbolNamesMap.empty()
            && addressToSymbolNamesMap.rbegin()->first < parameters.tokenMemorySize - 1) {
            auto lastAddress = addressToSymbolNamesMap.rbegin()->first;
            showTokenMemorySection(
                lastAddress + 1, parameters.tokenMemorySize - lastAddress - 1, currentMemoryEditIndex);
        }

        token.data = std::string(_tokenMemory, dataSize);
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showTokenMemorySection(int address, int numBytes, int& currentMemoryEditIndex)
{
    ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
    int height = ImGui::GetTextLineHeight() * ((numBytes + 7) / 8);
    ImGui::BeginChild(
        ("##TokenMemorySection" + std::to_string(currentMemoryEditIndex)).c_str(), ImVec2(0, height), false, 0);
    currentMemoryEditIndex++;
    _tokenMemoryEdits.at(currentMemoryEditIndex++)
        ->DrawContents(reinterpret_cast<void*>(&_tokenMemory[address]), numBytes, address);
    ImGui::EndChild();
    ImGui::PopFont();
    auto parameters = _simController->getSimulationParameters();
    if (address + numBytes < parameters.tokenMemorySize) {
        AlienImGui::Separator();
    }
}

void _InspectorWindow::showCompilationResult(CompilationResult const& compilationResult)
{
    ImGui::Text("Compilation result: ");
    if (compilationResult.compilationOk) {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(0.3, 1.0, 1.0));
        ImGui::SameLine();
        ImGui::Text("Success");
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)ImColor::HSV(0.05, 1.0, 1.0));
        ImGui::SameLine();
        ImGui::Text(("Error at line " + std::to_string(compilationResult.lineOfFirstError)).c_str());
        ImGui::PopStyleColor();
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto energy = toFloat(particle.energy);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters()
            .name("Energy")
            .textWidth(MaxParticleContentTextWidth),
        energy);
}

float _InspectorWindow::calcWindowWidth() const
{
    if (isCell()) {
        auto cell = std::get<CellDescription>(_editorModel->getInspectedEntity(_entityId));
        return StyleRepository::getInstance().scaleContent(280.0f + 50.0f * cell.tokens.size());
    }
    return StyleRepository::getInstance().scaleContent(280.0f);
}

void _InspectorWindow::addToken(CellDescription& cell)
{
    auto const& parameters = _simController->getSimulationParameters();

    cell.addToken(TokenDescription()
                      .setEnergy(parameters.tokenMinEnergy * 2)
                      .setData(std::string(parameters.tokenMemorySize, 0)));
}

