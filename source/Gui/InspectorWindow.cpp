#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include <boost/algorithm/string.hpp>

#include "ImguiMemoryEditor/imgui_memory_editor.h"
#include "Fonts/IconsFontAwesome5.h"

#include "Base/StringHelper.h"
#include "EngineInterface/CellComputationCompiler.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"

using namespace std::string_literals;

namespace
{
    auto const MaxCellContentTextWidth = 120.0f;
    auto const MaxParticleContentTextWidth = 80.0f;
    auto const CellFunctions =
        std::vector{"Computation"s, "Propulsion"s, "Scanner"s, "Digestion"s, "Construction"s, "Sensor"s, "Muscle"s};
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
    _cellInstructionMemoryEdit = std::make_shared<MemoryEditor>();
    _cellInstructionMemoryEdit->OptShowOptions = false;
    _cellInstructionMemoryEdit->OptShowAscii = false;
    _cellInstructionMemoryEdit->OptMidColsCount = 0;
    _cellInstructionMemoryEdit->Cols = 8;

    _cellDataMemoryEdit = std::make_shared<MemoryEditor>();
    _cellDataMemoryEdit->OptShowOptions = false;
    _cellDataMemoryEdit->OptShowAscii = false;
    _cellDataMemoryEdit->OptMidColsCount = 0;
    _cellDataMemoryEdit->Cols = 8;

    auto const& parameters = _simController->getSimulationParameters();
    for (int i = 0; i < parameters.tokenMemorySize; ++i) {
        auto tokenMemoryEdit = std::make_shared<MemoryEditor>();
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
    if (ImGui::Begin(generateTitle().c_str(), &_on, ImGuiWindowFlags_HorizontalScrollbar)) {
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
    bool hasChanges(ParticleDescription const& left, ParticleDescription const& right)
    {
        return left.energy != right.energy;
    }
}

void _InspectorWindow::processCell(CellDescription cell)
{
    if (ImGui::BeginTabBar(
            "##CellInspect", /*ImGuiTabBarFlags_AutoSelectNewTabs | */ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        auto origCell = cell;
        showCellGeneralTab(cell);
        if (cell.cellFeature.getType() == Enums::CellFunction_Computation) {
            showCellCodeTab(cell);
            showCellMemoryTab(cell);
        } else {
            showCellInOutChannelTab(cell);
        }
        for (int i = 0; i < cell.tokens.size(); ++i) {
            showTokenTab(cell, i);
        }
        auto const& parameters = _simController->getSimulationParameters();
        if (cell.tokens.size() < parameters.cellMaxToken) {
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_SetSelected)) {
                addToken(cell);
            }
            AlienImGui::Tooltip("Add token");
        }
        ImGui::EndTabBar();

        //fill up with zeros
        cell.cellFeature.constData.append(
            std::max(std::basic_string<char>::size_type(0), CellComputationCompiler::getMaxBytes(parameters) - cell.cellFeature.constData.size()), 0);
        origCell.cellFeature.constData.append(
            std::max(std::basic_string<char>::size_type(0), CellComputationCompiler::getMaxBytes(parameters) - origCell.cellFeature.constData.size()), 0);  

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
        cell.cellFeature.setType(static_cast<Enums::CellFunction>(type));

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
            AlienImGui::CheckboxParameters().name("Block token").textWidth(MaxCellContentTextWidth), cell.tokenBlocked);

        AlienImGui::Group("Metadata");

        StringHelper::copy(_cellName, IM_ARRAYSIZE(_cellName), cell.metadata.name);

        AlienImGui::InputText(
            AlienImGui::InputTextParameters().name("Name").textWidth(MaxCellContentTextWidth),
            _cellName,
            IM_ARRAYSIZE(_cellName));
        cell.metadata.name = std::string(_cellName);

        StringHelper::copy(_cellDescription, IM_ARRAYSIZE(_cellDescription), cell.metadata.description);

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
                if (_lastCompilationResult) {
                    _lastCompilationResult->compilationOk = true;
                }
                return CellComputationCompiler::decompileSourceCode(
                    cell.cellFeature.constData,
                    _simController->getSymbolMap(),
                    _simController->getSimulationParameters());
            }
            return cell.metadata.computerSourcecode;
        }();
        StringHelper::copy(_cellCode, IM_ARRAYSIZE(_cellCode), origSourcecode);
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
            _lastCompilationResult =
                std::make_shared<CompilationResult>(
                CellComputationCompiler::compileSourceCode(sourcecode, _simController->getSymbolMap(), _simController->getSimulationParameters()));
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

void _InspectorWindow::showCellInOutChannelTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem(ICON_FA_EXCHANGE_ALT " In/out channels", nullptr, ImGuiTabItemFlags_None)) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(Const::InfoTextColor));
        AlienImGui::Text("This is a pure information tab.");
        ImGui::SameLine();
        AlienImGui::HelpMarker("The following table shows where the cell obtains their input and output from the token memory.");
        ImGui::PopStyleColor();
        
        if (ImGui::BeginTable(
                "##",
                2,
                ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter
                    | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Semantic", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();
            if (cell.cellFeature.getType() == Enums::CellFunction_Scanner) {
                showScannerTableContent();
            }
            if (cell.cellFeature.getType() == Enums::CellFunction_Digestion) {
                showDigestionTableContent();
            }
            if (cell.cellFeature.getType() == Enums::CellFunction_Constructor) {
                showConstructionTableContent();
            }
            if (cell.cellFeature.getType() == Enums::CellFunction_Muscle) {
                showMuscleTableContent();
            }
            ImGui::EndTable();
        }

        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showTokenTab(CellDescription& cell, int tokenIndex)
{
    bool open = true;
    if (ImGui::BeginTabItem(("Token " + std::to_string(tokenIndex + 1)).c_str(), &open, ImGuiTabItemFlags_None)) {
        auto parameters = _simController->getSimulationParameters();

        AlienImGui::Group("Properties");
        auto& token = cell.tokens.at(tokenIndex);
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
            if (auto address = CellComputationCompiler::extractAddress(value)) {
                addressToSymbolNamesMap[*address].emplace_back(key);
            }
        }
        int currentMemoryEditIndex = 0;
        if (addressToSymbolNamesMap.empty() || addressToSymbolNamesMap.begin()->first != 0) {
            int numBytes = addressToSymbolNamesMap.empty() ? 256 : addressToSymbolNamesMap.begin()->first;
            showTokenMemorySection(0, numBytes, currentMemoryEditIndex);
        }

        std::optional<int> lastAddress;
        for (auto const& [address, symbolNames] : addressToSymbolNamesMap) {
            if (lastAddress && address - *lastAddress > 1) {
                showTokenMemorySection(*lastAddress + 1, address - *lastAddress - 1, currentMemoryEditIndex);
            }
            showTokenMemorySection(address, 1, currentMemoryEditIndex, symbolNames);
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
    if (!open) {
        delToken(cell, tokenIndex);
    }
}

void _InspectorWindow::showTokenMemorySection(
    int address,
    int numBytes,
    int& currentMemoryEditIndex,
    std::vector<std::string> const& symbols)
{
    ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
    int height = ImGui::GetTextLineHeight() * ((numBytes + 7) / 8);
    ImGui::BeginChild(
        ("##TokenMemorySection" + std::to_string(currentMemoryEditIndex)).c_str(), ImVec2(0, height), false, 0);
    currentMemoryEditIndex++;
    _tokenMemoryEdits.at(currentMemoryEditIndex++)
        ->DrawContents(reinterpret_cast<void*>(&_tokenMemory[address]), numBytes, address);
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::SetCursorPosX(StyleRepository::getInstance().scaleContent(205.0f));

    auto text = !symbols.empty() ? boost::join(symbols, "\n") : std::string("unnamed block");
    AlienImGui::Text(text);
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
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::CompilationSuccessColor);
        ImGui::SameLine();
        ImGui::Text("Success");
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::CompilationErrorColor);
        ImGui::SameLine();
        AlienImGui::Text("Error at line " + std::to_string(compilationResult.lineOfFirstError));
        ImGui::PopStyleColor();
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto origParticle = particle;
    auto energy = toFloat(particle.energy);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters()
            .name("Energy")
            .textWidth(MaxParticleContentTextWidth),
        energy);

    particle.energy = energy;
    if (hasChanges(particle, origParticle)) {
        _simController->changeParticle(particle);
    }
}

namespace
{
    std::string formatHex(int value)
    {
        std::stringstream stream;
        stream << "0x" << std::hex << static_cast<int>(value);
        return stream.str();
    }
}

void _InspectorWindow::showScannerTableContent()
{
    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUTPUT));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output:");
    AlienImGui::Text(formatHex(Enums::ScannerOut::SUCCESS) + ": cell scanned");
    AlienImGui::Text(formatHex(Enums::ScannerOut::FINISHED) + ": scanning process completed");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::INOUT_CELL_NUMBER));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: number of the cell to be scanned");
    AlienImGui::Text("Output: number of the next cell");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUT_ENERGY));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output: energy of scanned cell");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUT_ANGLE));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output: relative angle of scanned cell");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUT_DISTANCE));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output: relative distance of scanned cell");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUT_CELL_MAX_CONNECTIONS));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output: max connections of scanned cell");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUT_CELL_BRANCH_NO));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output: branch number of scanned cell");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUT_CELL_METADATA));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output: color of scanned cell");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Scanner::OUT_CELL_FUNCTION));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output: specialization of scanned cell");
    AlienImGui::Text(formatHex(Enums::CellFunction_Computation) + ": Computation");
    AlienImGui::Text(formatHex(Enums::CellFunction_Scanner) + ": Scanner");
    AlienImGui::Text(formatHex(Enums::CellFunction_Digestion) + ": Digestion");
    AlienImGui::Text(formatHex(Enums::CellFunction_Constructor) + ": Construction");
    AlienImGui::Text(formatHex(Enums::CellFunction_Sensor) + ": Sensor");
    AlienImGui::Text(formatHex(Enums::CellFunction_Muscle) + ": Muscle");
    ImGui::Spacing();

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(
        formatHex(Enums::Scanner::OUT_CELL_FUNCTION_DATA) + " - "
        + formatHex(Enums::Scanner::OUT_CELL_FUNCTION_DATA + 48 + 16 + 1));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output:\ninternal data of scanned cell\n(e.g. cell code and cell memory");
}

void _InspectorWindow::showDigestionTableContent()
{
    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Digestion::OUTPUT));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output:");
    AlienImGui::Text(formatHex(Enums::DigestionOut::NO_TARGET) + ": no target cell found");
    AlienImGui::Text(formatHex(Enums::DigestionOut::STRIKE_SUCCESSFUL) + ": target cell found");
}

void _InspectorWindow::showConstructionTableContent()
{
    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::OUTPUT));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output:");
    AlienImGui::Text(formatHex(Enums::ConstrOut::SUCCESS) + ": construction of new cell was successful");
    AlienImGui::Text(formatHex(Enums::ConstrOut::ERROR_NO_ENERGY) + ": error - not enough energy");
    AlienImGui::Text(formatHex(Enums::ConstrOut::ERROR_CONNECTION) + ": error - no free connection");
    AlienImGui::Text(formatHex(Enums::ConstrOut::ERROR_LOCK) + ": error - construction blocked by other processes");
    AlienImGui::Text(formatHex(Enums::ConstrOut::ERROR_DIST) + ": error - no free connection");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::INPUT));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: main command");
    AlienImGui::Text(formatHex(Enums::ConstrIn::DO_NOTHING) + ": do nothing");
    AlienImGui::Text(formatHex(Enums::ConstrIn::CONSTRUCT) + ": try construct cell");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_OPTION));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: options");
    AlienImGui::Text(formatHex(Enums::ConstrInOption::STANDARD) + ": standard construction process");
    AlienImGui::Text(formatHex(Enums::ConstrInOption::CREATE_EMPTY_TOKEN) + ": construct cell and token with empty memory");
    AlienImGui::Text(formatHex(Enums::ConstrInOption::CREATE_DUP_TOKEN) + ": construct cell and token with copied memory");
    AlienImGui::Text(formatHex(Enums::ConstrInOption::FINISH_NO_SEP) + ": construct cell and finish construction process without separation");
    AlienImGui::Text(formatHex(Enums::ConstrInOption::FINISH_WITH_SEP) + ": construct cell and finish construction process with separation");
    AlienImGui::Text(formatHex(Enums::ConstrInOption::FINISH_WITH_EMPTY_TOKEN_SEP) + ": construct cell with empty token and finish construction process with separation");
    AlienImGui::Text(formatHex(Enums::ConstrInOption::FINISH_WITH_DUP_TOKEN_SEP) + ": construct cell with copied token and finish construction process with separation");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_ANGLE_ALIGNMENT));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: align relative angle");
    AlienImGui::Text(formatHex(0) + ": no alignment");
    AlienImGui::Text(formatHex(1) + ": align angle to multiples of 360 degrees");
    AlienImGui::Text(formatHex(2) + ": align angle to multiples of 180 degrees");
    AlienImGui::Text(formatHex(3) + ": align angle to multiples of 120 degrees");
    AlienImGui::Text(formatHex(4) + ": align angle to multiples of 90 degrees");
    AlienImGui::Text(formatHex(5) + ": align angle to multiples of 72 degrees");
    AlienImGui::Text(formatHex(6) + ": align angle to multiples of 60 degrees");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_UNIFORM_DIST));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: uniform distances");
    AlienImGui::Text(
        formatHex(Enums::ConstrInUniformDist::NO)
        + ": if constructed cell is connected to a nearby cell then\n    the spatial distance is taken as reference distance");
    AlienImGui::Text(
        formatHex(Enums::ConstrInUniformDist::YES)
        + ": if constructed cell is connected to a nearby cell then\n    the reference distance will be equal to the given\n    input distance at address"
        + formatHex(Enums::Constr::IN_DIST));

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::INOUT_ANGLE));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: angle");
    AlienImGui::Text("[the reference angle between this cell and the constructed\ncell and previous constructed cell and the constructed cell]");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_DIST));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: distance");
    AlienImGui::Text("[the reference distance between the constructed cell and the\nprevious constructed cell]");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_CELL_MAX_CONNECTIONS));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: max connections of constructed cell");
    AlienImGui::Text(formatHex(0) +  ": adapt max connections automatically");
    auto const& parameters = _simController->getSimulationParameters();
    AlienImGui::Text(formatHex(1) + " - " + formatHex(parameters.cellMaxBonds) + ": max connections");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_CELL_BRANCH_NO));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: branch number of constructed cell");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_CELL_METADATA));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: color of constructed cell");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_CELL_FUNCTION));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: specialization of constructed cell");
    AlienImGui::Text(formatHex(Enums::CellFunction_Computation) + ": Computation");
    AlienImGui::Text(formatHex(Enums::CellFunction_Scanner) + ": Scanner");
    AlienImGui::Text(formatHex(Enums::CellFunction_Digestion) + ": Digestion");
    AlienImGui::Text(formatHex(Enums::CellFunction_Constructor) + ": Construction");
    AlienImGui::Text(formatHex(Enums::CellFunction_Sensor) + ": Sensor");
    AlienImGui::Text(formatHex(Enums::CellFunction_Muscle) + ": Muscle");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Constr::IN_CELL_FUNCTION_DATA) + " - " + formatHex(Enums::Constr::IN_CELL_FUNCTION_DATA + 48 + 16 + 1));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input:");
    AlienImGui::Text("internal data of constructed cell\n(e.g. cell code and cell memory");
}

void _InspectorWindow::showMuscleTableContent()
{
    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Muscle::OUTPUT));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Output:");
    AlienImGui::Text(formatHex(Enums::MuscleOut::SUCCESS) + ": muscle activity was performed");
    AlienImGui::Text(formatHex(Enums::MuscleOut::LIMIT_REACHED) + ": no activity was performed since distance limit is reached");

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    AlienImGui::Text(formatHex(Enums::Muscle::INPUT));

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input:");
    AlienImGui::Text(formatHex(Enums::MuscleIn::DO_NOTHING) + ": do nothing");
    AlienImGui::Text(formatHex(Enums::MuscleIn::CONTRACT) + ": contract cell connection and produce impulse");
    AlienImGui::Text(formatHex(Enums::MuscleIn::CONTRACT_RELAX) + ": contract cell connection");
    AlienImGui::Text(formatHex(Enums::MuscleIn::EXPAND) + ": expand cell connection and produce impulse");
    AlienImGui::Text(formatHex(Enums::MuscleIn::EXPAND_RELAX) + ": expand cell connection");
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

void _InspectorWindow::delToken(CellDescription& cell, int index)
{
    cell.tokens.erase(cell.tokens.begin() + index);
}

