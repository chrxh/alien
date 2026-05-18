#include "SignalsBufferDialog.h"

#include <algorithm>
#include <ranges>

#include <imgui.h>

#include <EngineInterface/EngineConstants.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr DialogTextWidth = 160.0f;
}

SignalsBufferDialog::SignalsBufferDialog()
    : AlienDialog("Signal buffer", {500.0f, 500.0f})
{}

void SignalsBufferDialog::initIntern() {}

void SignalsBufferDialog::shutdownIntern() {}

void SignalsBufferDialog::openIntern() {}

void SignalsBufferDialog::open(std::vector<SignalEntryDesc> const& entries, std::function<void(std::vector<SignalEntryDesc> const&)> const& onAdoptCallback)
{
    _channelsBuffer.clear();
    _channelsBuffer.reserve(entries.size());
    for (auto const& entry : entries) {
        _channelsBuffer.emplace_back(entry._channels);
    }
    _selectedEntry = 0;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::open();
}

void SignalsBufferDialog::open(
    std::vector<SignalEntryGenomeDesc> const& entries,
    std::function<void(std::vector<SignalEntryGenomeDesc> const&)> const& onAdoptCallback)
{
    _channelsBuffer.clear();
    _channelsBuffer.reserve(entries.size());
    for (auto const& entry : entries) {
        _channelsBuffer.emplace_back(entry._channels);
    }
    _selectedEntry = 0;
    _onAdoptCallback = onAdoptCallback;
    AlienDialog::open();
}

void SignalsBufferDialog::processIntern()
{
    auto buttonAreaHeight = scale(50.0f);
    if (ImGui::BeginChild("SignalsBufferContent", ImVec2(0, -buttonAreaHeight), false)) {
        int numEntries = static_cast<int>(_channelsBuffer.size());
        if (AlienGui::InputInt(AlienGui::InputIntParameters().name("Number of signals").textWidth(DialogTextWidth), numEntries)) {
            numEntries = std::clamp(numEntries, 0, MAX_CELL_MEMORY_ENTRIES);
            _channelsBuffer.resize(numEntries, std::vector<float>(NEURONS_PER_CELL, 0.0f));
        }
        if (!_channelsBuffer.empty()) {
            std::vector<std::string> entryTexts;
            for (auto entry : std::views::iota(0, numEntries)) {
                entryTexts.emplace_back(std::to_string(entry));
            }
            _selectedEntry = std::clamp(_selectedEntry, 0, numEntries - 1);

            AlienGui::Switcher(AlienGui::SwitcherParameters().name("Edit signal").values(entryTexts).textWidth(DialogTextWidth), _selectedEntry);
            _selectedEntry = std::clamp(_selectedEntry, 0, numEntries - 1);

            AlienGui::BeginIndent();
            auto& channels = _channelsBuffer.at(_selectedEntry);
            if (static_cast<int>(channels.size()) < NEURONS_PER_CELL) {
                channels.resize(NEURONS_PER_CELL, 0.0f);
            }
            for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                AlienGui::SliderFloat(
                    AlienGui::SliderFloatParameters().name("#" + std::to_string(i + 1)).format("%.2f").textWidth(DialogTextWidth).min(-2.0f).max(2.0f),
                    &channels.at(i));
            }
            AlienGui::EndIndent();
        }
    }
    ImGui::EndChild();

    AlienGui::Separator();

    if (AlienGui::Button("Adopt")) {
        ImGui::CloseCurrentPopup();
        onAdopt();
        close();
    }
    ImGui::SetItemDefaultFocus();

    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
        close();
    }
}

void SignalsBufferDialog::onAdopt()
{
    if (!_onAdoptCallback) {
        return;
    }
    if (auto* descCallback = std::get_if<std::function<void(std::vector<SignalEntryDesc> const&)>>(&_onAdoptCallback.value())) {
        std::vector<SignalEntryDesc> result;
        result.reserve(_channelsBuffer.size());
        for (auto const& channels : _channelsBuffer) {
            SignalEntryDesc entry;
            entry._channels = channels;
            result.emplace_back(std::move(entry));
        }
        if (*descCallback) {
            (*descCallback)(result);
        }
    } else if (auto* genomeCallback = std::get_if<std::function<void(std::vector<SignalEntryGenomeDesc> const&)>>(&_onAdoptCallback.value())) {
        std::vector<SignalEntryGenomeDesc> result;
        result.reserve(_channelsBuffer.size());
        for (auto const& channels : _channelsBuffer) {
            SignalEntryGenomeDesc entry;
            entry._channels = channels;
            result.emplace_back(std::move(entry));
        }
        if (*genomeCallback) {
            (*genomeCallback)(result);
        }
    }
}
