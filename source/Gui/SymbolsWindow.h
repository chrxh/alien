#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SymbolMap.h"

#include "AlienWindow.h"
#include "Definitions.h"

class _SymbolsWindow : public _AlienWindow
{
public:
    _SymbolsWindow(SimulationController const& simController);

private:
    void processIntern() override;

    enum class SymbolType {
        Variable, Constant
    };
    struct Entry {
        std::string name;
        SymbolType type;
        std::string value;
    };
    std::vector<Entry> getEntriesFromSymbolMap(SymbolMap const& symbolMap) const;
    void updateSymbolMapFromEntries(std::vector<Entry> const& entries);

    bool hasSymbolMapChanged() const;
    bool isEditValid() const;
    SymbolType getSymbolType(std::string const& value) const;

    void onClearEditFields();
    void onEditEntry(Entry const& entry);
    void onAddEntry(std::vector<Entry>& entries, std::string const& name, std::string const& value) const;
    void onUpdateEntry(std::vector<Entry>& entries, std::string const& name, std::string const& value) const;
    void onDeleteEntry(std::vector<Entry>& entries, std::string const& name) const;

    std::string _origSymbolName;
    char _symbolName[256];
    char _symbolValue[256];

    SimulationController _simController;
    OpenSymbolsDialog _openSymbolsDialog;
    SaveSymbolsDialog _saveSymbolsDialog;

    enum class Mode
    {
        Edit, Create
    };
    Mode _mode = Mode::Create;
};