#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _SimulationParametersWindow
{
public:
    _SimulationParametersWindow(StyleRepository const& styleRepository, SimulationController const& simController);

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    void createGroup(std::string const& name);
    void createFloatItem(
        std::string const& name,
        float& value,
        float min,
        float max,
        bool logarithmic = false,
        std::string const& format = "%.3f");
    void createIntItem(std::string const& name, int& value, int min, int max);

    StyleRepository _styleRepository;
    SimulationController _simController;

    bool _on = false;
};