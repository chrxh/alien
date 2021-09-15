#pragma once

#include "Definitions.h"

class _SimulationParametersWindow
{
public:
    _SimulationParametersWindow(StyleRepository const& styleRepository);

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    void createGroup(std::string const& name);
    void createFloatItem(std::string const& name, float& value);
    void helpMarker(std::string const& text);

    StyleRepository _styleRepository;
    bool _on = false;
};