#pragma once

#include "Definitions.h"

class AlienWindow
{
public:
    AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn);
    virtual ~AlienWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

protected:
    virtual void processIntern() = 0;
    virtual void processBackground() {}
    virtual void processActivated() {}

    bool _sizeInitialized = false;
    bool _on = false;
    std::string _title; 
    std::string _settingsNode;
};