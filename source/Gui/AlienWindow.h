#pragma once

#include "Definitions.h"
#include "ShutdownInterface.h"

class AlienWindow : public ShutdownInterface
{
public:
    AlienWindow(std::string const& title, std::string const& settingsNode, bool defaultOn);

    void process();

    bool isOn() const;
    void setOn(bool value);

protected:
    virtual void shutdownIntern() {};
    virtual void processIntern() = 0;
    virtual void processBackground() {}
    virtual void processActivated() {}

    bool _sizeInitialized = false;
    bool _on = false;
    std::string _title; 
    std::string _settingsNode;

private:
    void shutdown() override;
};