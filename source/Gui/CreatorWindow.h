#pragma once

#include "Definitions.h"

class _CreatorWindow
{
public:
    _CreatorWindow();
    ~_CreatorWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    bool _on = false;
};