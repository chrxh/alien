#pragma once

#include "LocationWidgets.h"

class LocationWindow
{
public:
    void init(std::string const& title, LocationWidgets const& widgets);
    void process();

    bool isOn() const;

private:
    std::string _title;
    LocationWidgets _widgets;
    int _id = 0;
    bool _on = false;
};
