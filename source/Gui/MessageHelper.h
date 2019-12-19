#pragma once
#include <QWidget>

class MessageHelper
{
public:
    static QWidget* getProgress(std::string message, QWidget* parent);
};