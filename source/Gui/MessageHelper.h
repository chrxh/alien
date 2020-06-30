#pragma once
#include <QWidget>

class MessageHelper
{
public:
    static QWidget* createProgressDialog(std::string message, QWidget* parent);
};