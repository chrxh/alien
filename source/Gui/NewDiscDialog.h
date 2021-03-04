#pragma once

#include <QDialog>
#include "ui_NewDiscDialog.h"

class NewDiscDialog : public QDialog
{
    Q_OBJECT

public:
    NewDiscDialog(QWidget *parent = Q_NULLPTR);
    ~NewDiscDialog();

    int getOuterRadius() const;
    int getInnerRadius() const;
    double getDistance() const;
    double getCellEnergy() const;
    int getColorCode() const;

private:
    bool validate() const;
    Q_SLOT void okClicked();

    Ui::NewDiscDialog ui;
};
