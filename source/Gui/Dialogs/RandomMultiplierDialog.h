#pragma once

#include <QDialog>

namespace Ui {
class RandomMultiplierDialog;
}

class RandomMultiplierDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RandomMultiplierDialog(QWidget *parent = nullptr);
    virtual ~RandomMultiplierDialog();

    int getNumberOfCopies ();
    bool isChangeAngle ();
    double getAngleMin ();
    double getAngleMax ();
    bool isChangeVelX ();
    double getVelXMin ();
    double getVelXMax ();
    bool isChangeVelY ();
    double getVelYMin ();
    double getVelYMax ();
    bool isChangeAngVel ();
    double getAngVelMin ();
    double getAngVelMax ();

private:
	Q_SLOT void okClicked();

private:
    Ui::RandomMultiplierDialog *ui;
};
