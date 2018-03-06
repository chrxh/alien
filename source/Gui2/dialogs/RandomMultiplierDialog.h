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

    int getNumber ();
    bool randomizeAngle ();
    qreal randomizeAngleMin ();
    qreal randomizeAngleMax ();
    bool randomizeVelX ();
    qreal randomizeVelXMin ();
    qreal randomizeVelXMax ();
    bool randomizeVelY ();
    qreal randomizeVelYMin ();
    qreal randomizeVelYMax ();
    bool randomizeAngVel ();
    qreal randomizeAngVelMin ();
    qreal randomizeAngVelMax ();

private:
    Ui::RandomMultiplierDialog *ui;
};
