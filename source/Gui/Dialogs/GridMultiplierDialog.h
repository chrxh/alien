#pragma once

#include <QDialog>
#include <QVector2D>

namespace Ui {
class GridMultiplierDialog;
}

class GridMultiplierDialog
	: public QDialog
{
    Q_OBJECT

public:
    GridMultiplierDialog (QVector2D centerPos, QWidget *parent = nullptr);
    virtual ~GridMultiplierDialog();

    qreal getInitialPosX ();
    qreal getInitialPosY ();
    bool changeVelocityX ();
    bool changeVelocityY ();
    bool changeAngle();
    bool changeAngularVelocity ();
    qreal getInitialVelX ();
    qreal getInitialVelY ();
    qreal getInitialAngle ();
    qreal getInitialAngVel ();

    int getHorizontalNumber ();
    qreal getHorizontalInterval ();
    qreal getHorizontalVelocityXIncrement ();
    qreal getHorizontalVelocityYIncrement ();
    qreal getHorizontalAngleIncrement ();
    qreal getHorizontalAngularVelocityIncrement ();

    int getVerticalNumber ();
    qreal getVerticalInterval ();
    qreal getVerticalVelocityXIncrement ();
    qreal getVerticalVelocityYIncrement ();
    qreal getVerticalAngleIncrement ();
    qreal getVerticalAngularVelocityIncrement ();


private:
    Ui::GridMultiplierDialog *ui;
};
