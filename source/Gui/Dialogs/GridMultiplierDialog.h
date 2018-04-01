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

    double getInitialPosX () const;
    double getInitialPosY () const;
    bool isChangeVelocityX () const;
    bool isChangeVelocityY () const;
    bool isChangeAngle() const;
    bool isChangeAngularVelocity () const;
    double getInitialVelX () const;
    double getInitialVelY () const;
    double getInitialAngle () const;
    double getInitialAngVel () const;

    int getHorizontalNumber () const;
    double getHorizontalInterval () const;
    double getHorizontalVelocityXIncrement () const;
    double getHorizontalVelocityYIncrement () const;
    double getHorizontalAngleIncrement () const;
    double getHorizontalAngularVelocityIncrement () const;

    int getVerticalNumber () const;
    double getVerticalInterval () const;
    double getVerticalVelocityXIncrement () const;
    double getVerticalVelocityYIncrement () const;
    double getVerticalAngleIncrement () const;
    double getVerticalAngularVelocityIncrement () const;

private:
	Q_SLOT void okClicked();

private:
    Ui::GridMultiplierDialog *ui;
};
