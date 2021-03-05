#include "GridMultiplierDialog.h"
#include "ui_GridMultiplierDialog.h"

#include "Settings.h"
#include "StringHelper.h"

GridMultiplierDialog::GridMultiplierDialog(QVector2D centerPos, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GridMultiplierDialog)
{
    ui->setupUi(this);

    ui->initialPosXEdit->setText(QString("%1").arg(centerPos.x()));
    ui->initialPosYEdit->setText(QString("%1").arg(centerPos.y()));

	ui->velXCheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::GridMulChangeVelXKey, Const::GridMulChangeVelXDefault));
	ui->velYCheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::GridMulChangeVelYKey, Const::GridMulChangeVelYDefault));
	ui->angleCheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::GridMulChangeAngleKey, Const::GridMulChangeAngleDefault));
	ui->angVelCheckBox->setChecked(
		GuiSettings::getSettingsValue(Const::GridMulChangeAngVelKey, Const::GridMulChangeAngVelDefault));

	ui->initialVelXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulInitialVelXKey, Const::GridMulInitialVelXDefault)));
	ui->initialVelYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulInitialVelYKey, Const::GridMulInitialVelYDefault)));
	ui->initialAngleEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulInitialAngleKey, Const::GridMulInitialAngleDefault)));
	ui->initialAngVelEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulInitialAngVelKey, Const::GridMulInitialAngVelDefault)));

	ui->horizontalNumberEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulHorNumberKey, Const::GridMulHorNumberDefault)));
	ui->horIntervalEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulHorIntervalKey, Const::GridMulHorIntervalDefault)));
	ui->horVelXIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulHorVelXIncKey, Const::GridMulHorVelXIncDefault)));
	ui->horVelYIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulHorVelYIncKey, Const::GridMulHorVelYIncDefault)));
	ui->horAngleIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulHorAngleIncKey, Const::GridMulHorAngleIncDefault)));
	ui->horAngVelIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulHorAngVelIncKey, Const::GridMulHorAngVelIncDefault)));

	ui->verticalNumberEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulVerNumberKey, Const::GridMulVerNumberDefault)));
	ui->verIntervalEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulVerIntervalKey, Const::GridMulVerIntervalDefault)));
	ui->verVelXIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulVerVelXIncKey, Const::GridMulVerVelXIncDefault)));
	ui->verVelYIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulVerVelYIncKey, Const::GridMulVerVelYIncDefault)));
	ui->verAngleIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulVerAngleIncKey, Const::GridMulVerAngleIncDefault)));
	ui->verAngVelIncEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridMulVerAngVelIncKey, Const::GridMulVerAngVelIncDefault)));

	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &GridMultiplierDialog::okClicked);
}

GridMultiplierDialog::~GridMultiplierDialog()
{
    delete ui;
}

double GridMultiplierDialog::getInitialPosX () const
{
    bool ok(true);
    return ui->initialPosXEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getInitialPosY () const
{
    bool ok(true);
    return ui->initialPosYEdit->text().toDouble(&ok);
}

bool GridMultiplierDialog::isChangeVelocityX () const
{
    return ui->velXCheckBox->isChecked();
}

bool GridMultiplierDialog::isChangeVelocityY () const
{
    return ui->velYCheckBox->isChecked();
}

bool GridMultiplierDialog::isChangeAngle() const
{
    return ui->angleCheckBox->isChecked();
}

bool GridMultiplierDialog::isChangeAngularVelocity () const
{
    return ui->angVelCheckBox->isChecked();
}

double GridMultiplierDialog::getInitialVelX () const
{
    bool ok(true);
    return ui->initialVelXEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getInitialVelY () const
{
    bool ok(true);
    return ui->initialVelYEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getInitialAngle () const
{
    bool ok(true);
    return ui->initialAngleEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getInitialAngVel () const
{
    bool ok(true);
    return ui->initialAngVelEdit->text().toDouble(&ok);
}

int GridMultiplierDialog::getHorizontalNumber () const
{
    bool ok(true);
    return ui->horizontalNumberEdit->text().toUInt(&ok);
}

double GridMultiplierDialog::getHorizontalInterval () const
{
    bool ok(true);
    return ui->horIntervalEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getHorizontalVelocityXIncrement () const
{
    bool ok(true);
    return ui->horVelXIncEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getHorizontalVelocityYIncrement () const
{
    bool ok(true);
    return ui->horVelYIncEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getHorizontalAngleIncrement () const
{
    bool ok(true);
    return ui->horAngleIncEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getHorizontalAngularVelocityIncrement () const
{
    bool ok(true);
    return ui->horAngVelIncEdit->text().toDouble(&ok);
}

int GridMultiplierDialog::getVerticalNumber () const
{
    bool ok(true);
    return ui->verticalNumberEdit->text().toUInt(&ok);
}

double GridMultiplierDialog::getVerticalInterval () const
{
    bool ok(true);
    return ui->verIntervalEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getVerticalVelocityXIncrement () const
{
    bool ok(true);
    return ui->verVelXIncEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getVerticalVelocityYIncrement () const
{
    bool ok(true);
    return ui->verVelYIncEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getVerticalAngleIncrement () const
{
    bool ok(true);
    return ui->verAngleIncEdit->text().toDouble(&ok);
}

double GridMultiplierDialog::getVerticalAngularVelocityIncrement () const
{
    bool ok(true);
    return ui->verAngVelIncEdit->text().toDouble(&ok);
}

void GridMultiplierDialog::okClicked()
{
	GuiSettings::setSettingsValue(Const::GridMulChangeVelXKey, isChangeVelocityX());
	GuiSettings::setSettingsValue(Const::GridMulChangeVelYKey, isChangeVelocityY());
	GuiSettings::setSettingsValue(Const::GridMulChangeAngleKey, isChangeAngle());
	GuiSettings::setSettingsValue(Const::GridMulChangeAngVelKey, isChangeAngularVelocity());

	GuiSettings::setSettingsValue(Const::GridMulInitialVelXKey, getInitialVelX());
	GuiSettings::setSettingsValue(Const::GridMulInitialVelYKey, getInitialVelY());
	GuiSettings::setSettingsValue(Const::GridMulInitialAngleKey, getInitialAngle());
	GuiSettings::setSettingsValue(Const::GridMulInitialAngVelKey, getInitialAngVel());

	GuiSettings::setSettingsValue(Const::GridMulHorNumberKey, getHorizontalNumber());
	GuiSettings::setSettingsValue(Const::GridMulHorIntervalKey, getHorizontalInterval());
	GuiSettings::setSettingsValue(Const::GridMulHorVelXIncKey, getHorizontalVelocityXIncrement());
	GuiSettings::setSettingsValue(Const::GridMulHorVelYIncKey, getHorizontalVelocityYIncrement());
	GuiSettings::setSettingsValue(Const::GridMulHorAngleIncKey, getHorizontalAngleIncrement());
	GuiSettings::setSettingsValue(Const::GridMulHorAngVelIncKey, getHorizontalAngularVelocityIncrement());

	GuiSettings::setSettingsValue(Const::GridMulVerNumberKey, getVerticalNumber());
	GuiSettings::setSettingsValue(Const::GridMulVerIntervalKey, getVerticalInterval());
	GuiSettings::setSettingsValue(Const::GridMulVerVelXIncKey, getVerticalVelocityXIncrement());
	GuiSettings::setSettingsValue(Const::GridMulVerVelYIncKey, getVerticalVelocityYIncrement());
	GuiSettings::setSettingsValue(Const::GridMulVerAngleIncKey, getVerticalAngleIncrement());
	GuiSettings::setSettingsValue(Const::GridMulVerAngVelIncKey, getVerticalAngularVelocityIncrement());

	accept();
}

